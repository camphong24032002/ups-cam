import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import ProbDenseInput
import adf


def keep_variance(x, min_variance):
    return x + min_variance

def convert_2_lpdn(model: nn.Module, convert_weights: bool = True) -> nn.Module:
    min_variance = 1e-3
    keep_variance_fn = lambda x: keep_variance(x, min_variance)
    adf_model = adf.Sequential()
    for name, module in (model._modules.items()):
        if len(list(module.children())) > 0:
            # recurse
            adf_model._modules[name] = convert_2_lpdn(module, convert_weights)
        else:
            if isinstance(module, nn.Conv2d):
                layer_new = adf.Conv2d(module.in_channels, module.out_channels, module.kernel_size, module.stride,
                                       module.padding, module.dilation, module.groups,
                                       module.bias is not None, module.padding_mode, keep_variance_fn=keep_variance_fn)
            elif isinstance(module, nn.Linear):
                layer_new = adf.Linear(module.in_features, module.out_features, module.bias is not None,
                                       keep_variance_fn=keep_variance_fn)
            elif isinstance(module, nn.ReLU):
                layer_new = adf.ReLU(keep_variance_fn=keep_variance_fn)
            elif isinstance(module, nn.LeakyReLU):
                layer_new = adf.LeakyReLU(negative_slope=module.negative_slope, keep_variance_fn=keep_variance_fn)
            elif isinstance(module, nn.Dropout):
                layer_new = adf.Dropout(module.p, keep_variance_fn=keep_variance_fn)
            elif isinstance(module, nn.MaxPool2d):
                layer_new = adf.MaxPool2d(keep_variance_fn=keep_variance_fn)
            elif isinstance(module, nn.ConvTranspose2d):
                layer_new = adf.ConvTranspose2d(module.in_channels, module.out_channels, module.kernel_size,
                                                module.stride, module.padding, module.output_padding, module.groups,
                                                module.bias, module.dilation, keep_variance_fn=keep_variance_fn)
            elif isinstance(module, nn.Hardswish):
                layer_new = adf.ReLU(keep_variance_fn=keep_variance_fn)
            else:
                raise NotImplementedError(f"Layer type {module} not supported")
            layer_old = module
            try:
                if convert_weights:
                    layer_new.weight = layer_old.weight
                    layer_new.bias = layer_old.bias
            except AttributeError:
                pass

            adf_model._modules[name] = layer_new

    return adf_model


class DASPModel(nn.Module):
    def __init__(self, first_layer, lpdn_model):
        super(DASPModel, self).__init__()
        self.first_layer = ProbDenseInput(first_layer.in_features, first_layer.out_features,
                                          use_bias=first_layer.bias is not None)
        self.lpdn_model = lpdn_model
        self.first_layer.weight = first_layer.weight
        self.first_layer.bias = first_layer.bias

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor, k: int):
        x1_mean, x1_var, x2_mean, x2_var = self.first_layer((inputs, mask, k))
        # print(self.lpdn_model)
        with torch.inference_mode():
            y1_mean, y1_var = self.lpdn_model(x1_mean, x1_var)
            y2_mean, y2_var = self.lpdn_model(x2_mean, x2_var)
        torch.cuda.empty_cache()  # Free unused cached memory

        res_1, res_2 = torch.stack([y1_mean, y1_var], -1), torch.stack([y2_mean, y2_var], -1)
        del y1_mean, y1_var, y2_mean, y2_var

        return res_1, res_2

    def release(self):
        del self.lpdn_model


class DASP:
    def __init__(self, model, player_generator=None, input_shape=None):
        self.model = model.to('cpu')
        self.player_generator = player_generator
        self.input_shape = input_shape
        self.inputs = None

        if self.input_shape is None:
            # In PyTorch, input shapes are not stored within the model.
            # You must provide the input shape explicitly.
            raise ValueError("Input shape must be provided explicitly in PyTorch.")
        else:
            logging.info(f"Inferred input shape: {self.input_shape}")

        self._build_dasp_model()

    def run(self, x, steps=None, batch_size=32):
        player_generator = self.player_generator
        player_generator.set_n_steps(steps)
        ks = player_generator.get_steps_list()

        logging.info(f"DASP: testing {len(ks)} coalition sizes:")
        if len(ks) < 10:
            logging.info(ks)
        else:
            logging.info(f"{ks[0]} {ks[1]} ... {ks[-2]} {ks[-1]}")

        result = None
        tile_input = [len(ks)] + [1] * (x.dim())
        tile_mask = [len(ks)] + [1] * (x.dim())

        for i, (mask, mask_output) in enumerate(player_generator):
            # Tiling and repeating inputs to match dimensions
            # print(mask.shape, mask_output.shape)
            x_tiled = x.repeat(tile_input)
            mask_tiled = mask.repeat(tile_mask)
            ks_tensor = torch.tensor(ks, dtype=torch.float32).unsqueeze(1)
            # print(mask_tiled.size(), mask_tiled.size(), ks_tensor.size())

            inputs = [x_tiled, mask_tiled, ks_tensor]
            # print("Before dasp", get_ram_usage())
            y1, y2 = self.dasp_model(*inputs)
            # print("After dasp", get_ram_usage())

            y1 = y1.view(len(ks), 1, -1, 2)
            y2 = y2.view(len(ks), 1, -1, 2)
            # print("Output", y1.shape, y2.shape)
            # raise
            y = torch.mean(y2[..., 0] - y1[..., 0], dim=0)
            # raise

            if torch.isnan(y).any():
                raise RuntimeError('Result contains NaNs! This should not happen...')

            # Compute Shapley Values as mean of all coalition sizes
            if result is None:
                result = torch.zeros(y.shape + mask_output.shape)

            # shape_mask = [1] * len(y.shape) + list(mask_output.shape)
            shape_out = list(y.shape) + [1] * len(mask_output.shape)

            y_reshaped = y.view(*shape_out)
            # mask_output_tensor = torch.tensor(mask_output)
            result += y_reshaped * mask_output
            # mask_output_tensor = None
            del x_tiled, mask_tiled, ks_tensor, y
        # print("Before release", get_ram_usage())
        # self.dasp_model.release()
        # del self.dasp_model
        # print("After release", get_ram_usage())

        return result
    

    def _convert_to_lpdn(self, model: nn.Module):
        return convert_2_lpdn(model, True)

    def _build_dasp_model(self):
        first_layer: nn.Linear = self.model[0]
        lpdn_model = self._convert_to_lpdn(self.model[1:])
        lpdn_model.noise_variance = 1e-3
        self.dasp_model = DASPModel(first_layer, lpdn_model=lpdn_model)
