import numpy as np
import torch
import torch.nn.functional as F
from utils import scale_cam_image, release_list

class ActivationsAndGradients:
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.gradients = []
        self.activations = []
        self.handles = []
        for target_layer in target_layers:
            self.handles.append(
                target_layer.register_forward_hook(self.save_activation))
            self.handles.append(
                target_layer.register_forward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        activation = output
        self.activations.append(activation.cpu().detach())

    def save_gradient(self, module, input, output):
        if not hasattr(output, "requires_grad") or not output.requires_grad:
            return
        def _store_grad(grad):
            self.gradients = [grad.cpu().detach()] + self.gradients

        output.register_hook(_store_grad)

    def __call__(self, x):
        self.gradients = []
        self.activations = []
        return self.model(x)

    def release(self):
        for handle in self.handles:
            handle.remove()

def get_grad_cam(model, input_tensor):
    target_layers = [model.features[-1]]

    activations_and_grads = ActivationsAndGradients(
                model, target_layers)
    input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)
    outputs = activations_and_grads(input_tensor)
    np_outputs = outputs.cpu().detach().numpy()
    target_classes = np.argmax(np_outputs, axis=1)

    model.zero_grad()
    loss = sum([output[target_class]
                for output, target_class in zip(outputs, target_classes)])
    loss.backward(retain_graph=True)
    
    activations_list = [a.cpu().data.numpy()
                        for a in activations_and_grads.activations]
    grads_list = [g.cpu().data.numpy()
                  for g in activations_and_grads.gradients]

    layer_activations = activations_list[0]
    layer_grads = grads_list[0]

    # get_cam_image

    weights = np.mean(layer_grads, axis=(2, 3))

    weighted_activations = weights[:, :, None, None] * layer_activations
    cam = weighted_activations.sum(axis=1)
    relu_cam = np.maximum(cam, 0)
    grad_cam = scale_cam_image(relu_cam)
    np_outputs = None
    layer_activations = None
    layer_grads = None
    del target_layers
    activations_and_grads.release()
    del activations_and_grads
    release_list(activations_list)
    release_list(grads_list)
    target_classes = None
    weights = None
    weighted_activations = None
    cam = None
    relu_cam = None
    return grad_cam

