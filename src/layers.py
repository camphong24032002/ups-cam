import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ProbDenseInput(nn.Module):
    def __init__(self, in_features, out_features, activation='linear', use_bias=True):
        super(ProbDenseInput, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.activation = activation

        # Define the weight and bias parameters
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if self.use_bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize weights and biases using Kaiming uniform initialization
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            # Calculate the bound for bias initialization
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs):
        # Unpack inputs
        inputs, mask, k = inputs
        # print("Prob")
        # print("Inputs", inputs)
        # print("Mask", mask)
        # print("weight", self.linear.weight.T.shape)
        # print("K", k.shape)

        # Calculate ghost
        ghost = torch.ones_like(inputs) * (1.0 - mask)
        inputs_i = inputs * (1.0 - mask)

        avg_inputs = F.avg_pool2d(inputs, kernel_size=7).squeeze()
        avg_inputs_i = F.avg_pool2d(inputs_i, kernel_size=7).squeeze()
        avg_ghost = F.avg_pool2d(ghost, kernel_size=7).squeeze()

        # print("Prob", inputs.shape, inputs_i.shape, ghost.shape)

        # Dot product calculations
        dot = torch.matmul(avg_inputs, self.weight.T)
        dot_i = torch.matmul(avg_inputs_i, self.weight.T)
        dot_mask = torch.matmul(avg_ghost, torch.ones_like(self.weight.T))
        dot_v = torch.matmul(avg_inputs_i**2, self.weight.T**2)


        # Compute mean without feature i
        mu = dot_i / dot_mask
        v = dot_v / dot_mask - mu ** 2

        # Compensate for number of players in current coalition
        # k = k.unsqueeze(1)
        mu1 = mu * k

        # Compute mean of the distribution that also includes player i
        mu2 = mu1 + (dot - dot_i)

        # Compensate for number of players in the coalition
        v1 = v * k * (1.0 - (k-1) / (dot_mask - 1))

        # Set something different than 0 if necessary
        v1 = torch.maximum(torch.tensor(0.00001), v1)

        # Variance of the distribution that includes it is the same
        v2 = v1.clone()

        if self.use_bias:
            mu1 = mu1 + self.bias
            mu2 = mu2 + self.bias

        filtered_mu1, filtered_v1 = filter_activation(self.activation, mu1, v1)
        filtered_mu2, filtered_v2 = filter_activation(self.activation, mu2, v2)

        del dot, dot_i, dot_mask, dot_v, mu, v,
        del inputs, inputs_i, ghost, avg_inputs, avg_inputs_i, avg_ghost
        del mu1, v1, mu2, v2

        # return torch.stack([mu1, v1, mu2, v2], dim=-1)
        return filtered_mu1, filtered_v1, filtered_mu2, filtered_v2

def filter_activation(activation, mu, v):
    """
    Adjusts the mean (mu) and variance (v) based on the activation function.

    Args:
        activation (str): Name of the activation function ('linear' or 'relu').
        mu (Tensor): Mean tensor.
        v (Tensor): Variance tensor.

    Returns:
        Tuple[Tensor, Tensor]: Adjusted mean and variance tensors.
    """
    if activation in [None, 'linear']:
        # Linear activation does not change mean and variance
        return mu, v
    elif activation == 'relu':
        # For ReLU activation, adjust mean and variance
        std = torch.sqrt(v)
        alpha = mu / (std + 1e-6)
        # Compute the cumulative distribution function (CDF) and probability density function (PDF)
        cdf = 0.5 * (1 + torch.erf(alpha / math.sqrt(2)))
        pdf = torch.exp(-0.5 * alpha ** 2) / math.sqrt(2 * math.pi)

        mu_relu = std * pdf + mu * cdf
        v_relu = ((mu ** 2 + v) * cdf) - mu_relu ** 2
        del std, alpha, cdf, pdf
        return mu_relu, v_relu
    else:
        # For unsupported activations, raise an error
        raise NotImplementedError(f"Activation function '{activation}' is not supported.")
