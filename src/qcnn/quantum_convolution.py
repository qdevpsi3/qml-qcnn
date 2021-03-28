import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions


class QuantumConv2d(nn.Conv2d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 quantum_eps,
                 quantum_cap,
                 quantum_ratio,
                 quantum_delta,
                 stride=1,
                 padding=0,
                 dilation=1):
        """Base class for quantum convolutional layer.
        
        Args:
            in_channels (int): number of input channels.
            out_channels (int): number of output channels.
            kernel_size (int): size of the convolution kernel.
            quantum_eps (float): precision of quantum multiplication.
            quantum_cap (float): value for cap 'relu' activation function.
            quantum_ratio (float): precision of quantum tomography.
            quantum_delta (float): precision of quantum gradient estimation.
            stride (int, optional): convolution stride. Defaults to 1.
            padding (int, optional): convolution padding. Defaults to 0.
            dilation (int, optional): convolution dilation. Defaults to 1.
        """
        super(QuantumConv2d, self).__init__(in_channels,
                                            out_channels,
                                            kernel_size,
                                            stride=stride,
                                            padding=padding,
                                            dilation=dilation,
                                            groups=1,
                                            bias=False,
                                            padding_mode='zeros')

        # quantum parameters
        self.quantum_eps = quantum_eps
        self.quantum_ratio = quantum_ratio
        self.quantum_cap = quantum_cap

        # add quantum gradient estimation error to the kernel
        def get_quantum_gradient(grad):
            noise = quantum_delta * torch.randn(grad.shape, device=grad.device)
            grad_norm = torch.norm(grad)
            error = noise * grad_norm
            grad += error
            return grad

        self.weight.register_hook(get_quantum_gradient)

    def extra_repr(self):
        s = '{in_channels}, {out_channels}, kernel_size={kernel_size}, '
        s += 'quantum_eps={quantum_eps}, quantum_cap = {quantum_cap}, '
        s += 'quantum_ratio={quantum_ratio}, quantum_delta = {quantum_delta}, '
        s += 'stride={stride}'
        if self.padding != (0, ) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1, ) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

    def forward(self, input):

        # get convolutional layer output
        output = super(QuantumConv2d, self).forward(input)
        output = torch.clamp(output, 0., self.quantum_cap)

        # get quantum output
        quantum_output = self.get_quantum_output(input, output)

        # update convolutional layer output
        output.data = quantum_output.data

        return output

    def get_quantum_output(self, input, output):
        with torch.no_grad():
            # get kernel norm
            kernel_matrix = self.weight.data.flatten(start_dim=1).transpose(
                0, 1)
            kernel_norm = torch.norm(kernel_matrix,
                                     dim=0).repeat(input.size(0), 1, 1)

            # get input norm
            input_matrix = torch.nn.functional.unfold(
                input,
                self.kernel_size,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation).transpose(1, 2)
            input_norm = torch.norm(input_matrix, dim=2).unsqueeze(2)

            # add gaussian noise
            vars = torch.bmm(input_norm, kernel_norm).reshape(output.shape)
            noise = self.quantum_eps * torch.randn(output.shape).to(
                vars.device)
            errors = 2 * torch.mul(vars, noise)
            output = output + errors
            output = torch.clamp(output, 0., self.quantum_cap)

            # quantum sampling
            num_samples = int(self.quantum_ratio * output.shape[1:].numel())
            probs = output.flatten(start_dim=1)
            distribution = torch.distributions.Categorical(probs=probs)
            samples = distribution.sample((num_samples, )).flatten()
            idxs = torch.arange(0,
                                input.size(0)).repeat(1,
                                                      num_samples).flatten()
            mask = torch.zeros_like(probs)
            mask[idxs, samples] = 1.
            mask = mask.reshape(output.shape)
            output = mask * output
        return output
