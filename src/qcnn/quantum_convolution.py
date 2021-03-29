import torch
import torch.nn as nn
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
        # convolution layer
        super(QuantumConv2d, self).__init__(in_channels,
                                            out_channels,
                                            kernel_size,
                                            stride=stride,
                                            padding=padding,
                                            dilation=dilation,
                                            groups=1,
                                            bias=False,
                                            padding_mode='zeros')

        # set/check quantum parameters
        self.set_quantum_params(quantum_eps, quantum_cap, quantum_ratio,
                                quantum_delta)

        # unfold operation
        self.unfold = nn.Unfold(kernel_size, dilation, padding, stride)

        # gradient operation
        self.weight.register_hook(self.simulate_quantum_gradient)

    def forward(self, input):

        # get convolutional layer output
        output = super(QuantumConv2d, self).forward(input)
        output = torch.clamp(output, 0., self.quantum_cap)

        # get quantum output
        quantum_output = self.simulate_quantum_output(input, output)

        # update convolutional layer output
        output.data = quantum_output.data

        return output

    def simulate_quantum_output(self, input, output):
        with torch.no_grad():
            if self.quantum_eps > 0.:
                # get kernel norm
                kernel_matrix = self.weight.data.flatten(start_dim=1)
                kernel_matrix = kernel_matrix.transpose(0, 1)
                kernel_norm = torch.norm(kernel_matrix, dim=0)
                kernel_norm = kernel_norm.repeat(input.size(0), 1, 1)

                # get input norm
                input_matrix = self.unfold(input)
                input_matrix = input_matrix.transpose(1, 2)
                input_norm = torch.norm(input_matrix, dim=2)
                input_norm = input_norm.unsqueeze(2)

                # add gaussian noise
                product_norm = torch.bmm(input_norm, kernel_norm)
                product_norm = product_norm.reshape(output.shape)
                noise = torch.randn(output.shape, device=output.device)
                output += 2 * self.quantum_eps * product_norm * noise
                output = torch.clamp(output, 0., self.quantum_cap)

            if self.quantum_ratio < 1.:
                # quantum sampling
                num_samples = int(self.quantum_ratio *
                                  output.shape[1:].numel())
                probs = output.flatten(start_dim=1)
                distribution = distributions.Categorical(probs=probs)
                samples = distribution.sample((num_samples, )).flatten()
                idxs = torch.arange(0, input.size(0))
                idxs = idxs.repeat(1, num_samples).flatten()
                mask = torch.zeros_like(probs)
                mask[idxs, samples] = 1.
                mask = mask.reshape(output.shape)
                output = mask * output
        return output

    def simulate_quantum_gradient(self, grad):
        if self.quantum_delta > 0.:
            # add quantum gradient estimation error to the kernel
            noise = torch.randn(grad.shape, device=grad.device)
            grad_norm = torch.norm(grad)
            error = self.quantum_delta * grad_norm * noise
            grad += error
        return grad

    def set_quantum_params(self,
                           quantum_eps=None,
                           quantum_cap=None,
                           quantum_ratio=None,
                           quantum_delta=None):
        if quantum_eps is not None:
            assert 0. <= quantum_eps <= 1., 'epsilon should verify: 0.<=eps<=1.'
            self.quantum_eps = quantum_eps
        if quantum_cap is not None:
            assert 0. < quantum_cap, 'cap should verify: 0.<cap'
            self.quantum_cap = quantum_cap
        if quantum_ratio is not None:
            assert 0. < quantum_ratio <= 1., 'ratio should verify: 0.<ratio<=1.'
            self.quantum_ratio = quantum_ratio
        if quantum_delta is not None:
            assert 0 <= quantum_delta <= 1., 'delta should verify: 0.<=delta<=1.'
            self.quantum_delta = quantum_delta

    def convert_to_classical(self):
        layer = nn.Conv2d(self.in_channels,
                          self.out_channels,
                          self.kernel_size,
                          self.stride,
                          self.padding,
                          self.dilation,
                          groups=1,
                          bias=False,
                          padding_mode='zeros')
        layer.weight.data = self.weight.data
        return layer

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
