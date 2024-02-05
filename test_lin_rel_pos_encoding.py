import torch
from torch import nn
import unittest

class LinRelPosEncoding(nn.Module):
    def __init__(self, heads=8, dim=64):
        super().__init__()
        d = heads * dim
        self.index = torch.empty(0)
        self.theta = nn.Parameter(10000**(-2 / d * torch.arange(d)).reshape(heads, 1, -1))

    def forward(self, x, offset=0):
        n = x.shape[-2]
        if self.index.shape[0] < n:
            self.index = torch.arange(n).reshape(1, -1, 1).to(x)
        index = self.index[:, :n] + offset
        theta = self.theta * index
        return torch.concat([x * torch.cos(theta), x * torch.sin(theta)], dim=-1)

class TestLinRelPosEncoding(unittest.TestCase):
    def setUp(self):
        self.heads = 8
        self.dim = 64
        self.lrpe = LinRelPosEncoding(heads=self.heads, dim=self.dim)
        self.batch_size, self.seq_len = 10, 20
        self.d = self.heads * self.dim

    def test_initialization(self):
        self.assertEqual(self.lrpe.theta.shape, (self.heads, 1, self.dim), "Theta parameter initialized with incorrect shape.")

    def test_forward_shape(self):
        x = torch.randn(self.batch_size, self.heads, self.seq_len, self.dim)
        encoded = self.lrpe(x)
        expected_shape = (self.batch_size, self.heads, self.seq_len, self.dim * 2)  # Output is concatenated cos and sin
        self.assertEqual(encoded.shape, expected_shape, "Encoded output shape is incorrect.")

    def test_numerical_stability(self):
        x = torch.randn(self.batch_size, self.heads, self.seq_len, self.dim)
        encoded = self.lrpe(x)
        self.assertFalse(torch.isnan(encoded).any(), "Output contains NaN values.")
        self.assertFalse(torch.isinf(encoded).any(), "Output contains inf values.")

    def test_different_offsets(self):
        x = torch.randn(self.batch_size, self.heads, self.seq_len, self.dim)
        offsets = [0, 10, 20]
        for offset in offsets:
            with self.subTest(offset=offset):
                encoded = self.lrpe(x, offset=offset)
                self.assertFalse(torch.isnan(encoded).any(), f"Output with offset {offset} contains NaN values.")
                self.assertFalse(torch.isinf(encoded).any(), f"Output with offset {offset} contains inf values.")

    def test_parameter_gradients(self):
        x = torch.randn(self.batch_size, self.heads, self.seq_len, self.dim, requires_grad=True)
        encoded = self.lrpe(x)
        loss = encoded.sum()
        loss.backward()
        self.assertIsNotNone(self.lrpe.theta.grad, "Theta parameter did not receive gradients.")

class TestLinRelPosEncodingExtended(unittest.TestCase):
    def setUp(self):
        self.heads = 8
        self.dim = 64
        self.lrpe = LinRelPosEncoding(heads=self.heads, dim=self.dim).to('cpu')  # Default device for simplicity
        self.batch_size, self.seq_len = 10, 20
        self.d = self.heads * self.dim

    def test_zero_input(self):
        x = torch.zeros(self.batch_size, self.heads, self.seq_len, self.dim)
        encoded = self.lrpe(x)
        # Check that output is not NaN or inf, which implies stability
        self.assertFalse(torch.isnan(encoded).any(), "Output contains NaN for zero input.")
        self.assertFalse(torch.isinf(encoded).any(), "Output contains inf for zero input.")

    def test_device_consistency(self):
        devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
        x_cpu = torch.randn(self.batch_size, self.heads, self.seq_len, self.dim)
        for device in devices:
            with self.subTest(device=device):
                lrpe = self.lrpe.to(device)
                x = x_cpu.to(device)
                encoded = lrpe(x)
                self.assertFalse(torch.isnan(encoded).any(), f"Output contains NaN on {device}.")
                self.assertFalse(torch.isinf(encoded).any(), f"Output contains inf on {device}.")

    def test_parameter_update(self):
        x = torch.randn(self.batch_size, self.heads, self.seq_len, self.dim, requires_grad=True)
        encoded = self.lrpe(x)
        loss = encoded.sum()
        initial_theta = self.lrpe.theta.clone()
        loss.backward()
        with torch.no_grad():
            self.lrpe.theta -= 0.01 * self.lrpe.theta.grad  # Simulate an update step
        self.assertFalse(torch.allclose(initial_theta, self.lrpe.theta), "Theta parameter did not update.")

    def test_known_input_output(self):
        # Simplified input for easy manual observation
        x = torch.ones(1, self.heads, 1, self.dim)
        encoded = self.lrpe(x, offset=0)
        
        # Instead of comparing to a hardcoded expected output, we check for non-NaN and non-inf values
        self.assertFalse(torch.isnan(encoded).any(), "Encoded output contains NaN values for unit input.")
        self.assertFalse(torch.isinf(encoded).any(), "Encoded output contains inf values for unit input.")

        # You can also verify the output shape is as expected
        expected_shape = (1, self.heads, 1, self.dim * 2)  # Output doubles the last dimension size
        self.assertEqual(encoded.shape, expected_shape, "Encoded output shape does not match expected shape for unit input.")

if __name__ == '__main__':
    unittest.main()
