import torch
from torch import nn, autograd
import unittest

from model.util import SGLU

class TestSGLU(unittest.TestCase):
    def setUp(self):
        # Initialize parameters for the test
        self.d_in, self.mult, self.d_out, self.bias = 64, 4, 64, False
        self.batch_size = 10
        self.sglu = SGLU(d_in=self.d_in, mult=self.mult, d_out=self.d_out, bias=self.bias)
        self.x = torch.randn(self.batch_size, self.d_in, requires_grad=True)

    def test_forward_shape(self):
        # Run the forward pass of SGLU
        y = self.sglu(self.x)
        # Assert the output shape is as expected
        self.assertEqual(y.shape, torch.Size([self.batch_size, self.d_out]), "The output shape of SGLU does not match the expected shape.")

    def test_gradient_flow(self):
        # Run the forward pass
        y = self.sglu(self.x)
        # Dummy loss and backward pass to check gradients
        loss = y.sum()
        loss.backward()
        # Check if gradients are computed for input
        self.assertIsNotNone(self.x.grad, "Gradients were not computed for the input.")

    def test_stability_with_zero_inputs(self):
        # Zero input tensor
        x_zero = torch.zeros(self.batch_size, self.d_in, requires_grad=True)
        # Run the forward pass
        y_zero = self.sglu(x_zero)
        # Check for NaN or inf in the output
        self.assertFalse(torch.isnan(y_zero).any(), "Output contains NaN values for zero input.")
        self.assertFalse(torch.isinf(y_zero).any(), "Output contains inf values for zero input.")

    def test_different_configurations(self):
        # Test various configurations of bias and mult
        for bias in (True, False):
            for mult in (1, 2, 5):
                with self.subTest(bias=bias, mult=mult):
                    sglu = SGLU(d_in=self.d_in, mult=mult, d_out=self.d_out, bias=bias)
                    y = sglu(self.x)
                    # Assert the output shape is correct
                    self.assertEqual(y.shape, torch.Size([self.batch_size, self.d_out]),
                                     f"Output shape incorrect for bias={bias} and mult={mult}.")

if __name__ == '__main__':
    unittest.main()
