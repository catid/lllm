import torch
from torch import nn
import unittest
import time

from model.util import SimpleRMSNorm
from model.linear_attention.srmsnorm import FastSimpleRMSNorm

class TestRMSNorm(unittest.TestCase):
    def setUp(self):
        # Initialize parameters for the test
        self.dim = 100
        self.cpu_norm = SimpleRMSNorm(self.dim)
        self.cuda_norm = FastSimpleRMSNorm(self.dim).cuda()
        self.batch_size, self.seq_len = 10, 20
        self.x_cpu = torch.randn(self.batch_size, self.seq_len, self.dim, requires_grad=True)
        self.x_cuda = self.x_cpu.clone().detach().cuda().requires_grad_(True)

    def test_forward_shape(self):
        # Test CPU
        y_cpu = self.cpu_norm(self.x_cpu)
        self.assertEqual(y_cpu.shape, self.x_cpu.shape, "The output shape of SimpleRMSNorm on CPU does not match the input shape.")
        
        # Test GPU
        y_cuda = self.cuda_norm(self.x_cuda)
        self.assertEqual(y_cuda.shape, self.x_cuda.shape, "The output shape of FastSimpleRMSNorm on CUDA does not match the input shape.")

    def test_output_match(self):
        # Ensure that the CPU and CUDA versions produce the same results
        y_cpu = self.cpu_norm(self.x_cpu)
        y_cuda = self.cuda_norm(self.x_cuda).cpu()
        
        # Due to potential small numerical differences between CPU and GPU calculations, a small tolerance is allowed
        self.assertTrue(torch.allclose(y_cpu, y_cuda, atol=1e-5, rtol=1e-3), "The outputs of CPU and CUDA versions do not match.")

    def test_gradient_flow(self):
        # Test CPU
        y_cpu = self.cpu_norm(self.x_cpu)
        loss_cpu = y_cpu.sum()
        loss_cpu.backward()
        self.assertIsNotNone(self.x_cpu.grad, "Gradients were not computed for the input on CPU.")

        # Test GPU
        y_cuda = self.cuda_norm(self.x_cuda)
        loss_cuda = y_cuda.sum()
        loss_cuda.backward()
        self.assertIsNotNone(self.x_cuda.grad, "Gradients were not computed for the input on CUDA.")

        # Ensure gradients are not None before comparison
        if self.x_cpu.grad is not None and self.x_cuda.grad is not None:
            # Check if gradients match
            gradients_match = torch.allclose(self.x_cpu.grad, self.x_cuda.grad.cpu(), atol=1e-5, rtol=1e-3)
            self.assertTrue(gradients_match, "Gradients do not match between CPU and CUDA.")
        else:
            # If either gradient is None, fail the test with a message
            self.fail("One of the gradients is None, indicating a failure in computing gradients.")

    def test_with_zero_input(self):
        # Test CPU
        x_zero_cpu = torch.zeros_like(self.x_cpu, requires_grad=True)
        y_zero_cpu = self.cpu_norm(x_zero_cpu)
        self.assertFalse(torch.isnan(y_zero_cpu).any(), "Output contains NaN values for zero input on CPU.")
        self.assertFalse(torch.isinf(y_zero_cpu).any(), "Output contains inf values for zero input on CPU.")

        # Test GPU
        x_zero_cuda = torch.zeros_like(self.x_cuda, requires_grad=True)
        y_zero_cuda = self.cuda_norm(x_zero_cuda)
        self.assertFalse(torch.isnan(y_zero_cuda).any(), "Output contains NaN values for zero input on CUDA.")
        self.assertFalse(torch.isinf(y_zero_cuda).any(), "Output contains inf values for zero input on CUDA.")

    def benchmark_norm(self, norm, x, device_name, batch_size):
        start_time = time.time()
        y = norm(x)
        torch.cuda.synchronize() if device_name == "CUDA" else None
        end_forward = time.time()
        loss = y.sum()
        loss.backward()
        torch.cuda.synchronize() if device_name == "CUDA" else None
        end_backward = time.time()

        if batch_size <= 2:
            # Do not print the first result (it has a bunch of initial latency that is not representative of real-world performance)
            return

        print(f"Batch Size: {batch_size}, {device_name} Forward Time: {end_forward - start_time:.6f} seconds")
        print(f"Batch Size: {batch_size}, {device_name} Backward Time: {end_backward - end_forward:.6f} seconds")

    def test_benchmark(self):
        for batch_size in [2**i for i in range(1, 16)]:  # Powers of two up to 32k
            print(f"\nBenchmarking with Batch Size: {batch_size}")
            x_cpu = torch.randn(batch_size, self.seq_len, self.dim, requires_grad=True)
            print("Benchmarking SimpleRMSNorm on CPU")
            self.benchmark_norm(self.cpu_norm, x_cpu, "CPU", batch_size)

            if torch.cuda.is_available():
                x_cuda = x_cpu.clone().detach().cuda().requires_grad_(True)
                print("Benchmarking FastSimpleRMSNorm on CUDA")
                self.benchmark_norm(self.cuda_norm, x_cuda, "CUDA", batch_size)
            else:
                print("CUDA is not available, skipping CUDA benchmark.")

if __name__ == '__main__':
    unittest.main()
