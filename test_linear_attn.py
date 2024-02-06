import torch
import unittest

from model.linear_attention.linear_attention import get_mask, get_full_mask, linear_attn  # Adjust the import statement based on your file organization
from model.linear_attention.lightning_attn_interface import lightning_attn_func

class TestAttentionFunctions(unittest.TestCase):
    def test_get_mask_various_sizes(self):
        for n in range(1, 10):  # Testing various sizes from 1 to 9
            with self.subTest(n=n):
                mask = get_mask(n)
                self.assertEqual(mask.shape, (n, n), f"Mask shape should be ({n},{n})")
                self.assertTrue(torch.all(mask.triu(1) <= 0), "Upper triangle should have non-positive values")
                self.assertTrue(torch.all(mask.diag() == 1), "Diagonal values should be 1 after exp")

    def test_get_full_mask_various_sizes_and_slopes(self):
        for n in range(1, 6):  # Smaller range due to the complexity of the test
            slopes = torch.tensor([1, 2, 3])
            with self.subTest(n=n):
                full_mask = get_full_mask(n, slopes)
                expected_shape = (slopes.size(0), n, n)
                self.assertEqual(full_mask.shape, expected_shape, f"Full mask shape should be {expected_shape}")

    def test_linear_attn_various_shapes(self):
        for n in range(1, 6):  # Testing with a manageable range of sizes
            b, h, d, s = 2, 3, 4, torch.tensor([1.0])
            q = torch.rand(b, h, n, d)
            k = torch.rand(b, h, n, d)
            v = torch.rand(b, h, n, d)
            with self.subTest(n=n):
                output = linear_attn(q, k, v, s)
                expected_shape = (b, h, n, d)
                self.assertEqual(output.shape, expected_shape, f"Output shape should be {expected_shape}")

class TestAttentionMechanismComparison(unittest.TestCase):
    def test_attention_output_comparison(self):
        b, h, n, d, s = 1, 8, 8, 32, torch.tensor([1.0])
        q = torch.rand(b, h, n, d, dtype=torch.float32)
        k = torch.rand(b, h, n, d, dtype=torch.float32)
        v = torch.rand(b, h, n, d, dtype=torch.float32)

        # Generate outputs from both functions
        output_linear = linear_attn(q, k, v, s)

        if torch.cuda.is_available():
            q = q.cuda()
            k = k.cuda()
            v = v.cuda()
            s = s.cuda()
            # Now call your function
            output_lightning = lightning_attn_func(q, k, v, s)
            output_lightning = output_lightning.cpu()
        else:
            print("CUDA is not available. Test cannot be performed.")

        # Manual comparison to find the first differing element
        close = torch.isclose(output_linear, output_lightning, atol=0.1)
        if not torch.all(close):
            diff = torch.where(close == False)
            first_diff_index = (diff[0][0].item(), diff[1][0].item(), diff[2][0].item(), diff[3][0].item())  # Adjust based on tensor dimensions
            print(f"First differing element at index {first_diff_index}: linear={output_linear[first_diff_index].item()}, lightning={output_lightning[first_diff_index].item()}")
            assert False, "Outputs from linear_attn and lightning_attn_func should be close"
        else:
            assert True, "All elements are close enough"

if __name__ == '__main__':
    unittest.main()
