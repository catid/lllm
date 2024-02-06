import torch
import unittest

from model.linear_attention.lightning_attn_interface import lightning_attn_func

# Reference version:

def get_mask(n, slope=1):
    mask = torch.triu(torch.zeros(n, n).float().fill_(float("-inf")), 1)
    # -n, ..., -2, -1, 0
    for i in range(n):
        x = torch.arange(i + 1)
        y = slope * x
        mask[i, : i + 1] = -torch.flip(y, [0])

    return torch.exp(mask)


def get_full_mask(n, slopes):
    arr = []
    for slope in slopes:
        arr.append(get_mask(n, slope.item()))
    mask = torch.stack(arr, dim=0)

    return mask


def linear_attn(q, k, v, s):
    b, h, n, d = q.shape
    mask = get_full_mask(n, s).to(q.device).to(torch.float32)
    qk = torch.matmul(q, k.transpose(2, 3))
    qk = (qk.to(torch.float32) * mask).to(q.dtype)
    o = torch.matmul(qk, v)

    return o

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

import math

def _build_slope_tensor(n_attention_heads: int):
    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(
                n
            )  # In the paper, we only train models that have 2^a heads for some a. This function has
        else:  # some good properties that only occur when the input is a power of 2. To maintain that even
            closest_power_of_2 = 2 ** math.floor(
                math.log2(n)
            )  # when the number of heads is not a power of 2, we use this workaround.
            return (
                get_slopes_power_of_2(closest_power_of_2)
                + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
            )

    # h, 1, 1
    slopes = torch.tensor(get_slopes(n_attention_heads)).reshape(
        n_attention_heads, 1, 1
    )

    return slopes

from model.linear_attention.triton_lightning_attn2 import lightning_attn2

class TestAttentionMechanismComparison(unittest.TestCase):
    def test_attention_output_comparison(self):
        trials = [
            (6, 8, 256, 128, 64),
            (6, 8, 512, 128, 64),
            (6, 8, 1024, 128, 64),
            (6, 8, 2048, 128, 64),
            (6, 8, 4096, 128, 64),
            #(6, 8, 8192, 128, 64), OOM
            (6, 8, 2048, 32, 64),
            (6, 8, 2048, 64, 64),
            (6, 12, 2048, 128, 64),
            (6, 16, 2048, 128, 64),
            (6, 20, 2048, 128, 64),
            (1, 8, 2048, 128, 64),
            (2, 8, 2048, 128, 64),
            (3, 8, 2048, 128, 64),
            (6, 8, 913, 128, 64),
            (6, 8, 513, 128, 64),
            (6, 8, 1213, 128, 64),
            (6, 8, 2048, 16, 64),
        ]

        for trial in trials:
            torch.cuda.empty_cache()

            (b, h, n, d, e) = trial

            print(f"Testing: {trial}")

            device = torch.device("cuda")
            dtype = torch.float16
            q = (torch.randn((b, h, n, d), dtype=dtype, device=device) / 10).requires_grad_()
            k = (torch.randn((b, h, n, d), dtype=dtype, device=device) / 10).requires_grad_()
            v = (torch.randn((b, h, n, e), dtype=dtype, device=device) / 10).requires_grad_()
            do = torch.randn((b, h, n, e), dtype=dtype, device=device) / 10
            s = _build_slope_tensor(h).to(q.device).to(torch.float32)

            o_ref = linear_attn(q, k, v, s)
            o = lightning_attn2(q, k, v, s)

            # backward
            o_ref.backward(do, retain_graph=True)
            dq_ref, q.grad = q.grad.clone(), None
            dk_ref, k.grad = k.grad.clone(), None
            dv_ref, v.grad = v.grad.clone(), None

            o.backward(do, retain_graph=True)
            dq, q.grad = q.grad.clone(), None
            dk, k.grad = k.grad.clone(), None
            dv, v.grad = v.grad.clone(), None

            assert torch.norm(o - o_ref) < 0.2, f"torch.norm(o - o_ref) = {torch.norm(o - o_ref)}"
            assert torch.norm(dq - dq_ref) < 0.2, f"torch.norm(dq - dq_ref) = {torch.norm(dq - dq_ref)}"
            assert torch.norm(dk - dk_ref) < 0.2, f"torch.norm(dk - dk_ref) = {torch.norm(dk - dk_ref)}"
            assert torch.norm(dv - dv_ref) < 0.2, f"torch.norm(dv - dv_ref) = {torch.norm(dv - dv_ref)}"

            # Manual comparison to find the first differing element
            close = torch.isclose(o_ref, o, atol=0.1)
            if not torch.all(close):
                diff = torch.where(close == False)
                first_diff_index = (diff[0][0].item(), diff[1][0].item(), diff[2][0].item(), diff[3][0].item())  # Adjust based on tensor dimensions
                print(f"First differing element at index {first_diff_index}: linear={o_ref[first_diff_index].item()}, lightning={o[first_diff_index].item()}")
                assert False, "Outputs from linear_attn and lightning_attn_func should be close"
            else:
                assert True, "All elements are close enough"

if __name__ == '__main__':
    unittest.main()
