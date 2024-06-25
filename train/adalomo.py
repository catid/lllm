import math

import torch
import torch.distributed as dist
from torch.optim import Optimizer

try:
    from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
except ImportError:
    from transformers.deepspeed import is_deepspeed_zero3_enabled

from transformers.utils import logging


class AdaLomo(Optimizer):
    """
    A custom optimizer class AdaLomo for gradient updates in distributed training.

    This class implements two gradient update functions :meth:`fuse_update` and :meth:`fuse_update_zero3`,
    used for non-ZeRO and ZeRO modes of gradient updates, respectively.

    :param model: The model to be optimized
    :param lr: Learning rate, default is 1e-3
    :param eps: Regularization coefficients. eps[0] prevents gradients from becoming too small,
                eps[1] prevents steps from being too large based on RMS scaling of parameters.
    :param clip_threshold: Threshold for normalizing the update matrix
    :param decay_rate: Decay rate for moving average of squared gradients
    :param clip_grad_norm: Norm threshold for gradient clipping

        .. note::

            clip_grad_norm must be positive.
    :param clip_grad_value: Value threshold for gradient clipping
    :param weight_decay: Weight decay coefficient, default is 0.0
    :param loss_scale: Loss scaling coefficient, used to enhance training precision, but too large may lead to NaNs
    """

    def __init__(
        self,
        model,
        lr=1e-3,
        loss_scale=2**10,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        clip_grad_norm=None,
        clip_grad_value=None,
        weight_decay=0.0,
    ):
        self.model = model
        self.lr = lr
        self.clip_grad_norm = clip_grad_norm
        self.clip_grad_value = clip_grad_value
        self.weight_decay = weight_decay
        self.loss_scale = loss_scale
        if self.weight_decay > 0.0:
            self.do_weight_decay = True
        else:
            self.do_weight_decay = False
        self.eps = eps
        self.step_num = 0
        self.decay_rate = decay_rate
        self.clip_threshold = clip_threshold

        # for grad norm
        if self.clip_grad_norm is not None and self.clip_grad_norm <= 0:
            raise ValueError(
                f"clip_grad_norm should be positive, got {self.clip_grad_norm}."
            )
        self.gather_norm = False
        self.grad_norms = []
        self.clip_coef = None

        # check if zero3 is enabled
        self.zero3_enabled = is_deepspeed_zero3_enabled()
        print(f"TEST: self.zero3_enabled = {self.zero3_enabled}")
        if self.zero3_enabled:  # zero3 is enabled
            self.grad_func = self.fuse_update_zero3()
        else:
            self.grad_func = self.fuse_update()

        self.exp_avg_sq = {}
        self.exp_avg_sq_row = {}
        self.exp_avg_sq_col = {}

        # register hook function, which will be called through the backward process
        for n, p in self.model.named_parameters():
            if self.zero3_enabled:
                if len(p.ds_shape) == 1:
                    self.exp_avg_sq[n] = torch.zeros(p.ds_shape[0], dtype=torch.float32).cuda()
                else:
                    self.exp_avg_sq_row[n] = torch.zeros(p.ds_shape[0], dtype=torch.float32).cuda()
                    self.exp_avg_sq_col[n] = torch.zeros(p.ds_shape[1], dtype=torch.float32).cuda()
            else:
                if len(p.data.shape) == 1:
                    self.exp_avg_sq[n] = torch.zeros(p.data.shape[0], dtype=torch.float32).cuda()
                else:
                    self.exp_avg_sq_row[n] = torch.zeros(p.data.shape[0], dtype=torch.float32).cuda()
                    self.exp_avg_sq_col[n] = torch.zeros(p.data.shape[1], dtype=torch.float32).cuda()

            if p.requires_grad:
                p.register_hook(self.grad_func)
        defaults = dict(
            lr=lr,
            eps=eps,
            weight_decay=weight_decay,
            clip_grad_norm=clip_grad_norm,
            clip_grad_value=clip_grad_value,
        )
        super(AdaLomo, self).__init__(self.model.parameters(), defaults)

    @staticmethod
    def _approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col):
        # copy from fairseq's adafactor implementation:
        # https://github.com/huggingface/transformers/blob/8395f14de6068012787d83989c3627c3df6a252b/src/transformers/optimization.py#L505
        r_factor = (
            (exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True))
            .rsqrt_()
            .unsqueeze(-1)
        )
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)

    @staticmethod
    def _rms(tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    def fuse_update(self):
        """
        Update model parameters' gradients in non-ZeRO mode.

        :return: func, a closure function to update model parameters' gradients
        """

        def func(x):
            """
            Closure function to update model parameters' gradients.
            """
            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        grad_fp32 = p.grad.to(torch.float32)
                        p.grad = None
                        if self.loss_scale:
                            grad_fp32.div_(self.loss_scale)
                        if self.gather_norm:
                            # we adopt two backward pass for gradient norm computation and parameter update, respectively.
                            self.grad_norms.append(torch.norm(grad_fp32, 2.0))
                        else:
                            # grad clip or norm
                            if (
                                self.clip_grad_value is not None
                                and self.clip_grad_value > 0
                            ):
                                # Clipping gradients by their value
                                grad_fp32.clamp_(
                                    min=-self.clip_grad_value, max=self.clip_grad_value
                                )
                            if (
                                self.clip_grad_norm is not None
                                and self.clip_grad_norm > 0
                                and self.clip_coef is not None
                            ):
                                # Normalize the gradient according to its norm (computed in another pass)
                                grad_fp32.mul_(self.clip_coef)

                            # To avoid math errors for edge cases
                            if self.step_num == 0 and self.decay_rate < 0:
                                decay_rate = - self.decay_rate
                            else:
                                decay_rate = self.decay_rate

                            beta2t = 1.0 - math.pow(self.step_num, decay_rate)
                            update = (grad_fp32**2) + self.eps[0]

                            if len(p.data.shape) > 1:
                                self.exp_avg_sq_row[n].mul_(beta2t).add_(
                                    update.mean(dim=-1), alpha=1.0 - beta2t
                                )
                                self.exp_avg_sq_col[n].mul_(beta2t).add_(
                                    update.mean(dim=-2), alpha=1.0 - beta2t
                                )
                                update = self._approx_sq_grad(
                                    self.exp_avg_sq_row[n], self.exp_avg_sq_col[n]
                                )
                                update.mul_(grad_fp32)
                            else:
                                self.exp_avg_sq[n].mul_(beta2t).add_(
                                    update, alpha=1.0 - beta2t
                                )
                                update = self.exp_avg_sq[n].rsqrt().mul_(grad_fp32)

                            update.div_(
                                (self._rms(update) / self.clip_threshold).clamp_(
                                    min=1.0
                                )
                            )

                            p_fp32 = p.data.to(torch.float32)
                            p_rms = torch.norm(p_fp32, 2.0) / math.sqrt(p.numel())
                            lr = self.lr
                            param_scale = max(self.eps[1], p_rms)
                            lr = lr * param_scale

                            if self.do_weight_decay:
                                p_fp32.mul_(1.0 - lr * self.weight_decay)
                            p_fp32.add_(update, alpha=-lr)
                            p.data.copy_(p_fp32)

            return x

        return func

    def fuse_update_zero3(self):
        """
        Update model parameters' gradients in ZeRO mode.

        :return: func, a closure function to update model parameters' gradients
        """

        def func(x):
            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    if p.grad is not None:
                        torch.distributed.all_reduce(
                            p.grad, op=torch.distributed.ReduceOp.AVG, async_op=False
                        )

                        grad_fp32 = p.grad.to(torch.float32)
                        p.grad = None
                        if self.loss_scale:
                            grad_fp32.div_(self.loss_scale)

                        if self.gather_norm:
                            # we adopt two backward pass for gradient norm computation and parameter update, respectively.
                            self.grad_norms.append(torch.norm(grad_fp32, 2.0))
                        else:  # update param
                            partition_size = p.ds_tensor.numel()
                            start = partition_size * self.dp_rank
                            end = min(start + partition_size, grad_fp32.numel())

                            if self.clip_grad_value is not None:
                                # Clipping gradients by their value
                                grad_fp32.clamp_(
                                    min=-self.clip_grad_value, max=self.clip_grad_value
                                )
                            if (
                                self.clip_grad_norm is not None
                                and self.clip_grad_norm > 0
                                and self.clip_coef is not None
                            ):
                                # Normalize the gradient according to its norm (computed in another pass)
                                grad_fp32.mul_(self.clip_coef)

                            # To avoid math errors for edge cases
                            if self.step_num == 0 and self.decay_rate < 0:
                                decay_rate = - self.decay_rate
                            else:
                                decay_rate = self.decay_rate
                            beta2t = 1.0 - math.pow(self.step_num, decay_rate)
                            update = (grad_fp32**2) + self.eps[0]  # Change to addcmul_

                            if len(p.ds_shape) > 1:
                                self.exp_avg_sq_row[n].mul_(beta2t).add_(
                                    update.mean(dim=-1), alpha=1.0 - beta2t
                                )
                                self.exp_avg_sq_col[n].mul_(beta2t).add_(
                                    update.mean(dim=-2), alpha=1.0 - beta2t
                                )
                                update = self._approx_sq_grad(
                                    self.exp_avg_sq_row[n], self.exp_avg_sq_col[n]
                                )
                                update.mul_(grad_fp32)
                            else:
                                self.exp_avg_sq[n].mul_(beta2t).add_(
                                    update, alpha=1.0 - beta2t
                                )
                                update = self.exp_avg_sq[n].rsqrt().mul_(grad_fp32)

                            update.div_(
                                (self._rms(update) / self.clip_threshold).clamp_(
                                    min=1.0
                                )
                            )

                            one_dim_update = update.view(-1)
                            partitioned_update = one_dim_update.narrow(
                                0, start, end - start
                            )
                            param_fp32 = p.ds_tensor.to(torch.float32)
                            partitioned_p = param_fp32.narrow(0, 0, end - start)

                            p_rms = torch.norm(partitioned_p, 2.0) ** 2
                            dist.all_reduce(p_rms, op=torch.distributed.ReduceOp.SUM)
                            p_rms = (p_rms / p.ds_numel).sqrt()

                            lr = self.lr
                            param_scale = max(self.eps[1], p_rms)
                            lr = lr * param_scale

                            if self.do_weight_decay:
                                partitioned_p.mul_(1.0 - lr * self.weight_decay)
                            partitioned_p.add_(partitioned_update, alpha=-lr)
                            p.ds_tensor[: end - start] = partitioned_p

            return x

        return func

    def fused_backward(self, loss, lr):
        """
        Perform one step of backward pass and update gradients of the model.

        :param loss: Loss value of the model
        :param lr: Learning rate
        """
        self.lr = lr
        if self.loss_scale:
            loss = loss * self.loss_scale
        self.step_num += 1
        loss.backward()
        # update the last parameter since the last parameter in the computation graph is not ready when calling hook functions
        # the argument of grad_func is just a placeholder, and it can be anything.
        self.grad_func(0)

    def grad_norm(self, loss):
        """
        Calculate the norm of gradients.

        :param loss: Loss value of the model
        """
        self.gather_norm = True
        self.grad_norms = []
        if self.loss_scale:
            loss = loss * self.loss_scale
        loss.backward(retain_graph=True)
        # update the last parameter since the last parameter in the computation graph is not ready when calling hook functions
        # the argument of grad_func is just a placeholder, and it can be anything.
        self.grad_func(0)

        with torch.no_grad():
            # The norm is computed over all gradients together, as if they were
            # concatenated into a single vector. Gradients are modified in-place.
            self.grad_norms = torch.stack(self.grad_norms)

            total_norm = torch.norm(self.grad_norms, 2.0)
            self.clip_coef = float(self.clip_grad_norm) / (total_norm + 1e-6)
            self.clip_coef = torch.clamp(self.clip_coef, max=1.0)
        self.gather_norm = False
