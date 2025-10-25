"""
Credit: laksjdjf
https://github.com/laksjdjf/cgem156-ComfyUI/blob/main/scripts/attention_couple/node.py

Modified by. Haoming02 to work with Forge
"""

import math
from functools import wraps
from typing import Callable

import torch
from modules.devices import device, dtype

from lib_couple.logging import logger

from .attention_masks import capped_lcm, get_mask


class AttentionCouple:
    def __init__(self):
        self.batch_size: int
        self.patches: dict[str, Callable] = {}
        self.manual: dict[str, list]
        self.checked: bool

    @torch.inference_mode()
    def patch_unet(
        self,
        model: torch.nn.Module,
        base_mask,
        kwargs: dict,
        *,
        isA1111: bool,
        width: int,
        height: int,
    ):
        num_conds = len([k for k in kwargs.keys() if k.startswith("cond_")]) + 1
        use_regional_neg = bool(kwargs.get("use_regional_neg", False))

        mask = [base_mask] + [kwargs[f"mask_{i}"] for i in range(1, num_conds)]
        mask = torch.stack(mask, dim=0).to(device=device, dtype=dtype)

        if mask.sum(dim=0).min().item() <= 0.0:
            logger.error("Image must contain weights on all pixels...")
            return None

        mask = mask / mask.sum(dim=0, keepdim=True)

        conds = [
            kwargs[f"cond_{i}"][0][0].to(device=device, dtype=dtype)
            for i in range(1, num_conds)
        ]
        num_tokens = [cond.shape[1] for cond in conds]
        neg_conds = []
        neg_num_tokens = []
        if use_regional_neg:
            try:
                neg_conds = [
                    kwargs[f"neg_cond_{i}"][0][0].to(device=device, dtype=dtype)
                    for i in range(1, num_conds)
                ]
            except KeyError:
                use_regional_neg = False
                neg_conds = []
            else:
                neg_num_tokens = [neg_cond.shape[1] for neg_cond in neg_conds]
                if not any(neg_num_tokens):
                    use_regional_neg = False

        if isA1111:
            self.manual = {
                "original_shape": [2, 4, height // 8, width // 8],
                "cond_or_uncond": [0, 1],
            }
            self.checked = False

        def _repeat_tokens(tensor: torch.Tensor, target_len: int, batch_repeat=None):
            if target_len <= 0:
                return tensor

            current_batch = tensor.shape[0]
            repeat_batch = batch_repeat if batch_repeat is not None else current_batch
            need_batch_repeat = repeat_batch != current_batch

            seq_len = tensor.shape[1]
            repeat_tokens = (
                math.ceil(target_len / seq_len) if seq_len < target_len else 1
            )

            if need_batch_repeat or repeat_tokens > 1:
                tensor = tensor.repeat(
                    repeat_batch if need_batch_repeat else 1,
                    repeat_tokens,
                    1,
                )

            if tensor.shape[1] > target_len:
                tensor = tensor[:, :target_len, :]

            return tensor

        @torch.inference_mode()
        def attn2_patch(q, k, v, extra_options=None):
            assert torch.allclose(k, v), "k and v should be the same"
            if extra_options is None:
                if not self.checked:
                    self.manual["original_shape"][0] = k.size(0)
                    self.manual["cond_or_uncond"] = list(range(k.size(0)))
                    self.checked = True

                extra_options = self.manual

            cond_or_unconds = extra_options["cond_or_uncond"]
            num_chunks = len(cond_or_unconds)
            self.batch_size = q.shape[0] // num_chunks
            q_chunks = q.chunk(num_chunks, dim=0)
            k_chunks = k.chunk(num_chunks, dim=0)
            k_tokens = k.shape[1]

            pos_target_len = capped_lcm(k_tokens, num_tokens)
            if not conds:
                logger.error("No conditioning tensors available for attention patch.")
                return q, k, v
            conds_tensor = torch.cat(
                [
                    _repeat_tokens(cond, pos_target_len, self.batch_size)
                    for cond in conds
                ],
                dim=0,
            )

            neg_target_len = pos_target_len
            neg_conds_tensor = None
            use_negative_masks = False
            if use_regional_neg and neg_conds:
                neg_target_len = capped_lcm(k_tokens, neg_num_tokens)
                neg_conds_tensor = torch.cat(
                    [
                        _repeat_tokens(neg_cond, neg_target_len, self.batch_size)
                        for neg_cond in neg_conds
                    ],
                    dim=0,
                )
                use_negative_masks = True

            qs, ks = [], []
            for i, cond_or_uncond in enumerate(cond_or_unconds):
                if cond_or_uncond == 1 and use_negative_masks:
                    neg_k_target = _repeat_tokens(k_chunks[i], neg_target_len)
                    qs.append(q_chunks[i].repeat(num_conds, 1, 1))
                    ks.append(torch.cat([neg_k_target, neg_conds_tensor], dim=0))
                elif cond_or_uncond == 1:
                    qs.append(q_chunks[i])
                    ks.append(_repeat_tokens(k_chunks[i], pos_target_len))
                else:
                    pos_k_target = _repeat_tokens(k_chunks[i], pos_target_len)
                    qs.append(q_chunks[i].repeat(num_conds, 1, 1))
                    ks.append(torch.cat([pos_k_target, conds_tensor], dim=0))

            qs = torch.cat(qs, dim=0).to(q)
            ks = torch.cat(ks, dim=0).to(k)

            if qs.size(0) % 2 == 1:
                empty = torch.zeros_like(qs[0]).unsqueeze(0)
                qs = torch.cat((qs, empty), dim=0)
                empty = torch.zeros_like(ks[0]).unsqueeze(0)
                ks = torch.cat((ks, empty), dim=0)

            return qs, ks, ks

        @torch.inference_mode()
        def attn2_output_patch(out, extra_options=None):
            if extra_options is None:
                self.checked = False
                extra_options = self.manual

            cond_or_unconds = extra_options["cond_or_uncond"]
            mask_downsample = get_mask(
                mask, self.batch_size, out.shape[1], extra_options["original_shape"]
            )
            outputs = []
            pos = 0
            for cond_or_uncond in cond_or_unconds:
                if cond_or_uncond == 1:  # uncond
                    if use_regional_neg and neg_conds:
                        masked_output = (
                            out[pos : pos + num_conds * self.batch_size]
                            * mask_downsample
                        ).view(num_conds, self.batch_size, out.shape[1], out.shape[2])
                        masked_output = masked_output.sum(dim=0)
                        outputs.append(masked_output)
                        pos += num_conds * self.batch_size
                    else:
                        outputs.append(out[pos : pos + self.batch_size])
                        pos += self.batch_size
                else:
                    masked_output = (
                        out[pos : pos + num_conds * self.batch_size] * mask_downsample
                    ).view(num_conds, self.batch_size, out.shape[1], out.shape[2])
                    masked_output = masked_output.sum(dim=0)
                    outputs.append(masked_output)
                    pos += num_conds * self.batch_size
            return torch.cat(outputs, dim=0)

        if isA1111:

            def patch_attn2(layer: str, module: torch.nn.Module):
                f: Callable = module.forward
                self.patches[layer] = f

                @wraps(f)
                def _f(x, context, *args, **kwargs):
                    q = x
                    k = v = context
                    _q, _k, _v = attn2_patch(q, k, v)
                    return f(_q, context=_k, *args, **kwargs)

                module.forward = _f

            def patch_attn2_out(layer: str, module: torch.nn.Module):
                f: Callable = module.forward
                self.patches[layer] = f

                @wraps(f)
                def _f(*args, **kwargs):
                    _o = f(*args, **kwargs)
                    return attn2_output_patch(_o)

                module.forward = _f

            for layer_name, module in model.named_modules():
                if "attn2" not in layer_name:
                    continue

                if layer_name.endswith("2"):
                    patch_attn2(layer_name, module)

                if layer_name.endswith("to_out"):
                    patch_attn2_out(layer_name, module)

            return True

        else:
            model.set_model_attn2_patch(attn2_patch)
            model.set_model_attn2_output_patch(attn2_output_patch)

            return model

    @torch.no_grad()
    def unpatch(self, model: torch.nn.Module):
        if not self.patches:
            return

        for layer_name, module in model.named_modules():
            if "attn2" not in layer_name:
                continue

            if layer_name.endswith(("attn2", "to_out")):
                module.forward = self.patches.pop(layer_name)
