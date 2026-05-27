import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from ..attention_utils import AttentionBase


class FocusDiffSelfAttentionControlMask(AttentionBase):
    def __init__(
        self,
        start_step=4,
        start_layer=10,
        layer_idx=None,
        step_idx=None,
        total_steps=50,
        mask=None,
        DoErase=False,
    ):
        super().__init__(total_steps)
        self.mask = mask
        self.DoErase = DoErase

    def getMask(self):
        return self.mask.clone()

    def attn_mask(self, q, k, v, num_heads, **kwargs):
        batch_size = q.shape[0] // num_heads
        height = width = int(np.sqrt(q.shape[1]))
        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        mask = self.getMask().unsqueeze(0).unsqueeze(0)
        mask = F.interpolate(mask, (height, width)).flatten().to(sim.dtype)
        neg = torch.finfo(sim.dtype).min
        if kwargs.get("Reverse"):
            sim = sim + mask.masked_fill(mask == 0, neg).masked_fill(mask == 1, 0)
        else:
            sim = sim + mask.masked_fill(mask == 1, neg)
        attn = sim.softmax(-1)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        return rearrange(out, "h (b n) d -> b n (h d)", b=batch_size, h=num_heads)

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        height = width = int(np.sqrt(q.shape[1]))
        out_self_attn = super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

        if is_cross:
            mask_fg_cross = self.getMask().unsqueeze(0).unsqueeze(0)
            mask_fg_cross = F.interpolate(mask_fg_cross, (height, width)).flatten()

            q_object = q[-num_heads:, mask_fg_cross == 1, :]
            k_object = k[-num_heads:, :, :]
            v_object = v[-num_heads:, :, :]
            sim_object = torch.einsum("b i d, b j d -> b i j", q_object, k_object) * kwargs.get("scale")
            out_object = torch.einsum("b i j, b j d -> b i d", sim_object.softmax(dim=-1), v_object)

            q_background = q[-num_heads:, mask_fg_cross == 0, :]
            k_background = k[:num_heads, :, :]
            v_background = v[:num_heads, :, :]
            sim_background = torch.einsum("b i d, b j d -> b i j", q_background, k_background) * kwargs.get("scale")
            out_background = torch.einsum(
                "b i j, b j d -> b i d", sim_background.softmax(dim=-1), v_background
            )

            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out[-num_heads:, mask_fg_cross == 1, :] = out_object
            out[-num_heads:, mask_fg_cross == 0, :] = out_background
            return rearrange(out, "(b h) n d -> b n (h d)", h=num_heads)

        out_target, out_intermediate, out_query_object, out_u_object, out_c_object = out_self_attn.chunk(5)
        if self.DoErase:
            out_target = self.attn_mask(q[:num_heads], k[num_heads : 2 * num_heads], v[num_heads : 2 * num_heads], num_heads, **kwargs)
        else:
            out_target_bg = self.attn_mask(q[:num_heads], k[num_heads : 2 * num_heads], v[num_heads : 2 * num_heads], num_heads, **kwargs)
            out_target_fg = self.attn_mask(q[:num_heads], k[-num_heads:], v[-num_heads:], num_heads, Reverse=True, **kwargs)
            out_object_bg_c = self.attn_mask(q[-num_heads:], k[2 * num_heads : 3 * num_heads], v[2 * num_heads : 3 * num_heads], num_heads, **kwargs)
            out_object_bg_u = self.attn_mask(
                q[-2 * num_heads : -num_heads],
                k[2 * num_heads : 3 * num_heads],
                v[2 * num_heads : 3 * num_heads],
                num_heads,
                **kwargs,
            )

            mask = self.getMask()
            mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), (height, width)).reshape(-1, 1)
            out_target = out_target_fg * mask + out_target_bg * (1 - mask)
            out_c_object = out_c_object * mask + out_object_bg_c * (1 - mask)
            out_u_object = out_u_object * mask + out_object_bg_u * (1 - mask)

        return torch.cat([out_target, out_intermediate, out_query_object, out_u_object, out_c_object], dim=0)
