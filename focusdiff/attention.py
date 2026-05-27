import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange


class FocusDiffAttentionControl:
    MODEL_LAYERS = {"SD": 16, "SDXL": 70}

    def __init__(
        self,
        base_cls,
        start_step=7,
        start_layer=16,
        layer_idx=None,
        step_idx=None,
        total_steps=50,
        mask=None,
        do_erase=False,
        model_type="SD",
    ):
        self.base = base_cls(total_steps)
        self.total_steps = total_steps
        self.start_step = start_step
        self.start_layer = start_layer
        self.total_layers = self.MODEL_LAYERS.get(model_type, 16)
        self.layer_idx = layer_idx if layer_idx is not None else list(range(start_layer, self.total_layers + 1))
        self.step_idx = step_idx if step_idx is not None else list(range(start_step, total_steps + 1))
        self.mask = mask.float()
        self.do_erase = do_erase
        self.cur_step = 0
        self.cur_att_layer = 0
        self.num_att_layers = -1

    def reset(self):
        self.cur_att_layer = 0
        self.cur_step = 0

    def after_step(self):
        self.cur_att_layer = 0
        self.cur_step = (self.cur_step + 1) % self.total_steps
        if self.cur_step == 0:
            self.reset()

    def __call__(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        out = self.forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.after_step()
        return out

    def _base_forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        return self.base.forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

    def _mask(self, h: int, w: int) -> torch.Tensor:
        mask = F.interpolate(self.mask[None, None], (h, w), mode="nearest")
        return mask.reshape(-1).to(dtype=torch.float32, device=self.mask.device)

    def _masked_attn(self, q, k, v, num_heads, reverse=False, **kwargs):
        b = q.shape[0] // num_heads
        h = w = int(np.sqrt(q.shape[1]))
        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        mask = self._mask(h, w).to(sim.device, sim.dtype)
        neg = torch.finfo(sim.dtype).min
        if reverse:
            sim = sim + mask.masked_fill(mask == 0, neg).masked_fill(mask == 1, 0)
        else:
            sim = sim + mask.masked_fill(mask == 1, neg)
        attn = sim.softmax(-1)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        return rearrange(out, "h (b n) d -> b n (h d)", b=b, h=num_heads)

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        out_self = self._base_forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
        h = w = int(np.sqrt(q.shape[1]))

        if is_cross:
            mask = self._mask(h, w).to(q.device).bool()
            q_object = q[-num_heads:, mask, :]
            k_object = k[-num_heads:, :, :]
            v_object = v[-num_heads:, :, :]
            sim_object = torch.einsum("b i d, b j d -> b i j", q_object, k_object) * kwargs.get("scale")
            out_object = torch.einsum("b i j, b j d -> b i d", sim_object.softmax(-1), v_object)

            q_background = q[-num_heads:, ~mask, :]
            k_background = k[:num_heads, :, :]
            v_background = v[:num_heads, :, :]
            sim_background = torch.einsum("b i d, b j d -> b i j", q_background, k_background) * kwargs.get("scale")
            out_background = torch.einsum("b i j, b j d -> b i d", sim_background.softmax(-1), v_background)

            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out[-num_heads:, mask, :] = out_object
            out[-num_heads:, ~mask, :] = out_background
            return rearrange(out, "(b h) n d -> b n (h d)", h=num_heads)

        chunks = out_self.chunk(5)
        out_target, out_intermediate, out_query_object, out_u_object, out_c_object = chunks
        if self.do_erase:
            out_target = self._masked_attn(q[:num_heads], k[num_heads : 2 * num_heads], v[num_heads : 2 * num_heads], num_heads, **kwargs)
        else:
            bg = self._masked_attn(q[:num_heads], k[num_heads : 2 * num_heads], v[num_heads : 2 * num_heads], num_heads, **kwargs)
            fg = self._masked_attn(q[:num_heads], k[-num_heads:], v[-num_heads:], num_heads, reverse=True, **kwargs)
            obj_bg_c = self._masked_attn(q[-num_heads:], k[2 * num_heads : 3 * num_heads], v[2 * num_heads : 3 * num_heads], num_heads, **kwargs)
            obj_bg_u = self._masked_attn(q[-2 * num_heads : -num_heads], k[2 * num_heads : 3 * num_heads], v[2 * num_heads : 3 * num_heads], num_heads, **kwargs)
            mask = self._mask(h, w).to(out_target.device, out_target.dtype).reshape(1, -1, 1)
            out_target = fg * mask + bg * (1 - mask)
            out_c_object = out_c_object * mask + obj_bg_c * (1 - mask)
            out_u_object = out_u_object * mask + obj_bg_u * (1 - mask)

        return torch.cat([out_target, out_intermediate, out_query_object, out_u_object, out_c_object], dim=0)
