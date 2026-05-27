import torch
import torch.nn as nn
from einops import rearrange, repeat


class AttentionBase:
    def __init__(self, max_step=50):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.max_step = max_step

    def reset(self):
        self.cur_att_layer = 0
        self.cur_step = 0

    def after_step(self):
        self.cur_att_layer = 0
        self.cur_step = (self.cur_step + 1) % self.max_step
        if self.cur_step == 0:
            self.reset()

    def __call__(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        out = self.forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.after_step()
        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        out = torch.einsum("b i j, b j d -> b i d", attn, v)
        return rearrange(out, "(b h) n d -> b n (h d)", h=num_heads)


def register_attention_editor_diffusers(model, editor: AttentionBase):
    def ca_forward(self, place_in_unet):
        def forward(x, encoder_hidden_states=None, attention_mask=None, context=None, mask=None):
            if encoder_hidden_states is not None:
                context = encoder_hidden_states
            if attention_mask is not None:
                mask = attention_mask

            to_out = self.to_out[0] if isinstance(self.to_out, nn.modules.container.ModuleList) else self.to_out
            num_heads = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q, k, v = map(lambda t: rearrange(t, "b n (h d) -> (b h) n d", h=num_heads), (q, k, v))

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale
            if mask is not None:
                mask = rearrange(mask, "b ... -> b (...)")
                mask = repeat(mask, "b j -> (b h) () j", h=num_heads)
                sim.masked_fill_(~mask, -torch.finfo(sim.dtype).max)

            attn = sim.softmax(dim=-1)
            out = editor(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, scale=self.scale)
            return to_out(out)

        return forward

    def register_editor(net, count, place_in_unet):
        for _, subnet in net.named_children():
            if net.__class__.__name__ == "Attention":
                net.forward = ca_forward(net, place_in_unet)
                return count + 1
            if hasattr(net, "children"):
                count = register_editor(subnet, count, place_in_unet)
        return count

    attention_count = 0
    for net_name, net in model.unet.named_children():
        if "down" in net_name:
            attention_count += register_editor(net, 0, "down")
        elif "mid" in net_name:
            attention_count += register_editor(net, 0, "mid")
        elif "up" in net_name:
            attention_count += register_editor(net, 0, "up")
    editor.num_att_layers = attention_count
