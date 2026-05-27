import torch
import torch.nn.functional as F
import numpy as np

from einops import rearrange

from .attention_utils import AttentionBase


class FocusDiffSelfAttentionControl(AttentionBase):
    def __init__(self, start_step=4, start_layer=10, layer_idx=None, step_idx=None, total_steps=50):
        """
        Original Interpolate Intermediate self-attention control for Stable-Diffusion model
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
        """
        super().__init__(total_steps)
        self.total_steps = total_steps
        self.start_step = start_step
        self.start_layer = start_layer
        self.layer_idx = layer_idx if layer_idx is not None else list(range(start_layer, 16))
        self.step_idx = step_idx if step_idx is not None else list(range(start_step, total_steps))
        print("step_idx: ", self.step_idx)
        print("layer_idx: ", self.layer_idx)

    def attn_batch(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        b = q.shape[0] // num_heads
        q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
        attn = sim.softmax(-1)
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "h (b n) d -> b n (h d)", b=b)
        return out

    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """
        
        # if kwargs.get("is_mask") is not None or is_cross or self.cur_step not in self.step_idx or self.cur_att_layer // 2 not in self.layer_idx:
        return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)

        # qu, qc = q.chunk(2)
        # ku, kc = k.chunk(2)
        # vu, vc = v.chunk(2)
        # attnu, attnc = attn.chunk(2)

        # out_u = self.attn_batch(qu, ku[:num_heads], vu[:num_heads], sim[:num_heads], attnu, is_cross, place_in_unet, num_heads, **kwargs)
        # out_c = self.attn_batch(qc, kc[:num_heads], vc[:num_heads], sim[:num_heads], attnc, is_cross, place_in_unet, num_heads, **kwargs)
        # out = torch.cat([out_u, out_c], dim=0)
        
        # return out
class FocusDiffSelfAttentionControlMask(FocusDiffSelfAttentionControl):
    def __init__(self,  start_step=4, start_layer=10, layer_idx=None, 
                 step_idx=None, total_steps=50, 
                 mask=None,DoErase=False):
        """
        Maske-guided Original Interpolate Intermediate to alleviate the problem of fore- and background confusion
        Args:
            start_step: the step to start mutual self-attention control
            start_layer: the layer to start mutual self-attention control
            layer_idx: list of the layers to apply mutual self-attention control
            step_idx: list the steps to apply mutual self-attention control
            total_steps: the total number of steps
            mask_s: source mask with shape (h, w)
            mask_t: target mask with same shape as source mask
        """
        super().__init__(start_step, start_layer, layer_idx, step_idx, total_steps)

        self.mask = mask  # source mask with shape (h, w)
        self.DoErase=DoErase
        print("Using mask-guided Original Interpolate Intermediate Ctrl Always Query BackGround")
        # if mask_save_dir is not None:
        #     os.makedirs(mask_save_dir, exist_ok=True)
        #     save_image(self.mask_s.unsqueeze(0).unsqueeze(0).to(torch.float32), os.path.join(mask_save_dir, "mask_s.png"))
        #     if self.mask_t is not None:
        #         save_image(self.mask_t.unsqueeze(0).unsqueeze(0).to(torch.float32), os.path.join(mask_save_dir, "mask_t.png"))
    def attn_mask(self, q, k, v, num_heads, **kwargs):
        B = q.shape[0] // num_heads
        H = W = int(np.sqrt(q.shape[1]))
        # q = rearrange(q, "(b h) n d -> h (b n) d", h=num_heads)
        # k = rearrange(k, "(b h) n d -> h (b n) d", h=num_heads)
        # v = rearrange(v, "(b h) n d -> h (b n) d", h=num_heads)

        sim = torch.einsum("h i d, h j d -> h i j", q, k) * kwargs.get("scale")
  
        mask = self.getMask().unsqueeze(0).unsqueeze(0)
        mask = F.interpolate(mask, (H, W)).flatten(0).unsqueeze(0)
        mask = mask.flatten().to(sim.dtype)
        # background 
        if kwargs.get("Reverse"):
            sim = sim + mask.masked_fill(mask == 0, torch.finfo(sim.dtype).min).masked_fill(mask == 1,0)
        else:
            sim = sim + mask.masked_fill(mask == 1, torch.finfo(sim.dtype).min)
            
        attn = sim.softmax(-1)
       
        out = torch.einsum("h i j, h j d -> h i d", attn, v)
        out = rearrange(out, "h (b n) d -> b n (h d)", b=B, h=num_heads)
        return out
 
    def getMask(self):
        return self.mask.clone()
    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """
        H = W = int(np.sqrt(q.shape[1]))
        out_self_attn=super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads,**kwargs)

        if is_cross:
         
            mask_fg_cross = self.getMask().unsqueeze(0).unsqueeze(0)
            mask_fg_cross = F.interpolate(mask_fg_cross, (H, W)).flatten(0).unsqueeze(0)
            mask_fg_cross = mask_fg_cross.flatten()
            
     
            q_object=q[-num_heads:,mask_fg_cross==1,:]
            k_object=k[-num_heads:,:,:]
            v_object=v[-num_heads:,:,:]
            
          
            sim_object = torch.einsum('b i d, b j d -> b i j', q_object, k_object) *kwargs.get("scale") # (b h) no nt
        
            attn_object= sim_object.softmax(dim=-1) # 8*128*77
            out_object = torch.einsum('b i j, b j d -> b i d', attn_object, v_object) 

            
            q_background=q[-num_heads:,mask_fg_cross==0,:]
            k_background=k[:num_heads:,:,:]
            v_background=v[:num_heads:,:,:]
            
            
            sim_background = torch.einsum('b i d, b j d -> b i j', q_background, k_background) *kwargs.get("scale") # (b h) no nt
            attn_background= sim_background.softmax(dim=-1) # 8*128*77
            out_background = torch.einsum('b i j, b j d -> b i d', attn_background, v_background) 
            
            out = torch.einsum('b i j, b j d -> b i d', attn, v) 
            out[-num_heads:,mask_fg_cross==1,:]=out_object
            out[-num_heads:,mask_fg_cross==0,:]=out_background

            out = rearrange(out, '(b h) n d -> b n (h d)', h=num_heads)
            
            return out



        out_target ,out_intermediate,out_query_object,out_u_object,out_c_object=out_self_attn.chunk(5)
        
        if self.DoErase:
            out_target = self.attn_mask(q[:num_heads], k[num_heads:2*num_heads], v[num_heads:2*num_heads], num_heads, **kwargs)
            
        else:
        
            out_target_bg_mask = self.attn_mask(q[:num_heads], k[num_heads:2*num_heads], v[num_heads:2*num_heads], num_heads, **kwargs)
            out_target_object_mask = self.attn_mask(q[:num_heads], k[-num_heads:], v[-num_heads:],  num_heads, Reverse=True, **kwargs)


            out_target_bg_mask_object = self.attn_mask(q[-num_heads:], k[2*num_heads:3*num_heads], v[2*num_heads:3*num_heads], num_heads, **kwargs)       
            out_target_bg_mask_object_u = self.attn_mask(q[-2*num_heads:-num_heads], k[2*num_heads:3*num_heads], v[2*num_heads:3*num_heads], num_heads, **kwargs)

            mask=self.getMask()
            mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), (H, W))
            mask = mask.reshape(-1, 1)  # (hw, 1)

            out_target = out_target_object_mask * mask  + out_target_bg_mask * (1 - mask)  

            out_c_object= out_c_object * mask  + out_target_bg_mask_object * (1 - mask) 
            out_u_object= out_u_object * mask  + out_target_bg_mask_object_u * (1 - mask) 

        
        out = torch.cat([out_target,out_intermediate, out_query_object,out_u_object,out_c_object], dim=0)

        return out
