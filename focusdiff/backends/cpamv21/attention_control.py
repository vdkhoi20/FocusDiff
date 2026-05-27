import os

import torch
import torch.nn.functional as F
import numpy as np

from einops import rearrange

from .attention_utils import AttentionBase

from torchvision.utils import save_image


class FocusDiffSelfAttentionControl(AttentionBase):
    MODEL_TYPE = {
        "SD": 16,
        "SDXL": 70
    }
    def __init__(self, start_step=4, start_layer=10, layer_idx=None, step_idx=None, total_steps=50, model_type="SD"):
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
        self.total_layers = self.MODEL_TYPE.get(model_type, 16)
        
        self.layer_idx = layer_idx if layer_idx is not None else list(range(start_layer, self.total_layers+1))
        self.step_idx = step_idx if step_idx is not None else list(range(start_step, total_steps+1))
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
        
        return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
class FocusDiffSelfAttentionControlMask(FocusDiffSelfAttentionControl):
    def __init__(self,  start_step=4, start_layer=10, layer_idx=None, 
                 step_idx=None, total_steps=50, 
                 mask_s=None, mask_t=None, model_type="SD"):
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
        super().__init__(start_step, start_layer, layer_idx, step_idx, total_steps, model_type)

        self.mask_s = mask_s 
        self.mask_t = mask_t  
        
    def attn_batch(self, q, k, v, num_heads,background=True,ref_v=None, **kwargs):
        H = W = int(np.sqrt(q.shape[1]))
        
        q = rearrange(q, "(b h) n d -> b h n d", h=num_heads)
        
        sim = torch.einsum("b h i d,h j d -> b h i j", q, k) * kwargs.get("scale")

        mask_sr = self.getMask(H, W)
        mask_sr=mask_sr.unsqueeze(0).unsqueeze(0).unsqueeze(0) # 1,1,1,h*w
        mask_sr=mask_sr.to(sim.dtype)
        # # background
        NEG_INF = -1e4 if sim.dtype == torch.float16 else -1e9
        if background:
            # pass
            sim = sim.masked_fill(mask_sr == 1 , NEG_INF)
            
        else:
            # pass
            sim = sim.masked_fill(mask_sr == 0, NEG_INF)
        
        attn = sim.softmax(-1)

        # breakpoint()
        
        out = torch.einsum("b h i j, h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)", h=num_heads)
        return out
    def getMask(self,H,W,target=True):
        if (target): mask=self.mask_t.clone()
        else: mask=self.mask_s.clone()
        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), (H, W),mode="nearest")

        mask = mask.reshape(-1).squeeze(-1) # (hw)
        return mask
    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """

        H = W = int(np.sqrt(q.shape[1]))
        out_self_attn=super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads,**kwargs)
        if is_cross:
            mask_fg_cross = self.getMask(H,W)
            
            # breakpoint()
    
            q_object=q[-num_heads:,mask_fg_cross==1,:] # h*b, w*h,dim
            k_object=k[-num_heads:,:,:] # h*b, length_tokens , dim 
            v_object=v[-num_heads:,:,:] # h*b, length_tokens, dim
            
            sim_object = torch.einsum('b i d, b j d -> b i j', q_object, k_object) *kwargs.get("scale") # b*h, w*h, length_tokens
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

    
        out_intermediate,out_u_target,out_c_target=out_self_attn.chunk(3)
        
        
        out_target_bg_u,out_target_bg_c = self.attn_batch(q[-2*num_heads:], k[:num_heads], v[:num_heads], num_heads,background=True, **kwargs).chunk(2)
        out_target_fg_u,out_target_fg_c = self.attn_batch(q[-2*num_heads:], k[:num_heads], v[:num_heads], num_heads,background=False, **kwargs).chunk(2)
        
         
        mask=self.getMask(H,W).unsqueeze(-1).unsqueeze(0)
        # breakpoint()
        
    
        if self.cur_step  in self.step_idx and self.cur_att_layer // 2  in self.layer_idx:
            out_u_target = out_target_fg_u * mask + out_target_bg_u * (1 - mask)
            out_c_target = out_target_fg_c * mask + out_target_bg_c * (1 - mask)
            
        else:
            out_u_target = out_u_target * mask  + out_target_bg_u * (1 - mask)
            out_c_target = out_c_target * mask  + out_target_bg_c * (1 - mask)


   
        out = torch.cat([out_intermediate, out_u_target,out_c_target], dim=0)

        return out
class FocusDiffSelfAttentionControlMaskExpand(FocusDiffSelfAttentionControlMask):
    def __init__(self,  start_step=4, start_layer=10, layer_idx=None, 
                 step_idx=None, total_steps=50, 
                 mask_s=None, 
                 mask_t=None,
                 thres_hold=0.25,
                 ref_token_ids_object=[1],
                 step_change_mask=5, model_type="SD"):
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
        super().__init__(start_step, start_layer, layer_idx, step_idx, total_steps,
                         mask_s,mask_t, model_type)

        self.step_change_mask=step_change_mask
        self.thres_hold=thres_hold
        self.refining_masks=[]
        self.cross_attention_maps=[]
        self.refining_mask=None
        self.ref_token_ids_object=ref_token_ids_object
        self.num_cross_attention=0
    def aggregate_cross_attn_map(self):
        attns=torch.stack(self.cross_attention_maps,dim=0)
        attns= attns.sum(dim=0)/self.num_cross_attention

        min_value = attns.min()
        max_value = attns.max()
        attns = (attns - min_value) / (max_value-min_value)

        gr_eq=attns>=self.thres_hold
        lt=attns<self.thres_hold
        attns[gr_eq]=1
        attns[lt]=0

        return attns
    def add_attn_map(self,num_heads,attn):
        
        attn_ra=attn[-num_heads:]
        size=int(attn_ra.shape[1]**0.5)
        attn_ra=attn_ra[...,self.ref_token_ids_object]
        attn_ra=rearrange(attn_ra, "he (w h) d -> he d w h",w=size)
        attn_ra=F.interpolate(attn_ra, size=(64, 64), mode='nearest')
        attn_ra=rearrange(attn_ra, "he d w h -> (he d) w h")
        attn_ra=attn_ra.sum(dim=0)/attn_ra.shape[0]/torch.max(attn_ra) # W,H

 
        self.cross_attention_maps.append(attn_ra)
    def getMask(self,H,W,target=True):
        if (not target):
            mask=self.mask_s.clone()
        else:
            mask = self.mask_t.clone()
            if (self.num_cross_attention>=self.step_change_mask*self.num_att_layers//2): 
                mask= self.refining_mask.clone()
        
        mask = F.interpolate(mask.unsqueeze(0).unsqueeze(0), (H, W),mode="nearest")
        mask = mask.reshape(-1).squeeze(-1)  # (hw, 1)
        
        return mask
    def forward(self, q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs):
        """
        Attention forward function
        """
        if is_cross:
            #auto refining mask
            self.num_cross_attention+=1
            if (self.num_cross_attention>=self.step_change_mask*self.num_att_layers//2):
                self.add_attn_map(num_heads,attn)
                if (self.num_cross_attention%(self.num_att_layers//2)==0):
                    self.refining_mask=self.aggregate_cross_attn_map()
                    self.refining_masks.append(self.refining_mask)
        return super().forward(q, k, v, sim, attn, is_cross, place_in_unet, num_heads, **kwargs)
