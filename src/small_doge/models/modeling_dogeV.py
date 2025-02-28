import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput
from .configuration_dogeV import DogeVConfig
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS


class DogeVRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        """
        Formula:
        rms_norm: x / (sqrt(mean(x^2)) + eps) * weight

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def _norm(self,hidden_states):
        return hidden_states / (torch.sqrt(torch.mean(hidden_states**2, dim=-1, keepdim=True)) + self.eps)
         

    def forward(self, hidden_states):
        output = self._norm(hidden_states.float().type_as(hidden_states))
        return output * self.weight
    
    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"

class DogeVResidual(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size)) 

    def forward(self, hidden_states, residual_states):
        return hidden_states + residual_states * self.weight
    
    def extra_repr(self):
        return f"{tuple(self.weight.shape)}"
    

class DogeVRotaryEmbedding(nn.Module):
    def __init__(self, config: DogeVConfig, device: None):
        super().__init__()
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_int_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        
        inv_freq, self.attention_scaling = self.rope_int_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1   
        if seq_len > self.max_seq_len_cached:
            inv_freq, self.attention_scaling = self.rope_int_fn(self.config, device, seq_len)
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        if seq_len < self.max_seq_len_cached and self.max_seq_len_cached > self.   original_max_seq_len:
            inv_freq, self.attention_scaling = self.rope_int_fn(self.config, device)
            
            self.original_inv_freq = self.original_inv_freq.to(device)
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if 'dynamic' in self.rope_type:
            self._dynamic_frequency_update(position_ids, x.device)

        # [head_dim/2] -> [1, head_dim/2, 1] -> [batch_size, head_dim/2, 1] 
        inv_freq_expand = self.inv_freq[None, :, None].expand(x.shape[0], -1, -1)
        #  [batch_size, seq_len, 1]
        position_ids_expand = position_ids[:, :, None].expand(-1, -1, x.shape[-1])
        # Force float32 (see https://github.com/huggingface/transformers/pull/292285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(
            device_type=device_type,
            enabled=False):
            # [batch_size, seq_len, head_dim/2]
            freq = (inv_freq_expand.float() * position_ids_expand.float()).transpose(1, 2)

            emb = torch.cat(freq,freq)
            cos = emb.cos()
            sin = emb.sin()
        
        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
    
def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, positon_ids=None, unsqueezze_dim=1):
   """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
   
   




   
  



        






         
