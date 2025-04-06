import torch
import numpy as np
from torch import Tensor
import torch.nn as nn
from SAM2.sam2.sam2.modeling.sam2_base import SAM2Base
from safetensors.torch import save_file
from safetensors import safe_open
from SAM2.sam2.sam2.modeling.sam.transformer import Attention
import torch.nn.functional as F

class Attention_LoRA(nn.Module):

    def __init__(
        self,
        self_attn: Attention,
        A_q, B_q, A_v, B_v,
    ) -> None:
        super().__init__()
        self.embedding_dim = self_attn.embedding_dim
        self.kv_in_dim = self_attn.kv_in_dim
        self.internal_dim = self_attn.internal_dim
        self.num_heads = self_attn.num_heads
        
        embedding_dim = self.embedding_dim
        
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.v_proj = nn.Linear(self.kv_in_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

        self.A_q = A_q
        self.B_q = B_q
        self.A_v = A_v
        self.B_v = B_v
        
        self.dropout_p = self_attn.dropout_p
        
        
    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q) + self.B_q(self.A_q(q))
        k = self.k_proj(k) 
        v = self.v_proj(v) + self.B_v(self.A_v(v))

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        dropout_p = self.dropout_p if self.training else 0.0
        # Attention
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out

class LoRA_qkv(nn.Module):
    """
    LoRA adaption for attention modules. Only for queries and values

    Arguments:
        qkv: Original block of attention
        linear_a_q: linear block for q
        linear_b_q: linear block for q
        linear_a_v: linear block for v
        linear_b_v: linear block for v

    Return:
        qkv(nn.Module): qkv block with all linear blocks added (equivalent to adding the matrix B*A)
    """

    def __init__(
            self,
            qkv,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
       
        super(LoRA_qkv, self).__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.d_model = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x: Tensor):
        qkv = self.qkv(x)
        q_ba = self.linear_b_q(self.linear_a_q(x))
        v_ba = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, :self.d_model] += q_ba #q part
        qkv[:, :, :, -self.d_model:] += v_ba #v part

        return qkv
    
    
def wrap_decoder_lora(sam_model, rank: int):
    for param in sam_model.sam_mask_decoder.parameters():
        param.requires_grad = False
        
    A_weights = []
    B_weights = []
    
    for t_layer_i, blk in enumerate(sam_model.sam_mask_decoder.transformer.layers):
        embedding_dim = blk.self_attn.embedding_dim
        kv_in_dim = blk.self_attn.kv_in_dim
        internal_dim = blk.self_attn.internal_dim
        
        A_q = nn.Linear(embedding_dim, rank, bias=False)
        B_q = nn.Linear(rank, internal_dim, bias=False)
        A_v = nn.Linear(kv_in_dim, rank, bias=False)
        B_v = nn.Linear(rank, embedding_dim, bias=False)
        A_weights.append(A_q)
        A_weights.append(A_v)
        B_weights.append(B_q)
        B_weights.append(B_v)
        blk.self_attn = Attention_LoRA(
            blk.self_attn,
            A_q,
            B_q,
            A_v,
            B_v
        )
        
        for w_A in A_weights:
            nn.init.kaiming_uniform_(w_A.weight, a=np.sqrt(5))
        for w_B in B_weights:
            nn.init.zeros_(w_B.weight)
            
    # sam_model.register_module("A_weights_decoder", nn.ModuleList(A_weights))
    # sam_model.register_module("B_weights_decoder", nn.ModuleList(B_weights))
    
def wrap_image_encoder_lora(sam_model, rank: int):
    rank = rank
    assert rank > 0
    # base_vit_dim = sam_model.image_encoder.patch_embed.proj.out_channels

        # In each block, you have an attention block => total blocks -> nb lora layers
    lora_layer = list(range(len(sam_model.image_encoder.trunk.blocks)))
    
    A_weights = []
    B_weights = []
    # freeze parameters of the image encoder
    for param in sam_model.image_encoder.parameters():
        param.requires_grad = False
        
    for t_layer_i, blk in enumerate(sam_model.image_encoder.trunk.blocks[-2:]):
        # if only lora on few layers
        if t_layer_i not in lora_layer:
            continue
        w_qkv_linear = blk.attn.qkv
        d_model = w_qkv_linear.in_features
        w_a_linear_q = nn.Linear(d_model, rank, bias=False)
        w_b_linear_q = nn.Linear(rank, d_model, bias=False)
        w_a_linear_v = nn.Linear(d_model, rank, bias=False)
        w_b_linear_v = nn.Linear(rank, d_model, bias=False)
       
        A_weights.append(w_a_linear_q)
        B_weights.append(w_b_linear_q)
        A_weights.append(w_a_linear_v)
        B_weights.append(w_b_linear_v)
        blk.attn.qkv = LoRA_qkv(
            w_qkv_linear,
            w_a_linear_q,
            w_b_linear_q,
            w_a_linear_v,
            w_b_linear_v
        )

        """
        Initialize the LoRA A and B matrices like in the paper
        """
        # Initalisation like in the paper
        for w_A in A_weights:
            nn.init.kaiming_uniform_(w_A.weight, a=np.sqrt(5))
        for w_B in B_weights:
            nn.init.zeros_(w_B.weight)
            
    # sam_model.register_module("A_weights_encoder", nn.ModuleList(A_weights))
    # sam_model.register_module("B_weights_encoder", nn.ModuleList(B_weights))
    
    
def save_lora_parameters(sam_model, filename: str):
    """
    Save the LoRA wieghts applied to the attention model as safetensors.
    Arguments:
        filenmame: Name of the file that will be saved
   
    Return:
        None: Saves a safetensors file
    """
    total_wts = {}
    for lora_at in ["encoder", "decoder"]:
        if f"A_weights_{lora_at}" in sam_model.__dict__:
            A_weights = sam_model.__dict__[f"A_weights_{lora_at}"]
            B_weights = sam_model.__dict__[f"B_weights_{lora_at}"]
    
            num_layer = len(A_weights)
            # sufix 03:d -> allows to have a name 1 instead of 001
            a_tensors = {f"{lora_at}_w_a_{i:03d}": A_weights[i].weight for i in range(num_layer)}
            b_tensors = {f"{lora_at}_w_b_{i:03d}": B_weights[i].weight for i in range(num_layer)}
            merged_dict = {**a_tensors, **b_tensors}
            total_wts.update(merged_dict)
            
    save_file(total_wts, filename)
    
    
def load_lora_parameters(sam_model:SAM2Base, filename: str):
    """
    Load a safetensor file of LoRA weights for the attention modules
    Arguments:
        filename: Name of the file containing the saved weights
   
    Return:
        None: Loads the weights to the SAM2Lora class
    """
    
    with safe_open(filename, framework="pt") as f:
        lora_at = []
        if "A_weights_encoder" in sam_model.__dict__:
            lora_at.append("encoder")
        if "A_weights_decoder" in sam_model.__dict__:
            lora_at.append("decoder")
            
        for lora in lora_at:
            A_weights = sam_model.__dict__[f"A_weights_{lora}"]
            B_weights = sam_model.__dict__[f"B_weights_{lora}"]
            for i, w_A_linear in enumerate(A_weights):
                saved_key = f"{lora}_w_a_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_A_linear.weight = nn.Parameter(saved_tensor)
            for i, w_B_linear in enumerate(B_weights):
                saved_key = f"{lora}_w_b_{i:03d}"
                saved_tensor = f.get_tensor(saved_key)
                w_B_linear.weight = nn.Parameter(saved_tensor)
                
                

def custom_save_lora_parameters(model, save_path):
    state_dict = model.state_dict()
    lora_param = {}
    for name, param in state_dict.items():
        if '_a_q' in name or 'A_q' in name or '_b_q' in name or 'B_q' in name:
            lora_param[name] = param
            
    torch.save(lora_param, save_path)
            
           
# save_lora_parameters(model, "lora.pth")

def custom_load_lora_parameters(model, load_path):
    error = model.load_state_dict(torch.load(load_path), strict=False)
    for name in error.missing_keys:
        if '_a_q' in name or 'A_q' in name or '_b_q' in name or 'B_q' in name:
            raise ValueError(f"Custom LoRA Missing key in state_dict: {name}")