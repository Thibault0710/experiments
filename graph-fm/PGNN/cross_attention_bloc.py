import argparse
import os
import time

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch
import torch.cuda.amp as amp

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from model import FourierPositionalEncoder

import torch
import clip
from PIL import Image
import math

def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x

def rope(q, k) :
    # q and k are (B, H, T, C), suppose c%2=0
    B, H, T, C = q.shape
    T_ = k.shape[2]
    if T_ > T :
        k_c, q_c = rope(k, q)
        return q_c, k_c
    
    q_c = torch.stack([q[..., ::2], q[..., 1::2]], dim=-1) # (B, H, T, C//2, 2)
    k_c = torch.stack([k[..., ::2], k[..., 1::2]], dim=-1) # (B, H, T, C//2, 2)
    with torch.amp.autocast(device_type=q.device.type, enabled=False) :
        thetas = torch.arange(0, T, device=q.device)[:, None] / 10000**(2*torch.arange(0, C//2, device=q.device)[None, :]/C) # (T, C//2)
        cos_thetas = thetas.cos().to(q.dtype)
        sin_thetas = thetas.sin().to(q.dtype)
    return (torch.stack([q_c[..., 0]*cos_thetas[None,None] - q_c[..., 1]*sin_thetas[None,None], q_c[..., 0]*sin_thetas[None,None] + q_c[..., 1]*cos_thetas[None,None]], dim=-1).reshape(B, H,T, C), 
            torch.stack([k_c[..., 0]*cos_thetas[None,None,:T_] - k_c[..., 1]*sin_thetas[None,None,:T_], k_c[..., 0]*sin_thetas[None,None,:T_] + k_c[..., 1]*cos_thetas[None,None, :T_]], dim=-1).reshape(B,H, T_, C) )


def rope_2d(q, k, grid_size) :
    """2D-axial RoPE for a token sequence of length T = grid_size**2 in row-major order
    (token i -> x = i % G, y = i // G). First half of the head-channel dim rotates by x,
    second half by y, so attention sees true 2D proximity. q, k: (B, H, T, C).
    Requires T == T_ == G**2 and head_dim % 4 == 0.
    """
    B, H, T,  C = q.shape
    T_           = k.shape[2]
    G            = grid_size
    assert T == T_ == G*G, f"rope_2d expects T == T_ == grid_size**2; got T={T}, T_={T_}, G^2={G*G}"
    assert C % 4 == 0,     f"rope_2d needs head_dim % 4 == 0; got C={C}"

    Cx     = C // 2
    pos    = torch.arange(T, device=q.device)
    x_pos  = (pos %  G)
    y_pos  = (pos // G)

    def _apply(q_part, k_part, pos_, Cp) :
        q_pairs = torch.stack([q_part[..., ::2], q_part[..., 1::2]], dim=-1)   # (B, H, T, Cp//2, 2)
        k_pairs = torch.stack([k_part[..., ::2], k_part[..., 1::2]], dim=-1)
        with torch.amp.autocast(device_type=q.device.type, enabled=False) :
            thetas = pos_[:, None].float() / 10000 ** (2 * torch.arange(Cp // 2, device=q.device)[None, :].float() / Cp)  # (T, Cp//2)
            cos_t  = thetas.cos().to(q.dtype)
            sin_t  = thetas.sin().to(q.dtype)
        q_rot = torch.stack([q_pairs[..., 0]*cos_t[None,None] - q_pairs[..., 1]*sin_t[None,None],
                             q_pairs[..., 0]*sin_t[None,None] + q_pairs[..., 1]*cos_t[None,None]], dim=-1).reshape(B, H, T,  Cp)
        k_rot = torch.stack([k_pairs[..., 0]*cos_t[None,None] - k_pairs[..., 1]*sin_t[None,None],
                             k_pairs[..., 0]*sin_t[None,None] + k_pairs[..., 1]*cos_t[None,None]], dim=-1).reshape(B, H, T_, Cp)
        return q_rot, k_rot

    qx, kx = _apply(q[..., :Cx], k[..., :Cx], x_pos, Cx)
    qy, ky = _apply(q[..., Cx:], k[..., Cx:], y_pos, C - Cx)
    return torch.cat([qx, qy], dim=-1), torch.cat([kx, ky], dim=-1)


def short_seq_pe(T, dim, device, dtype=torch.float32):  # claude: PE for SHORT sequences (T~4-64). 10000-base PE makes most dims constant when positions are 0..T-1; here freqs span [2π/T, π] so all dims resolve.
    assert dim % 2 == 0
    half = dim // 2
    pos   = torch.arange(T, device=device, dtype=dtype)
    freqs = torch.exp(torch.linspace(math.log(2 * math.pi / max(T, 1)), math.log(math.pi), half, device=device, dtype=dtype))
    s     = torch.outer(pos, freqs)  # (T, half)
    return torch.cat([torch.sin(s), torch.cos(s)], dim=-1)  # (T, dim)

class CrossAttentionBloc(torch.nn.Module) :
    def __init__(self, input_dim1, input_dim2, hidden_dim, n_heads, expansion_factor=4.0, ffn=True, pe="None") :
        super().__init__()
        
        self.i1      = input_dim1
        self.i2      = input_dim2
        self.hdim    = hidden_dim
        self.n_heads = n_heads
        self.expansion_factor = expansion_factor
        assert hidden_dim % n_heads == 0

        # for attention
        self.Wq = torch.nn.Linear(self.i1, self.hdim,  bias=False)
        self.Wk = torch.nn.Linear(self.i2, self.hdim,  bias=False)
        self.Wv = torch.nn.Linear(self.i2, self.hdim,  bias=False)
        self.o  = torch.nn.Linear(self.hdim ,self.i1, bias=False)
        self.temperature = 1

        # FFN layers
        self.has_ffn = ffn
        if self.has_ffn :
            self.ffn = torch.nn.Sequential(torch.nn.Linear(self.i1, int(self.expansion_factor*self.i1)), torch.nn.SiLU(), torch.nn.Linear(int(self.expansion_factor*self.i1), self.i1))
        

        # for RMS norm params
        self.gamma1 = torch.nn.Parameter(torch.ones(self.i1))
        self.gamma2 = torch.nn.Parameter(torch.ones(self.i2))
        self.gamma3 = torch.nn.Parameter(torch.ones(self.i1))  # claude: pre-FFN RMSnorm scale

        self.e = torch.nn.Parameter(torch.zeros(1, 6, self.i1))  # claude: 4->6 (scale1, shift1, gate_a, scale2, shift2, gate_f)

        self.eps = 1e-6
        self.capture_attn = False
        self.last_attn    = None

        self.pe = (pe or "none").lower()  # claude: normalize so "Rope" / "ROPE" / "rope" all match downstream checks

    def forward(self, x1, x2, modulation, key_padding_mask=None) :
        # x1, x2: (B, T, i1), (B, T', i2); modulation: (B, 6, i1) -> scale1, shift1, gate_a, scale2, shift2, gate_f
        # key_padding_mask: (B, T') True for real keys, False for padding
        modulation = modulation + self.e
        scale1, shift1, gate_a, scale2, shift2, gate_f = (m.unsqueeze(1) for m in modulation.unbind(dim=1))  # claude: 6-way unbind

        B, T, T_ = x1.shape[0], x1.shape[1], x2.shape[1]
        rms1 = ((x1**2).mean(dim=-1, keepdim=True)+self.eps).sqrt()
        rms2 = ((x2**2).mean(dim=-1, keepdim=True)+self.eps).sqrt()
        x1_norm = (x1/rms1 * self.gamma1)*(1+scale1) + shift1
        x2_norm =  x2/rms2 * self.gamma2
        if not self.has_ffn : # only on self attn
            x2_norm = x2_norm*(1+scale1) + shift1

        xq = self.Wq(x1_norm).view(B, T , self.n_heads, self.hdim//self.n_heads).permute(0, 2, 1, 3) # (B, h, T, d)
        xk = self.Wk(x2_norm).view(B, T_, self.n_heads, self.hdim//self.n_heads).permute(0, 2, 1, 3) # (B, h, T', d)
        xv = self.Wv(x2_norm).view(B, T_, self.n_heads, self.hdim//self.n_heads).permute(0, 2, 1, 3) # (B, h, T', d)

        if self.pe == "rope" :
            xq, xk = rope(xq, xk)
        elif self.pe == "rope_2d" :
            G = int(round(T ** 0.5))
            assert G*G == T, f"rope_2d expects a square grid; got T={T} (not a perfect square)"
            xq, xk = rope_2d(xq, xk, G)

        # should replace here all this by h = x1 + gate_a*self.o(torch.nn.functional.scaled_dot_product_attention(xq, xk, xv)permute(0, 2, 1, 3).reshape(B, T, self.hdim)
        h = x1 + gate_a * self.o(torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=key_padding_mask[:, None, None, :] if key_padding_mask is not None else None).transpose(1, 2).reshape(B, T, self.hdim))

        # old version without flash attention
        # scores = xq @ xk.transpose(-1, -2) / math.sqrt(self.hdim//self.n_heads) / self.temperature   # (B, h, T, T')
        # if key_padding_mask is not None:
        #     scores = scores.masked_fill(~key_padding_mask[:, None, None, :], float('-inf'))
        # scores = torch.softmax(scores, dim=-1)
        # if self.capture_attn:
        #     self.last_attn = scores.detach()  # (B, h, T, T') — only set when opted in

        # #apply v
        # h = x1 + gate_a*self.o(torch.einsum('bhtv, bhvd -> bthd', scores, xv).reshape(B, T, self.hdim))

        if self.has_ffn :
            rms_h  = ((h**2).mean(dim=-1, keepdim=True) + self.eps).sqrt()
            h_norm = (h / rms_h * self.gamma3) * (1 + scale2) + shift2  # claude: pre-FFN adaLN, was missing
            return h + gate_f * self.ffn(h_norm)
        return h

    def get_gradient_norm(self) :
        return ( ((self.Wq.weight**2).sum() + (self.Wk.weight**2).sum() + (self.Wv.weight**2).sum() + (self.o.weight**2).sum() + (self.ffn[0].weight**2).sum() + (self.ffn[-1].weight**2).sum()) / (self.Wq.weight.numel() + self.Wk.weight.numel() + self.Wv.weight.numel() + self.o.weight.numel() + self.ffn[0].weight.numel() + self.ffn[-1].weight.numel())).sqrt()

    def init_normal(self, std=0.02):
        torch.nn.init.normal_(self.Wq.weight, std=std)
        torch.nn.init.normal_(self.Wk.weight, std=std)
        torch.nn.init.normal_(self.Wv.weight, std=std)
        torch.nn.init.normal_(self.o.weight,  std=std)

        if self.has_ffn:
            torch.nn.init.normal_(self.ffn[0].weight, std=std)
            torch.nn.init.zeros_(self.ffn[0].bias)

            torch.nn.init.normal_(self.ffn[-1].weight, std=std)
            torch.nn.init.zeros_(self.ffn[-1].bias)
            
class MeshColor(torch.nn.Module) :
    def __init__(self, hidden_dim=256, hidden_dim_clip=512, n_blocs=8, n_pe_freqs=8, pe="none", n_heads=4) :
        """
        pe ∈ {"none", "rope", "sinusoidal"}. Case-insensitive. "none" is the safe default —
        spatial info is already in node features via FourierPositionalEncoder.
        """
        super().__init__()
        pe = (pe or "none").lower()
        # we try a flow matching architecture
        self.hdim      = hidden_dim
        self.n_blocs   = n_blocs
        self.hdim_clip = hidden_dim_clip
        self.input_dim = self.hdim  # claude: was hardcoded 256, now follows --hidden-dim so blocs/modulation/final_conv all match linear_node out
        self.freq_dim  = 16
        self.n_heads   = n_heads
        self.pe = pe

        # Pointwise node embedding: (rgb=3, fourier(pos)=4*L) -> hdim
        self.pe_          = FourierPositionalEncoder(num_frequencies=n_pe_freqs)
        self.linear_node = torch.nn.Sequential(torch.nn.Linear(3 + self.pe_.output_dim, self.hdim), torch.nn.SiLU())

        # CLip embedding
        self.linear_clip = torch.nn.Sequential(torch.nn.Linear(512, self.hdim_clip), torch.nn.SiLU())  # claude: was Linear(512,512), output now follows --hidden-dim-clip so cross-attn i2 matches

        # t injection, same as in Wan
        self.time_embedding = nn.Sequential(
            nn.Linear(self.freq_dim, self.hdim), nn.SiLU(), nn.Linear(self.hdim, self.hdim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(self.hdim, self.hdim * 6))  # claude: 4 -> 6 adaLN channels
        nn.init.zeros_(self.time_projection[-1].weight)  # DiT-Zero: modulation = 0 at init
        nn.init.zeros_(self.time_projection[-1].bias)

        # DiT blocs
        blocs = []
        for _ in range(n_blocs) :
            blocs.extend([CrossAttentionBloc(self.input_dim, self.input_dim, hidden_dim_clip, self.n_heads, ffn=False, pe=pe), # self
            CrossAttentionBloc(self.input_dim, hidden_dim_clip, hidden_dim_clip, self.n_heads, ffn=True, pe=None)] # cross with CLIP embedding
            )
        self.blocs = torch.nn.ModuleList(blocs)

        #final proj — pointwise Linear (was GCNConv: aggregated over neighbors and washed out per-node detail)
        self.final_conv  = torch.nn.Linear(self.input_dim, 3)
        torch.nn.init.zeros_(self.final_conv.weight)
        torch.nn.init.zeros_(self.final_conv.bias)

    def set_capture_attn(self, on: bool = True):
        for b in self.blocs:
            b.capture_attn = on

    def clear_attn(self):
        for b in self.blocs:
            b.last_attn = None
    
    def forward(self, data, clips, ts, clips_mask=None) :
        # data: PyG Batch (.x, .edge_index, .batch); data.x is (rgb, cartesian) per node.
        B          = data.num_graphs
        edge_index = data.edge_index
        batch_vec  = data.batch
        # time embeddings, same as in Wan 
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, ts).float()) # (B, hdim)
        e0 = self.time_projection(e).view(B, 6, self.hdim)  # claude: 4 -> 6

        rgb_in = data.x[:, :3]
        pos_in = data.x[:, 3:5]
        h = self.linear_node(torch.cat([rgb_in, self.pe_(pos_in)], dim=-1))
        x_features, mask = to_dense_batch(h, batch_vec)  # (B, max_nodes, hdim)

        if self.pe == "sinusoidal":  # Vaswani PE on dense-batch token index, added once before bloc stack
            T  = x_features.shape[1]
            pe = sinusoidal_embedding_1d(self.hdim, torch.arange(T, device=x_features.device)).to(x_features.dtype)
            x_features = x_features + pe.unsqueeze(0)

        clips = self.linear_clip(clips)                       # target: (B, T', C) UMT-style
        T_    = clips.shape[1]
        #if self.pe is None :
        #    pe    = short_seq_pe(T_, clips.shape[-1], device=clips.device, dtype=clips.dtype)  # claude: was sinusoidal_embedding_1d which is constant for short T; this resolves text positions 0..T-1
        #    clips = clips + pe

        for i in range(0, len(self.blocs), 2) :
            x_features = self.blocs[i  ](x_features, x_features, e0, key_padding_mask=mask)        # self
            x_features = self.blocs[i+1](x_features, clips,      e0, key_padding_mask=clips_mask)  # cross
        
        h = x_features[mask]
        return self.final_conv(h)

    @torch.no_grad()
    def sample(self, batch, init_x, clips, clips_mask=None, n_steps=20, t_start=1.0):
        """N-step Euler integration from `init_x` at t=t_start down to t=0.
        Mutates batch.x in place. Returns final x_0 estimate, shape (sum_N, 3).
        """
        device = init_x.device
        B   = batch.num_graphs
        pos = batch.pos
        x   = init_x.clone()
        dt  = t_start / n_steps
        for i in range(n_steps):
            t       = t_start - i * dt
            batch.x = torch.cat([x, pos], dim=-1)
            ts      = torch.full((B,), t, device=device)
            v       = self.forward(batch, clips, ts, clips_mask=clips_mask)
            x       = x - dt * v
        return x


class MeshColorAdaLn(torch.nn.Module) :
    def __init__(self, hidden_dim=256, hidden_dim_clip=512, n_blocs=8, n_pe_freqs=8, pe="none", n_heads=4, expansion_factor=4.0, res_connexions=False) :
        """
        pe ∈ {"none", "rope", "sinusoidal"}. Case-insensitive. "none" is the safe default —
        spatial info is already in node features via FourierPositionalEncoder.
        """
        super().__init__()
        pe = (pe or "none").lower()
        # we try a flow matching architecture
        self.hdim      = hidden_dim
        self.n_blocs   = n_blocs
        self.hdim_clip = hidden_dim_clip
        self.input_dim = self.hdim  # claude: was hardcoded 256, now follows --hidden-dim so blocs/modulation/final_conv all match linear_node out
        self.freq_dim  = 16
        self.n_heads   = n_heads
        self.pe        = pe
        self.expansion_factor = expansion_factor
        self.res_connexions = res_connexions

        # Pointwise node embedding: (rgb=3, fourier(pos)=4*L) -> hdim
        self.pe_          = FourierPositionalEncoder(num_frequencies=n_pe_freqs)
        self.linear_node = torch.nn.Sequential(torch.nn.Linear(3 + self.pe_.output_dim, self.hdim), torch.nn.SiLU())

        # t injection, same as in Wan
        self.time_embedding = nn.Linear(self.freq_dim, self.hdim)
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(self.hdim, self.hdim * 6))  # claude: 4 -> 6 adaLN channels
        #nn.init.zeros_(self.time_projection[-1].weight)  # DiT-Zero: modulation = 0 at init
        #nn.init.zeros_(self.time_projection[-1].bias)

        # DiT blocs
        blocs = []
        for _ in range(n_blocs) :
            blocs.extend([CrossAttentionBloc(self.input_dim, self.input_dim, hidden_dim_clip, self.n_heads, expansion_factor=self.expansion_factor, ffn=True, pe=pe),] # self
            )
        self.blocs = torch.nn.ModuleList(blocs)

        #final proj — pointwise Linear (was GCNConv: aggregated over neighbors and washed out per-node detail)
        self.final_conv  = torch.nn.Linear(self.input_dim, 3)

        self.label_embedding_dim = 16
        self.label_embedding     = torch.nn.Embedding(11, self.label_embedding_dim)
        self.clip_linear         = torch.nn.Sequential(torch.nn.Linear(self.label_embedding_dim, 4*self.label_embedding_dim), torch.nn.SiLU(), torch.nn.Linear(4*self.label_embedding_dim, n_blocs*6*self.hdim))
        torch.nn.init.zeros_(self.clip_linear[-1].weight)
        torch.nn.init.zeros_(self.clip_linear[-1].bias)

    def set_capture_attn(self, on: bool = True):
        for b in self.blocs:
            b.capture_attn = on

    def clear_attn(self):
        for b in self.blocs:
            b.last_attn = None
    
    def forward(self, data, labels, ts) :
        # data: PyG Batch (.x, .edge_index, .batch); data.x is (rgb, cartesian) per node. labels is (B )
        B          = data.num_graphs
        edge_index = data.edge_index
        batch_vec  = data.batch
        # time embeddings, same as in Wan 
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, ts).float()) # (B, hdim)
        e0 = self.time_projection(e).view(B, 6, self.hdim)  # claude: 4 -> 6

        rgb_in = data.x[:, :3]
        pos_in = data.x[:, 3:5]
        h = self.linear_node(torch.cat([rgb_in, self.pe_(pos_in)], dim=-1))
        x_features, mask = to_dense_batch(h, batch_vec)  # (B, max_nodes, hdim), we need masks to deal with different numbers of nodes per graph in a batch

        if self.pe == "sinusoidal":  # Vaswani PE on dense-batch token index, added once before bloc stack
            T  = x_features.shape[1]
            pe = sinusoidal_embedding_1d(self.hdim, torch.arange(T, device=x_features.device)).to(x_features.dtype)
            x_features = x_features + pe.unsqueeze(0)

        clips_modulations = self.clip_linear(self.label_embedding(labels)).view(B, self.n_blocs, 6, self.hdim).permute(1, 0, 2, 3)

        #if self.pe is None :
        #    pe    = short_seq_pe(T_, clips.shape[-1], device=clips.device, dtype=clips.dtype)  # claude: was sinusoidal_embedding_1d which is constant for short T; this resolves text positions 0..T-1
        #    clips = clips + pe
        if self.res_connexions :
            residuals = []

        for i in range(0, len(self.blocs)) :
            x_features = self.blocs[i](x_features, x_features, e0+clips_modulations[i], key_padding_mask=mask)        # self
            if self.res_connexions and i < len(self.blocs) // 2:
                residuals.append(x_features)
            elif self.res_connexions and i > (len(self.blocs) - 1) // 2:
                x_features = x_features + residuals[len(self.blocs) - 1 - i]


        h = x_features[mask]
        return self.final_conv(h)

    @torch.no_grad()
    def sample(self, batch, init_x, labels, clips_mask=None, n_steps=20, t_start=1.0):
        """N-step Euler integration from `init_x` at t=t_start down to t=0.
        Mutates batch.x in place. Returns final x_0 estimate, shape (sum_N, 3).
        """
        device = init_x.device
        B   = batch.num_graphs
        pos = batch.pos
        x   = init_x.clone()
        dt  = t_start / n_steps
        for i in range(n_steps):
            t       = t_start - i * dt
            batch.x = torch.cat([x, pos], dim=-1)
            ts      = torch.full((B,), t, device=device)
            v       = self.forward(batch, labels, ts, clips_mask=clips_mask)
            x       = x - dt * v
        return x

    
