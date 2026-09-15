import torch
import torch.nn as nn
import math
from timm.models.vision_transformer import Attention, Mlp

def modulate(x, shift, scale):
    if shift.ndim == 2:
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
    return x * (1 + scale) + shift


def apply_gate(x, gate):
    if gate.ndim == 2:
        gate = gate.unsqueeze(1)
    return gate * x

# ---------- timestep embedder ----------
class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb  # (B, hidden_size)

class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0.0)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + apply_gate(
            self.attn(modulate(self.norm1(x), shift_msa, scale_msa)),
            gate_msa,
        )
        x = x + apply_gate(
            self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)),
            gate_mlp,
        )
        return x


class RelationGatedGraphBlock(nn.Module):
    """Transformer block whose attention is biased by a learned slot graph."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.num_heads = int(num_heads)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            hidden_size, self.num_heads, dropout=0.0, batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size),
        )

    def forward(self, tokens, relation_gates):
        batch_size, num_slots = tokens.shape[:2]
        if relation_gates.shape != (batch_size, num_slots, num_slots):
            raise ValueError(
                "relation_gates must have shape [B, K, K], got "
                f"{tuple(relation_gates.shape)}"
            )
        attention_bias = relation_gates.clamp_min(1e-6).log()
        attention_bias = attention_bias.repeat_interleave(
            self.num_heads, dim=0
        )
        normalized = self.norm1(tokens)
        attended, _ = self.attn(
            normalized,
            normalized,
            normalized,
            attn_mask=attention_bias,
            need_weights=False,
        )
        tokens = tokens + attended
        return tokens + self.mlp(self.norm2(tokens))


class RelationGatedGraphEncoder(nn.Module):
    """Cross-slot encoder with soft gates and optional top-k sparsity."""

    def __init__(self, hidden_size, num_heads, depth, topk=3,
                 gate_floor=0.05, temperature=1.0, symmetric=True):
        super().__init__()
        self.topk = int(topk)
        self.gate_floor = float(gate_floor)
        self.temperature = float(temperature)
        self.symmetric = bool(symmetric)
        if not 0.0 <= self.gate_floor < 1.0:
            raise ValueError("gate_floor must be in [0, 1)")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        self.blocks = nn.ModuleList([
            RelationGatedGraphBlock(hidden_size, num_heads)
            for _ in range(int(depth))
        ])

    def compute_gates(self, relation_logits, strength=1.0):
        if relation_logits.ndim != 3:
            raise ValueError("relation_logits must have shape [B, K, K]")
        if self.symmetric:
            relation_logits = 0.5 * (
                relation_logits + relation_logits.transpose(1, 2)
            )
        gates = torch.sigmoid(relation_logits / self.temperature)
        gates = self.gate_floor + (1.0 - self.gate_floor) * gates

        num_slots = gates.shape[-1]
        diagonal = torch.eye(
            num_slots, dtype=torch.bool, device=gates.device
        ).unsqueeze(0)
        gates = torch.where(diagonal, torch.ones_like(gates), gates)

        # topk counts non-self neighbors. Symmetrize the selected support as
        # well as the logits so the resulting interaction graph is undirected.
        if 0 < self.topk < num_slots - 1:
            neighbor_scores = gates.detach().masked_fill(diagonal, -1.0)
            indices = neighbor_scores.topk(self.topk, dim=-1).indices
            support = torch.zeros_like(gates, dtype=torch.bool)
            support.scatter_(-1, indices, True)
            support = support | support.transpose(1, 2) | diagonal
            gates = torch.where(
                support, gates, torch.full_like(gates, self.gate_floor)
            )

        strength = float(max(0.0, min(1.0, strength)))
        return 1.0 + strength * (gates - 1.0)

    def forward(self, tokens, relation_logits, strength=1.0):
        gates = self.compute_gates(relation_logits, strength=strength)
        for block in self.blocks:
            tokens = block(tokens, gates)
        return tokens

class DiTFeaturePredictor(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_size=256,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        timestep_embed_dim=256,
        use_pos_embed=True,
        param_type: str = "v",
        num_tokens: int = 1,
        latent_action_dim: int = 16,
        graph_depth: int = 0,
        relation_gating: bool = False,
        relation_topk: int = 3,
        relation_gate_floor: float = 0.05,
        relation_gate_temperature: float = 1.0,
    ):
        super().__init__()
        assert hidden_size % num_heads == 0
        assert param_type in ("x0", "eps", "v")
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.param_type = param_type

        self.num_tokens = int(num_tokens)
        self.noise_proj = nn.Linear(input_dim, hidden_size)

        self.vis_proj   = nn.Linear(input_dim, hidden_size)
        self.act_proj   = nn.Linear(latent_action_dim, hidden_size)

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, hidden_size)) if use_pos_embed else None

        self.t_embedder = TimestepEmbedder(hidden_size, frequency_embedding_size=timestep_embed_dim)
        self.ca_proj = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True)
        )

        self.graph_depth = int(graph_depth)
        self.relation_gating = bool(relation_gating)
        if self.graph_depth > 0:
            self.centroid_proj = nn.Sequential(
                nn.Linear(3, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
            )
            if self.relation_gating:
                self.graph_encoder = RelationGatedGraphEncoder(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    depth=self.graph_depth,
                    topk=relation_topk,
                    gate_floor=relation_gate_floor,
                    temperature=relation_gate_temperature,
                )
            else:
                graph_layer = nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=num_heads,
                    dim_feedforward=hidden_size * 4,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                self.graph_encoder = nn.TransformerEncoder(
                    graph_layer, num_layers=self.graph_depth
                )
            self.slot_ca_proj = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size, bias=True),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size, bias=True),
            )

        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)])
        self.output_layer = nn.Linear(hidden_size, input_dim)
        self.pred_norm = nn.LayerNorm(input_dim, elementwise_affine=True)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        for blk in self.blocks:
            last = blk.adaLN_modulation[-1]
            if isinstance(last, nn.Linear):
                nn.init.constant_(last.weight, 0.0)
                if last.bias is not None:
                    nn.init.constant_(last.bias, 0.0)
        if self.pos_embed is not None:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.constant_(self.output_layer.bias, 0.0)

    @staticmethod
    def alpha_sigma_from_alphas_cumprod(t: torch.Tensor, alphas_cumprod: torch.Tensor):
        alpha_bar = alphas_cumprod[t]                   # (B,)
        alpha_t = torch.sqrt(alpha_bar).unsqueeze(-1)   # (B,1)
        sigma_t = torch.sqrt(1.0 - alpha_bar).unsqueeze(-1)
        return alpha_t, sigma_t

    @staticmethod
    def target_from_param(param_type: str, x0, eps, alpha_t, sigma_t, x_t):
        if param_type == "x0":
            return x0
        elif param_type == "eps":
            return eps
        elif param_type == "v":
            return alpha_t * eps - sigma_t * x0
        else:
            raise ValueError(f"Unknown param_type {param_type}")

    @staticmethod
    def pred_to_x0(param_type: str, pred, alpha_t, sigma_t, x_t):
        if param_type == "x0":
            return pred
        elif param_type == "eps":
            return (x_t - sigma_t * pred) / alpha_t
        elif param_type == "v":
            return alpha_t * x_t - sigma_t * pred
        else:
            raise ValueError(f"Unknown param_type {param_type}")

    def forward(self, z, t, vis, act, centroids=None,
                relation_logits=None, relation_gate_strength=1.0):
        """Predict one feature per token.

        Baseline inputs may be [B, D]. Slot inputs are [B, K, D], with a
        possibly independent diffusion timestep per slot [B, K].
        """
        squeeze_token = z.ndim == 2
        if squeeze_token:
            z = z.unsqueeze(1)
            vis = vis.unsqueeze(1)
            act = act.unsqueeze(1)

        if z.ndim != 3:
            raise ValueError("z must have shape [B, D] or [B, K, D]")
        if z.shape[1] > self.num_tokens:
            raise ValueError(
                f"received {z.shape[1]} tokens, configured for {self.num_tokens}"
            )

        x = self.noise_proj(z)                  # (B, K, H)
        if self.pos_embed is not None:
            x = x + self.pos_embed[:, :x.shape[1], :]

        if t.ndim == 1:
            t_emb = self.t_embedder(t).unsqueeze(1).expand(
                -1, x.shape[1], -1
            )
        elif t.ndim == 2:
            t_emb = self.t_embedder(t.reshape(-1)).reshape(
                t.shape[0], t.shape[1], -1
            )
        else:
            raise ValueError("t must have shape [B] or [B, K]")

        v = self.vis_proj(vis)
        a = self.act_proj(act)
        if self.graph_depth > 0:
            if centroids is None:
                raise ValueError("centroids are required for slot-graph mode")
            graph_tokens = v + a + self.centroid_proj(centroids)
            if self.relation_gating:
                if relation_logits is None:
                    raise ValueError(
                        "relation_logits are required for relation-gated mode"
                    )
                graph_condition = self.graph_encoder(
                    graph_tokens,
                    relation_logits,
                    strength=relation_gate_strength,
                )
            else:
                graph_condition = self.graph_encoder(graph_tokens)
            c = self.slot_ca_proj(
                torch.cat([t_emb, graph_condition], dim=-1)
            )
        else:
            c = self.ca_proj(torch.cat([t_emb, v, a], dim=-1))

        for blk in self.blocks:
            x = blk(x, c)

        pred = self.output_layer(x)
        return pred[:, 0] if squeeze_token else pred
