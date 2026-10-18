"""Arquitectura GPT estándar y Looped Transformer con instrumentación interna."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


DEFAULT_MODEL_CFG: Dict[str, Any] = {
    "arch": "looped",          # "gpt" | "looped"
    "vocab_size": 100,         # Tamaño del vocabulario de caracteres
    "context_length": 512,     # Longitud máxima de secuencia
    "emb_dim": 256,            # Dimensión del estado oculto (d)
    "n_heads": 8,              # Número de cabezas de atención (head_dim = 32)
    "n_layers": 1,             # Bloques con pesos distintos (L en GPT, 1 en Looped)
    "n_loops": 4,              # Iteraciones del bucle recurrente (N en Looped)
    "drop_rate": 0.0,          # Sin dropout sobre datos sintéticos
    "qkv_bias": False,         # Sin sesgo en las proyecciones Q, K, V
}


class MultiHeadAttention(nn.Module):
    """Atención multi-cabeza causal con extracción opcional de mapas de atención."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        context_length: int,
        dropout: float,
        num_heads: int,
        qkv_bias: bool = False,
    ) -> None:
        super().__init__()
        if d_out % num_heads != 0:
            raise ValueError("d_out debe ser divisible por num_heads")

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        b, num_tokens, _ = x.shape

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / (keys.shape[-1] ** 0.5), dim=-1)
        attn_weights_dropped = self.dropout(attn_weights)

        context_vec = (attn_weights_dropped @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        if return_attn:
            return context_vec, attn_weights
        return context_vec


class LayerNorm(nn.Module):
    """Normalización por capa (Layer Normalization)."""

    def __init__(self, emb_dim: int) -> None:
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    """Activación Gaussian Error Linear Unit (aproximación con tanh)."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (
            1.0 + torch.tanh(
                torch.sqrt(torch.tensor(2.0 / torch.pi, device=x.device, dtype=x.dtype))
                * (x + 0.044715 * torch.pow(x, 3))
            )
        )


class FeedForward(nn.Module):
    """Red feed-forward posicional con expansión 4x."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TransformerBlock(nn.Module):
    """Bloque Transformer Pre-LayerNorm con conexiones residuales."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(
        self,
        x: torch.Tensor,
        return_attn: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        shortcut = x
        x = self.norm1(x)
        if return_attn:
            x, attn_weights = self.att(x, return_attn=True)
        else:
            x = self.att(x, return_attn=False)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        if return_attn:
            return x, attn_weights
        return x


class GPTModel(nn.Module):
    """Modelo GPT estándar de L capas con pesos independientes por capa."""

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        self.cfg = copy.deepcopy(cfg)
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def embed(self, in_idx: torch.Tensor) -> torch.Tensor:
        """Calcula el estado inicial h_0 sumando embeddings de token y posición."""
        _, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        return self.drop_emb(tok_embeds + pos_embeds)

    def head(self, h: torch.Tensor) -> torch.Tensor:
        """Proyecta cualquier estado oculto intermedio o final a logits del vocabulario (Logit Lens)."""
        return self.out_head(self.final_norm(h))

    def forward(
        self,
        in_idx: torch.Tensor,
        n_loops: Optional[int] = None,
        return_hidden: bool = False,
        return_attn: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, List[torch.Tensor]]]]:
        x = self.embed(in_idx)
        hidden_states: List[torch.Tensor] = [x] if return_hidden else []
        attn_maps: List[torch.Tensor] = []

        for block in self.trf_blocks:
            if return_attn:
                x, attn_w = block(x, return_attn=True)
                attn_maps.append(attn_w)
            else:
                x = block(x, return_attn=False)
            if return_hidden:
                hidden_states.append(x)

        logits = self.head(x)
        if return_hidden or return_attn:
            extras: Dict[str, List[torch.Tensor]] = {}
            if return_hidden:
                extras["hidden"] = hidden_states
            if return_attn:
                extras["attn"] = attn_maps
            return logits, extras
        return logits


class LoopedGPT(GPTModel):
    """
    Looped Transformer que hereda de GPTModel y aplica sus bloques de forma
    recurrente durante N iteraciones (con pesos compartidos entre vueltas).
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        full_cfg = {**DEFAULT_MODEL_CFG, **cfg}
        super().__init__(full_cfg)
        self.cfg = full_cfg
        self.n_layers = int(full_cfg["n_layers"])
        self.n_loops = int(full_cfg["n_loops"])

    @property
    def shared_blocks(self) -> nn.Sequential:
        """Alias hacia self.trf_blocks para explicitar que los pesos se comparten."""
        return self.trf_blocks

    def forward(
        self,
        in_idx: torch.Tensor,
        n_loops: Optional[int] = None,
        return_hidden: bool = False,
        return_attn: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, List[torch.Tensor]]]]:
        loops = self.n_loops if n_loops is None else int(n_loops)
        x = self.embed(in_idx)

        hidden_states: List[torch.Tensor] = [x] if return_hidden else []
        attn_maps: List[torch.Tensor] = []

        for _ in range(loops):
            for block in self.trf_blocks:
                if return_attn:
                    x, attn_w = block(x, return_attn=True)
                    attn_maps.append(attn_w)
                else:
                    x = block(x, return_attn=False)
                if return_hidden:
                    hidden_states.append(x)

        logits = self.head(x)
        if return_hidden or return_attn:
            extras: Dict[str, List[torch.Tensor]] = {}
            if return_hidden:
                extras["hidden"] = hidden_states
            if return_attn:
                extras["attn"] = attn_maps
            return logits, extras
        return logits


def build_model(cfg: Dict[str, Any]) -> nn.Module:
    """Factoría unificada que construye GPTModel o LoopedGPT a partir de un diccionario cfg."""
    full_cfg = {**DEFAULT_MODEL_CFG, **cfg}
    arch = full_cfg.get("arch", "gpt")
    if arch == "gpt":
        return GPTModel(full_cfg)
    if arch == "looped":
        return LoopedGPT(full_cfg)
    raise ValueError(f"Arquitectura desconocida: {arch!r}. Usa 'gpt' o 'looped'.")


def count_params(model: nn.Module, trainable_only: bool = True) -> int:
    """Cuenta el número total de parámetros de un modelo."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def effective_depth(cfg: Dict[str, Any]) -> int:
    """Profundidad computacional efectiva: L en GPT estándar, n_layers * n_loops en Looped."""
    if cfg.get("arch", "gpt") == "gpt":
        return int(cfg["n_layers"])
    return int(cfg.get("n_layers", 1)) * int(cfg["n_loops"])


def block_flops_per_token(cfg: Dict[str, Any], seq_len: Optional[int] = None) -> int:
    """
    FLOPs aproximados (2 * MACs) por token en un único bloque Transformer:
      - Proyecciones Q, K, V y salida: 4 * (2 * d^2) = 8 * d^2
      - Atención QK^T y AV:            2 * (2 * T * d) = 4 * T * d
      - Red FeedForward (d -> 4d -> d): 2 * (2 * 4 * d^2) = 16 * d^2
      Total por bloque = 24 * d^2 + 4 * T * d
    """
    d = int(cfg["emb_dim"])
    T = int(seq_len if seq_len is not None else cfg["context_length"])
    return 24 * (d ** 2) + 4 * T * d


def flops_per_token(cfg: Dict[str, Any], seq_len: Optional[int] = None) -> int:
    """FLOPs totales de forward por token incluyendo todos los pasos efectivos y la cabeza."""
    d = int(cfg["emb_dim"])
    V = int(cfg["vocab_size"])
    steps = effective_depth(cfg)
    return steps * block_flops_per_token(cfg, seq_len=seq_len) + 2 * d * V


def gpt_cfg_from(cfg: Dict[str, Any], n_layers: int) -> Dict[str, Any]:
    """Crea una configuración de GPTModel estándar con n_layers capas."""
    out = copy.deepcopy(cfg)
    out["arch"] = "gpt"
    out["n_layers"] = int(n_layers)
    out["n_loops"] = 1
    return out


def unroll_to_gpt(looped_model: LoopedGPT, n_loops: Optional[int] = None) -> GPTModel:
    """
    Desenrolla un LoopedGPT en un GPTModel estándar de profundidad efectiva L * N
    copiando los pesos del bloque compartido en cada capa.
    """
    loops = looped_model.n_loops if n_loops is None else int(n_loops)
    total_layers = looped_model.n_layers * loops
    gpt_cfg = gpt_cfg_from(looped_model.cfg, n_layers=total_layers)
    gpt = GPTModel(gpt_cfg)

    with torch.no_grad():
        gpt.tok_emb.load_state_dict(looped_model.tok_emb.state_dict())
        gpt.pos_emb.load_state_dict(looped_model.pos_emb.state_dict())
        gpt.final_norm.load_state_dict(looped_model.final_norm.state_dict())
        gpt.out_head.load_state_dict(looped_model.out_head.state_dict())

        idx = 0
        for _ in range(loops):
            for k in range(looped_model.n_layers):
                gpt.trf_blocks[idx].load_state_dict(
                    looped_model.trf_blocks[k].state_dict()
                )
                idx += 1
    return gpt


def looped_from_gpt(
    gpt_model: GPTModel,
    n_loops: Optional[int] = None,
    mode: str = "first",
) -> LoopedGPT:
    """
    Construye un LoopedGPT (n_layers=1) a partir de un GPTModel inicializando
    el bloque único mediante 'first', 'last', 'middle' o 'mean'.
    """
    L = len(gpt_model.trf_blocks)
    loops = L if n_loops is None else int(n_loops)
    loop_cfg = copy.deepcopy(gpt_model.cfg)
    loop_cfg["arch"] = "looped"
    loop_cfg["n_layers"] = 1
    loop_cfg["n_loops"] = loops
    looped = LoopedGPT(loop_cfg)

    with torch.no_grad():
        looped.tok_emb.load_state_dict(gpt_model.tok_emb.state_dict())
        looped.pos_emb.load_state_dict(gpt_model.pos_emb.state_dict())
        looped.final_norm.load_state_dict(gpt_model.final_norm.state_dict())
        looped.out_head.load_state_dict(gpt_model.out_head.state_dict())

        if mode == "first":
            looped.trf_blocks[0].load_state_dict(gpt_model.trf_blocks[0].state_dict())
        elif mode == "last":
            looped.trf_blocks[0].load_state_dict(gpt_model.trf_blocks[-1].state_dict())
        elif mode == "middle":
            looped.trf_blocks[0].load_state_dict(gpt_model.trf_blocks[L // 2].state_dict())
        elif mode == "mean":
            avg_state = copy.deepcopy(gpt_model.trf_blocks[0].state_dict())
            for key in avg_state:
                stacked = torch.stack(
                    [gpt_model.trf_blocks[i].state_dict()[key].float() for i in range(L)],
                    dim=0,
                )
                avg_state[key] = stacked.mean(dim=0).to(avg_state[key].dtype)
            looped.trf_blocks[0].load_state_dict(avg_state)
        else:
            raise ValueError(f"Modo desconocido: {mode!r}")

    return looped
