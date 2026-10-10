"""Standalone inference operators preserving native names, sharing and arithmetic.

Do not import training code here: this file is included in the HF artifact.
"""

import math
import torch
from torch import nn
from torch.nn import functional as F


class ExportNorm(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        shape = tuple(spec["shape"])
        self.register_parameter(
            "weight", nn.Parameter(torch.ones(shape)) if spec["weight"] else None
        )
        self.register_parameter("bias", nn.Parameter(torch.zeros(shape)) if spec["bias"] else None)

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            dtype = x.dtype
            y = x.float() if self.spec["full_precision"] else x
            if self.spec["norm_type"] == "layer":
                y = F.layer_norm(
                    y,
                    tuple(self.spec["shape"]),
                    None if self.weight is None else self.weight.type_as(y),
                    None if self.bias is None else self.bias.type_as(y),
                    self.spec["eps"],
                )
            else:
                y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.spec["eps"])
                if self.spec["norm_type"] == "qwen_rms":
                    y = y.to(dtype)
                if self.weight is not None:
                    weight = (
                        self.weight.to(dtype)
                        if self.spec.get("weight_input_dtype")
                        else self.weight
                    )
                    y = weight.type_as(y) * y
                if self.bias is not None:
                    bias = self.bias.to(dtype) if self.spec.get("weight_input_dtype") else self.bias
                    y = y + bias.type_as(y)
            return y.to(dtype)


def nonlinear(x, name, norm=None):
    if name == "identity":
        return x
    if name == "silu":
        return F.silu(x)
    if name == "silu_rms_norm":
        x = F.silu(x)
    if name in ("rms_norm", "silu_rms_norm") and norm is not None:
        return norm(x)
    raise ValueError(f"Unsupported or incomplete nonlinearity {name}")


def install_children(module, spec):
    for name, child in spec.get("children", {}).items():
        setattr(module, name, build_op(child))


class ExportProjection(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        install_children(self, spec)
        if spec["kind"] == "adaptive":
            self.alpha = nn.Parameter(torch.zeros(spec["alpha_shape"], dtype=torch.float32))

    def activate(self, latent):
        if self.spec["kind"] == "adaptive":
            alpha = self.alpha.float()
            return ((1 - alpha) * latent.float() + alpha * F.silu(latent).float()).to(latent.dtype)
        return nonlinear(latent, self.spec["nonlinearity"], getattr(self, "nonlinear_norm", None))

    def forward(self, x):
        return self.u(self.activate(self.v(x)))


class ExportRotation(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.weight = nn.Parameter(torch.zeros(spec["groups"], spec["rank"], dtype=torch.float32))
        self.register_parameter(
            "residual_gate",
            nn.Parameter(torch.zeros(spec["groups"], dtype=torch.float32))
            if spec["residual_gate"]
            else None,
        )

    def forward(self, y, z):
        with torch.autocast(device_type=y.device.type, enabled=False):
            cfg = self.spec
            if cfg["mode"] == "static":
                controller = torch.ones(cfg["rank"], dtype=torch.float32, device=z.device)
            else:
                controller = z.float()
                controller = controller * torch.rsqrt(
                    controller.square().mean(-1, keepdim=True) + 1e-6
                )
            a = cfg["a_max"] * torch.tanh(F.linear(controller, self.weight.float()))
            c, s = (1 - a.square()) / (1 + a.square()), (2 * a) / (1 + a.square())
            pairs = y.float().reshape(*y.shape[:-1], cfg["groups"], -1, 2)
            even, odd = pairs[..., 0], pairs[..., 1]
            c, s = c.unsqueeze(-1), s.unsqueeze(-1)
            out = torch.stack((c * even - s * odd, s * even + c * odd), -1)
            if self.residual_gate is not None:
                gate = torch.tanh(self.residual_gate.float()).reshape(
                    *((1,) * (pairs.ndim - 3)), cfg["groups"], 1, 1
                )
                out = pairs + gate * (out - pairs)
            return out.reshape_as(y).to(y.dtype)


def rational_clip(x, beta):
    direct_x = x.clamp(min=-beta, max=beta)
    direct = direct_x * torch.rsqrt(1 + (direct_x / beta).square())
    reciprocal = x.sign() * beta * torch.rsqrt(1 + (beta / x.abs().clamp_min(beta)).square())
    return torch.where(x.abs() <= beta, direct, reciprocal)


def channel_control(gate, up, cfg):
    if cfg is None or cfg["control"] == "standard" or cfg["scope"] == "none":
        return F.silu(gate) * up
    dtype = gate.dtype
    if dtype in (torch.float16, torch.bfloat16):
        gate, up = gate.float(), up.float()
    cg, cu = cfg["scope"] in ("both", "gate"), cfg["scope"] in ("both", "up")
    control, bg, bu = cfg["control"], cfg["beta_gate"], cfg["beta_up"]
    factor = F.silu(gate)
    if control in ("situ", "situ_rms"):
        if cg:
            factor = bg * torch.tanh(gate / bg) * torch.sigmoid(gate)
        if cu:
            if control == "situ_rms":
                rms = (
                    up.square().mean(-1, keepdim=True).clamp_min(torch.finfo(up.dtype).tiny).sqrt()
                )
                beta = cfg["rms_beta_scale"] * rms
                up = beta * torch.tanh(up / beta)
            else:
                up = bu * torch.tanh(up / bu)
    elif control == "asymmetric_rational_clip":
        if cu:
            up = rational_clip(up, bu)
        if cg:
            linear = torch.where(gate > 0, rational_clip(gate, bg), gate)
            linear = torch.where(torch.isneginf(linear), torch.zeros_like(linear), linear)
            factor = linear * torch.sigmoid(gate)
    elif control in ("dpskv4_clip", "dpskv4_clip_situ"):
        if cg:
            factor = F.silu(gate.clamp_max(10.0))
        if cu:
            up = up.clamp(-10.0, 10.0) if control == "dpskv4_clip" else bu * torch.tanh(up / bu)
    else:
        raise ValueError(f"Unknown channel control: {control}")
    return (factor * up).to(dtype)


class ExportDenseFFN(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        install_children(self, spec)

    def forward(self, x):
        gate, up = self.w1(x), self.w3(x)
        if self.spec.get("channel_control"):
            hidden = channel_control(gate, up, self.spec["channel_control"])
        elif self.spec["activation"] == "gelu":
            hidden = F.gelu(gate, approximate=self.spec["gelu_approximate"]) * up
        else:
            hidden = F.silu(gate) * up
        return self.w2(hidden)


class ExportMoSEFFN(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        install_children(self, spec)
        for name, shape in spec["bias_shapes"].items():
            self.register_parameter(
                name, None if shape is None else nn.Parameter(torch.zeros(shape))
            )

    def forward(self, x):
        gate, up = None, None
        if getattr(self, "linear_u", None) is not None:
            linear_gate = linear_up = self.linear_u(x)
        elif getattr(self, "gate_linear_u", None) is not None:
            linear_gate, linear_up = self.gate_linear_u(x), self.up_linear_u(x)
        else:
            linear_gate = linear_up = None
        if linear_gate is not None:
            gate, up = self.gate_linear_v(linear_gate), self.up_linear_v(linear_up)
        if getattr(self, "nonlinear_u", None) is not None:
            gate_input = up_input = self.nonlinear_u(x)
            gate_norm = up_norm = getattr(self, "gate_up_nonlinear_norm", None)
        elif getattr(self, "gate_nonlinear_u", None) is not None:
            gate_input, up_input = self.gate_nonlinear_u(x), self.up_nonlinear_u(x)
            gate_norm, up_norm = (
                getattr(self, "gate_nonlinear_norm", None),
                getattr(self, "up_nonlinear_norm", None),
            )
        else:
            gate_input = up_input = None
        if gate_input is not None:
            g = self.gate_nonlinear_v(
                nonlinear(gate_input, self.spec["gate_nonlinearity"], gate_norm)
            )
            u = self.up_nonlinear_v(nonlinear(up_input, self.spec["up_nonlinearity"], up_norm))
            gate, up = (g if gate is None else gate + g), (u if up is None else up + u)
        if self.gate_bias is not None:
            gate, up = gate + self.gate_bias, up + self.up_bias
        hidden = channel_control(gate, up, self.spec["channel_control"])
        if getattr(self, "w_down", None) is not None:
            return self.w_down(hidden)
        out = None
        if getattr(self, "down_linear_u", None) is not None:
            out = self.down_linear_v(self.down_linear_u(hidden))
        if getattr(self, "down_nonlinear_u", None) is not None:
            latent = nonlinear(
                self.down_nonlinear_u(hidden),
                self.spec["down_nonlinearity"],
                getattr(self, "down_nonlinear_norm", None),
            )
            contribution = self.down_nonlinear_v(latent)
            if getattr(self, "output_subspace", None) is not None:
                contribution = self.output_subspace(contribution, latent)
            out = contribution if out is None else out + contribution
        return out if self.down_bias is None else out + self.down_bias


def apply_rope(tensor, position_ids, spec):
    dtype = tensor.dtype
    with torch.autocast(device_type=tensor.device.type, enabled=False):
        freqs = torch.tensor(spec["inv_freq"], dtype=torch.float32, device=tensor.device)
        angles = position_ids.float().unsqueeze(-1) * freqs
        angles = torch.cat((angles, angles), -1).unsqueeze(1)
        c, s = angles.cos() * spec["attention_rescale"], angles.sin() * spec["attention_rescale"]
        value = tensor.float() if spec["full_precision"] else tensor
        c, s = c.to(value.dtype), s.to(value.dtype)
        rotate, rest = value[..., : spec["rotary_dim"]], value[..., spec["rotary_dim"] :]
        first, second = rotate.chunk(2, -1)
        rotated = rotate * c + torch.cat((-second, first), -1) * s
        return torch.cat((rotated, rest), -1).to(dtype)


class ExportAttention(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        install_children(self, spec)
        for alias, canonical in spec["input_aliases"]:
            getattr(self, alias).v = getattr(self, canonical).v

    def forward(
        self,
        x,
        *,
        position_ids,
        cache_position,
        attention_mask=None,
        past_key_values=None,
        layer_idx=0,
        output_attentions=False,
    ):
        cfg = self.spec
        batch, length, _ = x.shape
        q, k, v = self.w_q(x), self.w_k(x), self.w_v(x)
        if cfg["clip_qkv"] is not None:
            q, k, v = [t.clamp(-cfg["clip_qkv"], cfg["clip_qkv"]) for t in (q, k, v)]
        if not cfg["use_head_qk_norm"]:
            q = q if self.q_norm is None else self.q_norm(q)
            k = k if self.k_norm is None else self.k_norm(k)
        q = q.view(batch, length, cfg["n_heads"], cfg["head_dim"])
        k, v = [t.view(batch, length, cfg["n_kv_heads"], cfg["head_dim"]) for t in (k, v)]
        if cfg["use_head_qk_norm"]:
            q = q if self.q_norm is None else self.q_norm(q)
            k = k if self.k_norm is None else self.k_norm(k)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
        if cfg["rope"] is not None:
            q, k = (
                apply_rope(q, position_ids, cfg["rope"]),
                apply_rope(k, position_ids, cfg["rope"]),
            )
        if past_key_values is not None:
            k, v = past_key_values.update(k, v, layer_idx, {"cache_position": cache_position})
        repeats = cfg["n_heads"] // cfg["n_kv_heads"]
        if repeats != 1:
            k, v = k.repeat_interleave(repeats, dim=1), v.repeat_interleave(repeats, dim=1)
        key_pos = torch.arange(k.shape[-2], device=x.device)
        distances = cache_position.reshape(-1, 1) - key_pos.reshape(1, -1)
        allowed = distances >= 0
        if cfg["window_size"] is not None:
            allowed = allowed & (distances < cfg["window_size"])
        allowed = allowed[None, None, :, :]
        additive = None
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                if attention_mask.shape[-1] != k.shape[-2]:
                    raise ValueError("attention_mask must include all cached and current tokens")
                allowed = allowed & attention_mask[:, None, None, :].bool()
            elif attention_mask.ndim == 4:
                if attention_mask.dtype == torch.bool:
                    allowed = allowed & attention_mask
                else:
                    additive = attention_mask.to(q.dtype)
            else:
                raise ValueError("attention_mask must have rank 2 or 4")
        mask = allowed if additive is None else additive.masked_fill(~allowed, float("-inf"))
        weights = None
        if output_attentions:
            scale = cfg["scale"] if cfg["scale"] is not None else 1 / math.sqrt(cfg["head_dim"])
            scores = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
            scores = (
                scores.masked_fill(~mask, float("-inf"))
                if mask.dtype == torch.bool
                else scores + mask
            )
            weights = torch.nan_to_num(scores.softmax(-1)).to(q.dtype)
            weights = F.dropout(weights, p=cfg["dropout"], training=self.training)
            att = torch.matmul(weights, v)
        else:
            att = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=cfg["dropout"] if self.training else 0.0,
                scale=cfg["scale"],
            )
        att = att.transpose(1, 2).contiguous()
        if cfg["gate"] is not None:
            g = self.w_g(x)
            g = g.float() if cfg["gate"]["full_precision"] else g
            values = g.sigmoid().to(att.dtype)
            if cfg["gate"]["granularity"] == "headwise":
                att = att * values.unsqueeze(-1)
            elif cfg["gate"]["granularity"] == "elementwise":
                att = att.reshape(batch, length, -1) * values
            else:
                raise ValueError("Unsupported attention gate granularity")
        return self.w_out(att.reshape(batch, length, -1)), weights


class ExportBlock(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        install_children(self, spec)

    def forward(self, x, **attention_kwargs):
        cfg = self.spec
        att_input = self.attention_norm(x) if cfg["norm_order"] != "post" else x
        att, weights = self.attention(att_input, **attention_kwargs)
        if cfg["norm_order"] == "post":
            att = self.attention_norm(att)
        elif cfg["norm_order"] == "peri":
            att = self.post_attention_norm(att)
        h = torch.add(
            x, F.dropout(att, cfg["attention_dropout"], self.training), alpha=cfg["attention_alpha"]
        )
        ff_input = self.feed_forward_norm(h) if cfg["norm_order"] != "post" else h
        ff = self.feed_forward(ff_input)
        if cfg["norm_order"] == "post":
            ff = self.feed_forward_norm(ff)
        elif cfg["norm_order"] == "peri":
            ff = self.post_feed_forward_norm(ff)
        return torch.add(
            h, F.dropout(ff, cfg["ffn_dropout"], self.training), alpha=cfg["ffn_alpha"]
        ), weights


class ExportResidualMoSEFFN(ExportMoSEFFN):
    """Same tensor names, with both residual switches serialized in the spec."""

    def __init__(self, spec):
        if spec.get("latent_source") != "up":
            raise ValueError("Residual FFN requires Up latent as its skip source")
        if spec["wide_residual"] and spec["hidden_size"] != spec["d_model"]:
            raise ValueError("Wide FFN residual has incompatible dimensions")
        super().__init__(spec)

    def forward(self, x):
        gate_latent = self.gate_nonlinear_u(x)
        up_latent = self.up_nonlinear_u(x)
        hidden = F.silu(self.gate_nonlinear_v(gate_latent)) * self.up_nonlinear_v(up_latent)
        down_input = hidden + x if self.spec["wide_residual"] else hidden
        down_latent = self.down_nonlinear_u(down_input)
        if self.spec["latent_residual"]:
            down_latent = down_latent + up_latent
        out = self.down_nonlinear_v(self.down_nonlinear_norm(down_latent))
        return out + hidden if self.spec["wide_residual"] else out


def build_op(spec):
    if spec is None:
        return None
    kind = spec["kind"]
    if kind == "linear":
        return nn.Linear(spec["in_features"], spec["out_features"], bias=spec["bias"])
    if kind == "embedding":
        return nn.Embedding(spec["num_embeddings"], spec["embedding_dim"], spec["padding_idx"])
    builders = {
        "norm": ExportNorm,
        "low_rank": ExportProjection,
        "adaptive": ExportProjection,
        "rotation": ExportRotation,
        "ffn_dense": ExportDenseFFN,
        "ffn_mose": ExportMoSEFFN,
        "ffn_residual_mose": ExportResidualMoSEFFN,
        "attention": ExportAttention,
        "block": ExportBlock,
    }
    if kind not in builders:
        raise ValueError(f"HF runtime has no operator for {kind}")
    return builders[kind](spec)
