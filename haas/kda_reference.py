"""CPU reference for Haas's KDA prediction network, from the real modules.

Builds haas_asr.prednet.KDAPredictionNetwork (Haas-ASR's own code, with
fla 0.5.2's KimiDeltaAttention inside) and runs it one token at a time on
CPU. fla's decode path calls three Triton kernels, which need a GPU; they
are replaced here by torch code taken from fla itself:

  fused_recurrent_kda      -> fla.ops.kda.naive.naive_recurrent_kda, with the
                              l2norm / gate / beta-sigmoid steps the fused
                              kernel does first (fla/ops/kda/fused_recurrent.py)
  ShortConvolution.step    -> the "equivalent to" torch lines in fla's
                              fla/modules/conv/short_conv.py
  FusedRMSNormGated        -> x * rsqrt(mean(x^2) + eps) * w * sigmoid(g)

Everything else (projections, rearranges, cache handling, residuals, norms,
Haas's state packing) is the unmodified upstream code.

Needs: torch, einops, transformers, flash-linear-attention==0.5.2 and a
Haas-ASR checkout (HAAS_ASR_SRC or --haas-src, the directory holding
haas_asr/). NeMo is not needed: prednet.py only subclasses NeMo's
RNNTDecoder, which is stubbed.
"""

from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F

PREFIX = "decoder.prediction."


def _install_stubs(haas_src: Path) -> None:
    # nemo.collections.asr.modules.rnnt.RNNTDecoder is only a base class
    # for KDATransducerDecoder, which the reference does not use.
    if "nemo.collections.asr.modules.rnnt" not in sys.modules:
        for name in ("nemo", "nemo.collections", "nemo.collections.asr", "nemo.collections.asr.modules"):
            sys.modules.setdefault(name, types.ModuleType(name))
        rnnt = types.ModuleType("nemo.collections.asr.modules.rnnt")
        rnnt.RNNTDecoder = torch.nn.Module
        sys.modules["nemo.collections.asr.modules.rnnt"] = rnnt
    # Import haas_asr.prednet without haas_asr/__init__.py (which pulls in NeMo).
    if "haas_asr" not in sys.modules:
        pkg = types.ModuleType("haas_asr")
        pkg.__path__ = [str(haas_src / "haas_asr")]
        sys.modules["haas_asr"] = pkg


def _patch_fla() -> None:
    import fla.layers.kda as kda_layer
    from fla.modules import FusedRMSNormGated, ShortConvolution
    from fla.ops.kda.naive import naive_recurrent_kda

    def fused_recurrent_kda(q, k, v, g, beta, A_log=None, dt_bias=None, scale=None, initial_state=None,
                            output_final_state=False, use_qk_l2norm_in_kernel=False,
                            use_gate_in_kernel=False, use_beta_sigmoid_in_kernel=False,
                            allow_neg_eigval=False, lower_bound=None, state_v_first=False,
                            cu_seqlens=None, **kwargs):
        assert cu_seqlens is None and lower_bound is None and not allow_neg_eigval
        q, k, v, g, beta = (t.float() for t in (q, k, v, g, beta))
        HV, K = v.shape[2], q.shape[-1]
        if use_qk_l2norm_in_kernel:
            q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)
            k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        if use_gate_in_kernel:
            bias = dt_bias.float().view(HV, K) if dt_bias is not None else 0.0
            g = -A_log.float().exp().view(HV, 1) * F.softplus(g + bias)
        if use_beta_sigmoid_in_kernel:
            beta = beta.sigmoid()
        h0 = initial_state
        if h0 is not None and state_v_first:
            h0 = h0.transpose(-1, -2)
        o, ht = naive_recurrent_kda(q, k, v, g, beta, scale=scale, initial_state=h0, output_final_state=True)
        if state_v_first:
            ht = ht.transpose(-1, -2).contiguous()
        return o, (ht if output_final_state else None)

    def short_conv_step(self, x, residual, cache, output_final_state=False, cu_seqlens=None):
        assert cu_seqlens is None
        B, _, D = x.shape
        W = self.kernel_size[0]
        if cache is None:
            cache = x.new_zeros(B, D, W)
        cache.copy_(cache.roll(shifts=-1, dims=-1))
        cache[:, :, -1] = x[:, 0]
        y = (cache * self.weight.view(D, W)).sum(-1)
        if self.bias is not None:
            y = y + self.bias
        if self.activation in ("silu", "swish"):
            y = F.silu(y)
        y = y.unsqueeze(1)
        if residual is not None:
            y = y + residual
        return y, (cache if output_final_state else None)

    def rms_norm_gated(self, x, g, residual=None, prenorm=False, residual_in_fp32=False):
        assert residual is None and not prenorm and self.activation == "sigmoid"
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight.float()
        return (y * torch.sigmoid(g.float())).to(x.dtype)

    kda_layer.fused_recurrent_kda = fused_recurrent_kda
    ShortConvolution.step = short_conv_step
    FusedRMSNormGated.forward = rms_norm_gated


def load_prednet_module(haas_src: str | os.PathLike | None = None):
    """Return haas_asr.prednet with fla patched for CPU."""
    src = Path(haas_src or os.environ.get("HAAS_ASR_SRC", ""))
    if not (src / "haas_asr" / "prednet.py").exists():
        raise SystemExit("set HAAS_ASR_SRC (or --haas-src) to Haas-ASR's src/ directory")
    _install_stubs(src)
    _patch_fla()
    return importlib.import_module("haas_asr.prednet")


def build_prednet(prednet_cfg: dict, haas_src=None):
    """A KDAPredictionNetwork from a decoder.prednet config (eval mode)."""
    prednet = load_prednet_module(haas_src)
    net = prednet.KDAPredictionNetwork(
        hidden_size=int(prednet_cfg["pred_hidden"]),
        num_layers=int(prednet_cfg["pred_rnn_layers"]),
        config=prednet.KDAConfig.from_prednet(prednet_cfg),
    )
    return net.eval()


class ReferencePredictor:
    """The predictor as the transducer decoders use it: embed, then one step."""

    def __init__(self, cfg: dict, state_dict: dict, haas_src=None):
        prednet_cfg = cfg["decoder"]["prednet"]
        self.net = build_prednet(prednet_cfg, haas_src)
        sub = {k[len(PREFIX + "dec_rnn."):]: v for k, v in state_dict.items()
               if k.startswith(PREFIX + "dec_rnn.")}
        self.net.load_state_dict(sub, strict=True)
        self.embed = state_dict[PREFIX + "embed.weight"].float()
        self.hidden = int(prednet_cfg["pred_hidden"])

    @torch.no_grad()
    def step(self, last_token: int, state):
        """Previous token (-1 = start: NeMo's blank, whose embed row is the
        zero padding row) -> (output [H], new state)."""
        if last_token < 0:
            x = torch.zeros(1, 1, self.hidden)
        else:
            x = self.embed[last_token].view(1, 1, -1)
        out, new_state = self.net(x, state)
        return out[0, 0].clone(), new_state


def resolve_nemo(spec: str | os.PathLike) -> Path:
    """A local .nemo, or the one .nemo in a Hugging Face repo (downloaded)."""
    path = Path(spec).expanduser()
    if path.is_file():
        return path
    from huggingface_hub import hf_hub_download, list_repo_files

    names = [n for n in list_repo_files(str(spec)) if n.endswith(".nemo")]
    if len(names) != 1:
        raise SystemExit(f"{spec}: expected one .nemo in the repo, found {names}")
    return Path(hf_hub_download(str(spec), names[0]))


def load_nemo(path: str | os.PathLike):
    """(cfg dict, state_dict) from a .nemo archive (or HF repo id), without NeMo."""
    import io
    import tarfile

    import yaml

    path = resolve_nemo(path)

    with tarfile.open(path) as tf:
        names = tf.getnames()

        def read(suffix):
            for n in names:
                if n.endswith(suffix):
                    return tf.extractfile(n).read()
            raise FileNotFoundError(suffix)

        cfg = yaml.safe_load(read("model_config.yaml"))
        sd = torch.load(io.BytesIO(read("model_weights.ckpt")), map_location="cpu", weights_only=False)
    return cfg, sd
