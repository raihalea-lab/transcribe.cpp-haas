"""Write a random-weight .nemo shaped like Haas-ja-v1.0, for testing the
converter and the C++ KDA predictor without the real checkpoint.

The prediction network has Haas's real geometry and parameter names (it is
built from haas_asr.prednet + fla, see kda_reference.py); the encoder is a
tiny FastConformer so the file stays small. The tokenizer is a 3072-piece
SentencePiece BPE with byte fallback trained on random kana/kanji text.

    python haas/make_synthetic_nemo.py out/synthetic.nemo --haas-src ../Haas-ASR/src
"""

from __future__ import annotations

import argparse
import io
import random
import tarfile
import tempfile
from pathlib import Path

import torch
import yaml

from kda_reference import build_prednet

VOCAB = 3072
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
CHANNELS = 32
CONV_K = 9
N_MELS = 128
PRED_H = 640
JOINT_H = 640
DURATIONS = [0, 1, 2, 3, 4]


def config() -> dict:
    return {
        "sample_rate": 16000,
        "preprocessor": {
            "_target_": "nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor",
            "sample_rate": 16000, "normalize": "per_feature", "window_size": 0.025,
            "window_stride": 0.01, "window": "hann", "features": N_MELS, "n_fft": 512,
            "log": True, "frame_splicing": 1, "dither": 1.0e-05, "pad_to": 0, "pad_value": 0.0,
        },
        "encoder": {
            "_target_": "nemo.collections.asr.modules.ConformerEncoder",
            "feat_in": N_MELS, "feat_out": -1, "n_layers": N_LAYERS, "d_model": D_MODEL,
            "use_bias": False, "subsampling": "dw_striding", "subsampling_factor": 8,
            "subsampling_conv_channels": CHANNELS, "causal_downsampling": False,
            "ff_expansion_factor": 4, "self_attention_model": "rel_pos", "n_heads": N_HEADS,
            "att_context_size": [-1, -1], "att_context_style": "regular", "xscaling": False,
            "untie_biases": True, "pos_emb_max_len": 5000, "conv_kernel_size": CONV_K,
            "conv_norm_type": "batch_norm", "conv_context_size": None, "dropout": 0.1,
        },
        "decoder": {
            "_target_": "haas_asr.prednet.KDATransducerDecoder",
            "normalization_mode": None, "random_state_sampling": False, "blank_as_pad": True,
            "vocab_size": VOCAB,
            "prednet": {
                "pred_hidden": PRED_H, "pred_rnn_layers": 2, "t_max": None, "dropout": 0.2,
                "kda_num_heads": 5, "kda_head_dim": 64, "kda_expand_v": 1.0,
                "kda_conv_size": 4, "kda_final_norm": True,
            },
        },
        "joint": {
            "_target_": "nemo.collections.asr.modules.RNNTJoint",
            "log_softmax": None, "preserve_memory": False, "fuse_loss_wer": True,
            "fused_batch_size": 4, "num_classes": VOCAB, "vocabulary": [],
            "num_extra_outputs": len(DURATIONS),
            "jointnet": {"joint_hidden": JOINT_H, "activation": "relu", "dropout": 0.2,
                         "encoder_hidden": D_MODEL, "pred_hidden": PRED_H},
        },
        "decoding": {"strategy": "greedy_batch", "model_type": "tdt", "durations": DURATIONS,
                     "greedy": {"max_symbols": 10}},
        "tokenizer": {"dir": None, "type": "bpe", "model_path": "nemo:synthetic_tokenizer.model"},
    }


def tokenizer_proto(seed: int) -> bytes:
    import sentencepiece as spm

    rng = random.Random(seed)
    kana = [chr(c) for c in range(0x3041, 0x3097)] + [chr(c) for c in range(0x30A1, 0x30FB)]
    kanji = [chr(c) for c in range(0x4E00, 0x4E00 + 400)]
    alphabet = kana + kanji + list("、。ー")
    lines = []
    for _ in range(40000):
        n = rng.randint(4, 30)
        lines.append("".join(rng.choice(alphabet[: 120 if rng.random() < 0.7 else len(alphabet)])
                             for _ in range(n)))
    with tempfile.TemporaryDirectory() as tmp:
        corpus = Path(tmp) / "corpus.txt"
        corpus.write_text("\n".join(lines), encoding="utf-8")
        spm.SentencePieceTrainer.train(
            input=str(corpus), model_prefix=str(Path(tmp) / "sp"), model_type="bpe",
            vocab_size=VOCAB, byte_fallback=True, character_coverage=1.0,
            split_by_unicode_script=False, num_threads=4, minloglevel=2,
        )
        return (Path(tmp) / "sp.model").read_bytes()


def state_dict(cfg: dict, haas_src: str | None, seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)

    def rnd(*shape, std=0.05):
        return torch.randn(*shape, generator=g) * std

    sd: dict[str, torch.Tensor] = {}
    C, d, dff = CHANNELS, D_MODEL, 4 * D_MODEL
    pe = "encoder.pre_encode."
    sd[pe + "conv.0.weight"], sd[pe + "conv.0.bias"] = rnd(C, 1, 3, 3, std=0.3), rnd(C)
    for i in (2, 5):  # depthwise
        sd[pe + f"conv.{i}.weight"], sd[pe + f"conv.{i}.bias"] = rnd(C, 1, 3, 3, std=0.3), rnd(C)
    for i in (3, 6):  # pointwise
        sd[pe + f"conv.{i}.weight"], sd[pe + f"conv.{i}.bias"] = rnd(C, C, 1, 1, std=0.2), rnd(C)
    sd[pe + "out.weight"], sd[pe + "out.bias"] = rnd(d, C * (N_MELS // 8), std=0.02), rnd(d)
    for i in range(N_LAYERS):
        p = f"encoder.layers.{i}."
        for norm in ("norm_feed_forward1", "norm_self_att", "norm_conv", "norm_feed_forward2", "norm_out"):
            sd[p + norm + ".weight"], sd[p + norm + ".bias"] = 1 + rnd(d), rnd(d)
        for ff in ("feed_forward1", "feed_forward2"):
            sd[p + ff + ".linear1.weight"], sd[p + ff + ".linear2.weight"] = rnd(dff, d), rnd(d, dff)
        for lin in ("linear_q", "linear_k", "linear_v", "linear_out", "linear_pos"):
            sd[p + f"self_attn.{lin}.weight"] = rnd(d, d, std=0.1)
        sd[p + "self_attn.pos_bias_u"] = rnd(N_HEADS, d // N_HEADS)
        sd[p + "self_attn.pos_bias_v"] = rnd(N_HEADS, d // N_HEADS)
        sd[p + "conv.pointwise_conv1.weight"] = rnd(2 * d, d, 1, std=0.1)
        sd[p + "conv.depthwise_conv.weight"] = rnd(d, 1, CONV_K, std=0.1)
        sd[p + "conv.pointwise_conv2.weight"] = rnd(d, d, 1, std=0.1)
        sd[p + "conv.batch_norm.weight"], sd[p + "conv.batch_norm.bias"] = 1 + rnd(d), rnd(d)
        sd[p + "conv.batch_norm.running_mean"] = rnd(d)
        sd[p + "conv.batch_norm.running_var"] = 1 + rnd(d).abs()
        sd[p + "conv.batch_norm.num_batches_tracked"] = torch.tensor(100)

    # Prediction network: the real modules, randomized past their init so
    # that every parameter matters (unit norms / zero biases would hide
    # wiring mistakes).
    net = build_prednet(cfg["decoder"]["prednet"], haas_src)
    with torch.no_grad():
        for name, prm in net.named_parameters():
            if name.endswith("A_log"):
                prm.copy_(torch.empty_like(prm).uniform_(0.0, 2.5, generator=g))
            elif name.endswith("dt_bias"):
                prm.copy_(rnd(*prm.shape, std=1.0) - 2.0)
            elif prm.dim() == 1:
                prm.copy_((1.0 if name.endswith("weight") else 0.0) + rnd(*prm.shape, std=0.2))
            else:
                prm.copy_(rnd(*prm.shape, std=1.0 / prm.shape[-1] ** 0.5))
    for k, v in net.state_dict().items():
        sd[f"decoder.prediction.dec_rnn.{k}"] = v.detach().clone()
    embed = rnd(VOCAB + 1, PRED_H, std=1.0)
    embed[VOCAB] = 0.0  # blank = padding_idx row
    sd["decoder.prediction.embed.weight"] = embed

    n_out = VOCAB + 1 + len(DURATIONS)
    sd["joint.enc.weight"], sd["joint.enc.bias"] = rnd(JOINT_H, d, std=0.2), rnd(JOINT_H)
    sd["joint.pred.weight"], sd["joint.pred.bias"] = rnd(JOINT_H, PRED_H, std=0.05), rnd(JOINT_H)
    sd["joint.joint_net.2.weight"], sd["joint.joint_net.2.bias"] = rnd(n_out, JOINT_H, std=0.1), rnd(n_out)
    # Keep blank and short durations competitive so decoding emits tokens.
    sd["joint.joint_net.2.bias"][VOCAB] -= 0.5
    return sd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--haas-src", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = config()
    sd = state_dict(cfg, args.haas_src, args.seed)
    proto = tokenizer_proto(args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    torch.save(sd, buf)
    with tarfile.open(args.out, "w") as tf:
        for name, data in (("./model_config.yaml", yaml.safe_dump(cfg).encode()),
                           ("./model_weights.ckpt", buf.getvalue()),
                           ("./synthetic_tokenizer.model", proto)):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB, {len(sd)} tensors)")


if __name__ == "__main__":
    main()
