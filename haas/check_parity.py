"""Compare transcribe.cpp's KDA predictor with the PyTorch reference.

    python haas/check_parity.py <model.nemo | HF repo id> <model.gguf> --haas-src ../Haas-ASR/src \
        [--parity-bin build/bin/haas-kda-parity]

(haas/parity.sh wraps this with the right package versions.)

1. steps: the predictor output after each token of a random token sequence
   (state carried across steps), C++ vs reference (kda_reference.py).
2. tdt:   TDT greedy decoding of random encoder frames with the C++ decoder
   (decoder.cpp, including its predictor dispatch and state commits) vs the
   same loop in Python on the reference predictor and the .nemo's joint.

Use the F32 GGUF for a tight comparison. A Q8_0 GGUF drifts in (1) (pass
--tol 2e-2) and may legitimately change a near-tie token in (2).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from kda_reference import ReferencePredictor, load_nemo


def run(cmd: list[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"{' '.join(cmd)} failed:\n{r.stdout}\n{r.stderr}")


def check_steps(pred: ReferencePredictor, vocab: int, gguf: str, binary: str, n: int, seed: int,
                tol: float) -> bool:
    rng = np.random.default_rng(seed)
    tokens = [-1] + rng.integers(0, vocab, size=n - 1).tolist()
    with tempfile.TemporaryDirectory() as tmp:
        tok_path, out_path = Path(tmp) / "tokens.txt", Path(tmp) / "out.f32"
        tok_path.write_text(" ".join(map(str, tokens)))
        run([binary, "steps", gguf, str(tok_path), str(out_path)])
        got = np.fromfile(out_path, dtype=np.float32).reshape(len(tokens), -1)

    state, ref = None, []
    for t in tokens:
        out, state = pred.step(t, state)
        ref.append(out.numpy())
    ref = np.stack(ref)
    err = np.abs(got - ref).max(axis=1)
    rel = err / np.abs(ref).max(axis=1)
    rms = np.sqrt(((got - ref) ** 2).mean(axis=1) / (ref ** 2).mean(axis=1))
    print(f"[steps] {len(tokens)} steps: max |diff| {err.max():.3e} (first {err[0]:.3e}, last {err[-1]:.3e}), "
          f"max rel {rel.max():.3e}, rms rel {rms.max():.3e}")
    ok = bool(rms.max() < tol)
    print(f"[steps] {'PASS' if ok else 'FAIL'}")
    return ok


def check_tdt(pred: ReferencePredictor, cfg: dict, sd: dict, gguf: str, binary: str, T: int, seed: int) -> bool:
    d_enc = int(cfg["encoder"]["d_model"])
    vocab = int(cfg["decoder"]["vocab_size"])
    durations = list(cfg["decoding"]["durations"])
    max_symbols = int((cfg["decoding"].get("greedy") or {}).get("max_symbols") or 10)
    rng = np.random.default_rng(seed)
    enc = rng.standard_normal((T, d_enc)).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        enc_path, out_path = Path(tmp) / "enc.f32", Path(tmp) / "tokens.txt"
        enc.tofile(enc_path)
        run([binary, "tdt", gguf, str(enc_path), str(T), str(out_path)])
        got = [tuple(map(int, line.split())) for line in out_path.read_text().splitlines()]

    f = lambda k: sd[k].double().numpy()  # noqa: E731
    enc_proj = enc.astype(np.float64) @ f("joint.enc.weight").T + f("joint.enc.bias")
    w_pred, b_pred = f("joint.pred.weight"), f("joint.pred.bias")
    w_out, b_out = f("joint.joint_net.2.weight"), f("joint.joint_net.2.bias")
    blank = vocab

    ref, step, new_symbols, last, state, g, dirty = [], 0, 0, -1, None, None, True
    cand = None
    while step < T:
        if dirty:
            g, cand = pred.step(last, state)
            dirty = False
        h = np.maximum(enc_proj[step] + g.double().numpy() @ w_pred.T + b_pred, 0.0)
        logits = h @ w_out.T + b_out
        tok = int(np.argmax(logits[: vocab + 1]))
        dur = durations[int(np.argmax(logits[vocab + 1:]))]
        is_blank = tok == blank
        if not is_blank:
            ref.append((tok, step, dur))
            last, state, dirty = tok, cand, True
        step += dur
        new_symbols += 1
        if dur != 0:
            new_symbols = 0
        elif max_symbols > 0 and (new_symbols >= max_symbols or is_blank):
            step += 1  # blank with duration 0 repeats until max_symbols forces a step
            new_symbols = 0

    same = got == ref
    print(f"[tdt] T={T}: C++ {len(got)} tokens, reference {len(ref)} tokens, identical: {same}")
    if not same:
        for i, (a, b) in enumerate(zip(got, ref)):
            if a != b:
                print(f"[tdt] first difference at token {i}: C++ {a} vs reference {b}")
                break
    print(f"[tdt] {'PASS' if same else 'FAIL'}")
    return same


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("nemo")
    ap.add_argument("gguf")
    ap.add_argument("--haas-src", default=None)
    ap.add_argument("--parity-bin", default=str(Path(__file__).resolve().parent.parent / "build/bin/haas-kda-parity"))
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol", type=float, default=1e-5,
                    help="max per-step rms relative error (F32: 1e-5; try 2e-2 for Q8_0)")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    cfg, sd = load_nemo(args.nemo)
    pred = ReferencePredictor(cfg, sd, args.haas_src)
    vocab = int(cfg["decoder"]["vocab_size"])
    ok = check_steps(pred, vocab, args.gguf, args.parity_bin, args.steps, args.seed, args.tol)
    ok &= check_tdt(pred, cfg, sd, args.gguf, args.parity_bin, args.frames, args.seed)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
