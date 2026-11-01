// arch/parakeet/kda_predictor.h - Kimi Delta Attention prediction network.
//
// Haas (third-intelligence/Haas-ja-v1.0) is a Parakeet TDT whose
// prediction network is a stack of pre-norm residual KDA blocks instead of
// NeMo's LSTM (haas_asr/prednet.py, KDAPredictionNetwork):
//
//     x = x + kda(LayerNorm(x))        per block
//     x = LayerNorm(x)                 once, when final_norm
//
// One KDA layer, for a single token (fla 0.5.2, fla/layers/kda.py with
// use_short_conv, the recurrent kernel and state_v_first):
//
//     q, k, v = SiLU(causal_conv1d(q/k/v_proj(x)))       depthwise, no bias
//     g       = -exp(A_log[h]) * softplus(f_proj(x) + dt_bias)
//     beta    = sigmoid(b_proj(x))
//     q, k    = per-head l2norm (x / sqrt(sum(x^2) + 1e-6)); q *= 1/sqrt(K)
//     S       = S * exp(g); S += k (x) beta * (v - S^T k); o = S^T q
//     o       = RMSNorm_head(o) * w * sigmoid(g_proj(x))
//     out     = o_proj(o)
//
// The encoder, joint and TDT loop are Parakeet's, so this file only
// supplies the predictor step; decoder.cpp dispatches to it. The state of
// one step (per layer: recurrent memory + the short-conv history) travels
// in LstmState::kda so every decoder in decoder.cpp keeps its LstmState
// bookkeeping (commit on emission, checkpoints, streaming carry-over)
// unchanged. LstmState::h[l] holds the residual stream after block l;
// h.back() is the predictor output (after the final norm).
//
// Everything here runs on the decoder's CPU backend, one token per step,
// like the LSTM predictor. Weights are dequantized to fp32 at load.

#pragma once

#include "ggml-backend.h"
#include "ggml.h"
#include "transcribe.h"

#include <cstdint>
#include <vector>

struct gguf_context;

namespace transcribe::parakeet {

struct LstmState;

// stt.parakeet.predictor.kind = "kda" and stt.parakeet.predictor.kda.*.
// `enabled` is false for every LSTM GGUF (kind absent or "lstm").
struct KdaHParams {
    bool    enabled    = false;
    int32_t num_heads  = 0;      // query/key heads == value heads (no GVA)
    int32_t head_dim   = 0;      // K == V (expand_v = 1)
    int32_t conv_size  = 0;      // short-conv kernel, includes the current token
    bool    final_norm = false;
    float   norm_eps   = 1e-5f;  // block + final LayerNorm
    float   o_norm_eps = 1e-5f;  // per-head gated RMSNorm

    int32_t key_dim() const { return num_heads * head_dim; }
};

transcribe_status read_kda_hparams(const gguf_context * gguf, KdaHParams & hp);

// Model-resident tensors (borrowed from ctx_meta), validated at load.
struct KdaLayerTensors {
    ggml_tensor * norm_w    = nullptr;  // [H]          pred.kda.{i}.norm_in.weight
    ggml_tensor * norm_b    = nullptr;  // [H]          pred.kda.{i}.norm_in.bias
    ggml_tensor * q_proj    = nullptr;  // [Kd, H]
    ggml_tensor * k_proj    = nullptr;  // [Kd, H]
    ggml_tensor * v_proj    = nullptr;  // [Kd, H]
    ggml_tensor * q_conv    = nullptr;  // [Kd, 1, W]   pred.kda.{i}.conv.q.weight
    ggml_tensor * k_conv    = nullptr;  // [Kd, 1, W]
    ggml_tensor * v_conv    = nullptr;  // [Kd, 1, W]
    ggml_tensor * f_proj0   = nullptr;  // [K, H]       low-rank gate
    ggml_tensor * f_proj1   = nullptr;  // [Kd, K]
    ggml_tensor * b_proj    = nullptr;  // [heads, H]
    ggml_tensor * g_proj0   = nullptr;  // [K, H]       output gate
    ggml_tensor * g_proj1   = nullptr;  // [Kd, K]
    ggml_tensor * g_proj1_b = nullptr;  // [Kd]
    ggml_tensor * A_log     = nullptr;  // [heads]
    ggml_tensor * dt_bias   = nullptr;  // [Kd]
    ggml_tensor * o_norm_w  = nullptr;  // [K]
    ggml_tensor * o_proj    = nullptr;  // [H, Kd]
};

struct KdaTensors {
    std::vector<KdaLayerTensors> layers;
    ggml_tensor *                final_norm_w = nullptr;  // [H] when final_norm
    ggml_tensor *                final_norm_b = nullptr;
};

transcribe_status load_kda_tensors(ggml_context *     ctx_meta,
                                   const KdaHParams & hp,
                                   int                pred_hidden,
                                   int                n_layers,
                                   KdaTensors &       out);

// fp32 weights made resident once at load (q/k/v projections and the
// three short convs fused, the f/g/b input projections fused), shared by
// every decode. Owned; freed by the destructor.
struct KdaPredictor {
    struct Layer {
        ggml_tensor * norm_w = nullptr;  // [H]
        ggml_tensor * norm_b = nullptr;  // [H]
        ggml_tensor * w_qkv  = nullptr;  // ne [H, 3*Kd]
        ggml_tensor * conv_w = nullptr;  // ne [W, 3*Kd]
        ggml_tensor * w_fgb  = nullptr;  // ne [H, 2*K + heads]
        ggml_tensor * w_f1   = nullptr;  // ne [K, Kd]
        ggml_tensor * dt_b   = nullptr;  // [Kd]
        ggml_tensor * neg_a  = nullptr;  // [Kd] = -exp(A_log[h]) per head, repeated K times
        ggml_tensor * w_g1   = nullptr;  // ne [K, Kd]
        ggml_tensor * g1_b   = nullptr;  // [Kd]
        ggml_tensor * o_nw   = nullptr;  // [K]
        ggml_tensor * w_o    = nullptr;  // ne [Kd, H]
    };

    KdaHParams         hp;
    int                hidden = 0;
    std::vector<Layer> layers;
    ggml_tensor *      final_w = nullptr;
    ggml_tensor *      final_b = nullptr;

    ggml_context *        ctx     = nullptr;
    ggml_backend_t        backend = nullptr;  // alloc-only; never compute'd
    ggml_backend_buffer_t buf     = nullptr;

    // Floats of one layer's state in LstmState::kda: the recurrent memory
    // (heads x [V, K], fla's state_v_first layout) then the conv history
    // ((W-1) previous inputs per q/k/v channel, oldest first).
    size_t recurrent_floats() const;
    size_t conv_floats() const;

    size_t layer_state_floats() const { return recurrent_floats() + conv_floats(); }

    size_t state_floats() const { return layer_state_floats() * layers.size(); }

    KdaPredictor() = default;
    ~KdaPredictor();
    KdaPredictor(const KdaPredictor &)             = delete;
    KdaPredictor & operator=(const KdaPredictor &) = delete;
};

// Dequantizes `t` and makes it resident. Returns TRANSCRIBE_OK or a load error.
transcribe_status build_kda_predictor(const KdaHParams & hp, const KdaTensors & t, int pred_hidden, KdaPredictor & out);

// Per-decode step graph, built on the decoder's CPU backend (borrowed).
struct KdaGraph {
    ggml_context *             ctx   = nullptr;
    ggml_backend_buffer_t      buf   = nullptr;
    ggml_cgraph *              graph = nullptr;
    ggml_tensor *              x     = nullptr;  // [H] input embedding
    std::vector<ggml_tensor *> s_in;             // per layer [K, K, heads] recurrent state
    std::vector<ggml_tensor *> conv_in;          // per layer [W-1, 3*Kd] conv history
    std::vector<ggml_tensor *> gdn;              // per layer delta-net result: o then the new state
    std::vector<ggml_tensor *> conv_out;         // per layer [W-1, 3*Kd]
    std::vector<ggml_tensor *> h;                // per layer [H] residual stream (last: after final norm)
    bool                       ready = false;

    KdaGraph() = default;
    ~KdaGraph();
    KdaGraph(const KdaGraph &)             = delete;
    KdaGraph & operator=(const KdaGraph &) = delete;
};

bool build_kda_graph(KdaGraph & g, const KdaPredictor & p, ggml_backend_t backend);

// One predictor step: `x` is the embedding of the previous token (zeros at
// the start), `prev` the committed state (empty kda = start state). Writes
// next.kda and next.h (next.h must hold one [H] vector per layer) and
// returns next.h.back().data(), or nullptr if the compute fails.
const float * kda_predictor_step(const KdaPredictor & p,
                                 KdaGraph &           g,
                                 ggml_backend_t       backend,
                                 const float *        x,
                                 const LstmState &    prev,
                                 LstmState &          next);

}  // namespace transcribe::parakeet
