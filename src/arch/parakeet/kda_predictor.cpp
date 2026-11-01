// arch/parakeet/kda_predictor.cpp - Kimi Delta Attention prediction network.
//
// See kda_predictor.h for the math and the state contract.

#include "kda_predictor.h"

#include "decoder.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "gguf.h"
#include "transcribe-backend.h"
#include "transcribe-log.h"
#include "transcribe-meta.h"
#include "transcribe-weights-util.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

namespace transcribe::parakeet {

namespace {

constexpr const char * kTag = "parakeet";

using transcribe::weights::lname;

#define KDA_GET_F32(slot, name, ...)                                                                      \
    do {                                                                                                  \
        ggml_tensor * _t =                                                                                \
            transcribe::weights::find_tensor(ctx_meta, (name), { GGML_TYPE_F32 }, { __VA_ARGS__ }, kTag); \
        if (_t == nullptr)                                                                                \
            return TRANSCRIBE_ERR_GGUF;                                                                   \
        (slot) = _t;                                                                                      \
    } while (0)

#define KDA_GET_CONV(slot, name, ...)                                                                          \
    do {                                                                                                       \
        ggml_tensor * _t = transcribe::weights::find_tensor(ctx_meta, (name), { TRANSCRIBE_QUANT_CONV_TYPES }, \
                                                            { __VA_ARGS__ }, kTag);                            \
        if (_t == nullptr)                                                                                     \
            return TRANSCRIBE_ERR_GGUF;                                                                        \
        (slot) = _t;                                                                                           \
    } while (0)

#define KDA_GET_LIN(slot, name, ...)                                                                             \
    do {                                                                                                         \
        ggml_tensor * _t = transcribe::weights::find_tensor(ctx_meta, (name), { TRANSCRIBE_QUANT_LINEAR_TYPES }, \
                                                            { __VA_ARGS__ }, kTag);                              \
        if (_t == nullptr)                                                                                       \
            return TRANSCRIBE_ERR_GGUF;                                                                          \
        (slot) = _t;                                                                                             \
    } while (0)

// All elements of `t` as fp32 (dequantized), appended to `out`.
bool append_f32(const ggml_tensor * t, std::vector<float> & out) {
    if (t == nullptr) {
        return false;
    }
    const int64_t n      = ggml_nelements(t);
    const size_t  nbytes = ggml_nbytes(t);
    const size_t  base   = out.size();
    out.resize(base + static_cast<size_t>(n));
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data() + base, 0, nbytes);
        return true;
    }
    const auto * tt = ggml_get_type_traits(t->type);
    if (tt == nullptr || tt->to_float == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet kda: tensor \"%s\" type %s has no to_float", t->name,
                ggml_type_name(t->type));
        return false;
    }
    std::vector<uint8_t> raw(nbytes);
    ggml_backend_tensor_get(t, raw.data(), 0, nbytes);
    tt->to_float(raw.data(), out.data() + base, n);
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// GGUF metadata + tensors
// ---------------------------------------------------------------------------

transcribe_status read_kda_hparams(const gguf_context * gguf, KdaHParams & hp) {
    hp = KdaHParams{};
    std::string kind;
    if (auto st = read_optional_string_kv(gguf, "stt.parakeet.predictor.kind", kTag, "lstm", kind);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (kind == "lstm") {
        return TRANSCRIBE_OK;
    }
    if (kind != "kda") {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet: unsupported predictor kind \"%s\" (allowed: lstm, kda)",
                kind.c_str());
        return TRANSCRIBE_ERR_GGUF;
    }
    hp.enabled = true;
    if (auto st = read_required_u32_kv(gguf, "stt.parakeet.predictor.kda.num_heads", kTag, hp.num_heads);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = read_required_u32_kv(gguf, "stt.parakeet.predictor.kda.head_dim", kTag, hp.head_dim);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = read_required_u32_kv(gguf, "stt.parakeet.predictor.kda.conv_size", kTag, hp.conv_size);
        st != TRANSCRIBE_OK) {
        return st;
    }
    if (auto st = read_optional_bool_kv(gguf, "stt.parakeet.predictor.kda.final_norm", kTag, true, hp.final_norm);
        st != TRANSCRIBE_OK) {
        return st;
    }
    for (auto [key, slot] : {
             std::pair<const char *, float *>{ "stt.parakeet.predictor.kda.norm_eps",   &hp.norm_eps   },
             std::pair<const char *, float *>{ "stt.parakeet.predictor.kda.o_norm_eps", &hp.o_norm_eps }
    }) {
        if (read_float32_kv(gguf, key, *slot) == KvResult::BadType) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet: KV \"%s\" has the wrong type", key);
            return TRANSCRIBE_ERR_GGUF;
        }
    }
    if (hp.num_heads <= 0 || hp.head_dim <= 0 || hp.conv_size < 2 || !(hp.norm_eps > 0.0f) || !(hp.o_norm_eps > 0.0f)) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR,
                "parakeet: invalid kda hparams (num_heads=%d head_dim=%d conv_size=%d norm_eps=%g o_norm_eps=%g)",
                hp.num_heads, hp.head_dim, hp.conv_size, static_cast<double>(hp.norm_eps),
                static_cast<double>(hp.o_norm_eps));
        return TRANSCRIBE_ERR_GGUF;
    }
    return TRANSCRIBE_OK;
}

transcribe_status load_kda_tensors(ggml_context *     ctx_meta,
                                   const KdaHParams & hp,
                                   int                pred_hidden,
                                   int                n_layers,
                                   KdaTensors &       out) {
    const int64_t H  = pred_hidden;
    const int64_t K  = hp.head_dim;
    const int64_t NH = hp.num_heads;
    const int64_t Kd = hp.key_dim();
    const int64_t W  = hp.conv_size;

    out = KdaTensors{};
    out.layers.assign(static_cast<size_t>(n_layers), KdaLayerTensors{});
    for (int i = 0; i < n_layers; ++i) {
        KdaLayerTensors & l = out.layers[static_cast<size_t>(i)];
        KDA_GET_F32(l.norm_w, lname("pred.kda.%d.norm_in.weight", i), H);
        KDA_GET_F32(l.norm_b, lname("pred.kda.%d.norm_in.bias", i), H);
        KDA_GET_LIN(l.q_proj, lname("pred.kda.%d.q_proj.weight", i), H, Kd);
        KDA_GET_LIN(l.k_proj, lname("pred.kda.%d.k_proj.weight", i), H, Kd);
        KDA_GET_LIN(l.v_proj, lname("pred.kda.%d.v_proj.weight", i), H, Kd);
        KDA_GET_CONV(l.q_conv, lname("pred.kda.%d.conv.q.weight", i), W, 1, Kd);
        KDA_GET_CONV(l.k_conv, lname("pred.kda.%d.conv.k.weight", i), W, 1, Kd);
        KDA_GET_CONV(l.v_conv, lname("pred.kda.%d.conv.v.weight", i), W, 1, Kd);
        KDA_GET_LIN(l.f_proj0, lname("pred.kda.%d.f_proj.0.weight", i), H, K);
        KDA_GET_LIN(l.f_proj1, lname("pred.kda.%d.f_proj.1.weight", i), K, Kd);
        KDA_GET_LIN(l.b_proj, lname("pred.kda.%d.b_proj.weight", i), H, NH);
        KDA_GET_LIN(l.g_proj0, lname("pred.kda.%d.g_proj.0.weight", i), H, K);
        KDA_GET_LIN(l.g_proj1, lname("pred.kda.%d.g_proj.1.weight", i), K, Kd);
        KDA_GET_F32(l.g_proj1_b, lname("pred.kda.%d.g_proj.1.bias", i), Kd);
        KDA_GET_F32(l.A_log, lname("pred.kda.%d.A_log", i), NH);
        KDA_GET_F32(l.dt_bias, lname("pred.kda.%d.dt_bias", i), Kd);
        KDA_GET_F32(l.o_norm_w, lname("pred.kda.%d.o_norm.weight", i), K);
        KDA_GET_LIN(l.o_proj, lname("pred.kda.%d.o_proj.weight", i), Kd, H);
    }
    if (hp.final_norm) {
        KDA_GET_F32(out.final_norm_w, "pred.kda.final_norm.weight", H);
        KDA_GET_F32(out.final_norm_b, "pred.kda.final_norm.bias", H);
    }
    return TRANSCRIBE_OK;
}

#undef KDA_GET_F32
#undef KDA_GET_CONV
#undef KDA_GET_LIN

// ---------------------------------------------------------------------------
// Resident fp32 weights
// ---------------------------------------------------------------------------

size_t KdaPredictor::recurrent_floats() const {
    return static_cast<size_t>(hp.num_heads) * static_cast<size_t>(hp.head_dim) * static_cast<size_t>(hp.head_dim);
}

size_t KdaPredictor::conv_floats() const {
    return static_cast<size_t>(hp.conv_size - 1) * 3 * static_cast<size_t>(hp.key_dim());
}

KdaPredictor::~KdaPredictor() {
    if (buf != nullptr) {
        safe_buffer_free(buf);
    }
    if (ctx != nullptr) {
        ggml_free(ctx);
    }
    if (backend != nullptr) {
        safe_backend_free(backend);
    }
}

transcribe_status build_kda_predictor(const KdaHParams & hp, const KdaTensors & t, int pred_hidden, KdaPredictor & p) {
    const int H  = pred_hidden;
    const int K  = hp.head_dim;
    const int NH = hp.num_heads;
    const int Kd = hp.key_dim();
    const int W  = hp.conv_size;
    const int L  = static_cast<int>(t.layers.size());
    if (!hp.enabled || H <= 0 || L <= 0) {
        return TRANSCRIBE_ERR_GGUF;
    }

    p.hp     = hp;
    p.hidden = H;
    p.layers.assign(static_cast<size_t>(L), KdaPredictor::Layer{});

    p.backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (p.backend == nullptr) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet kda: ggml CPU backend init failed");
        return TRANSCRIBE_ERR_BACKEND;
    }
    ggml_init_params ip{};
    ip.mem_size   = ggml_tensor_overhead() * static_cast<size_t>(12 * L + 2);
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    p.ctx         = ggml_init(ip);
    if (p.ctx == nullptr) {
        return TRANSCRIBE_ERR_BACKEND;
    }

    for (auto & ly : p.layers) {
        ly.norm_w = ggml_new_tensor_1d(p.ctx, GGML_TYPE_F32, H);
        ly.norm_b = ggml_new_tensor_1d(p.ctx, GGML_TYPE_F32, H);
        ly.w_qkv  = ggml_new_tensor_2d(p.ctx, GGML_TYPE_F32, H, 3 * Kd);
        ly.conv_w = ggml_new_tensor_2d(p.ctx, GGML_TYPE_F32, W, 3 * Kd);
        ly.w_fgb  = ggml_new_tensor_2d(p.ctx, GGML_TYPE_F32, H, 2 * K + NH);
        ly.w_f1   = ggml_new_tensor_2d(p.ctx, GGML_TYPE_F32, K, Kd);
        ly.dt_b   = ggml_new_tensor_1d(p.ctx, GGML_TYPE_F32, Kd);
        ly.neg_a  = ggml_new_tensor_1d(p.ctx, GGML_TYPE_F32, Kd);
        ly.w_g1   = ggml_new_tensor_2d(p.ctx, GGML_TYPE_F32, K, Kd);
        ly.g1_b   = ggml_new_tensor_1d(p.ctx, GGML_TYPE_F32, Kd);
        ly.o_nw   = ggml_new_tensor_1d(p.ctx, GGML_TYPE_F32, K);
        ly.w_o    = ggml_new_tensor_2d(p.ctx, GGML_TYPE_F32, Kd, H);
    }
    if (hp.final_norm) {
        p.final_w = ggml_new_tensor_1d(p.ctx, GGML_TYPE_F32, H);
        p.final_b = ggml_new_tensor_1d(p.ctx, GGML_TYPE_F32, H);
    }
    p.buf = ggml_backend_alloc_ctx_tensors(p.ctx, p.backend);
    if (p.buf == nullptr) {
        return TRANSCRIBE_ERR_BACKEND;
    }

    // Host bytes are PyTorch row-major [out, in], i.e. ggml ne [in, out];
    // fusing along `out` is plain concatenation.
    std::vector<float> host;
    auto               upload = [&](ggml_tensor * dst, std::initializer_list<const ggml_tensor *> srcs) -> bool {
        host.clear();
        for (const ggml_tensor * s : srcs) {
            if (!append_f32(s, host)) {
                return false;
            }
        }
        if (host.size() != static_cast<size_t>(ggml_nelements(dst))) {
            log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet kda: fused weight size %zu != %lld", host.size(),
                    static_cast<long long>(ggml_nelements(dst)));
            return false;
        }
        ggml_backend_tensor_set(dst, host.data(), 0, host.size() * sizeof(float));
        return true;
    };

    for (int i = 0; i < L; ++i) {
        const KdaLayerTensors & s  = t.layers[static_cast<size_t>(i)];
        KdaPredictor::Layer &   ly = p.layers[static_cast<size_t>(i)];
        const bool              ok =
            upload(ly.norm_w, { s.norm_w }) && upload(ly.norm_b, { s.norm_b }) &&
            upload(ly.w_qkv, { s.q_proj, s.k_proj, s.v_proj }) && upload(ly.conv_w, { s.q_conv, s.k_conv, s.v_conv }) &&
            upload(ly.w_fgb, { s.f_proj0, s.g_proj0, s.b_proj }) && upload(ly.w_f1, { s.f_proj1 }) &&
            upload(ly.dt_b, { s.dt_bias }) && upload(ly.w_g1, { s.g_proj1 }) && upload(ly.g1_b, { s.g_proj1_b }) &&
            upload(ly.o_nw, { s.o_norm_w }) && upload(ly.w_o, { s.o_proj });
        if (!ok) {
            return TRANSCRIBE_ERR_GGUF;
        }
        // Gate decay rate per (head, key dim): -exp(A_log[h]) for every k.
        host.clear();
        if (!append_f32(s.A_log, host) || static_cast<int>(host.size()) != NH) {
            return TRANSCRIBE_ERR_GGUF;
        }
        std::vector<float> neg_a(static_cast<size_t>(Kd));
        for (int h = 0; h < NH; ++h) {
            const float a = -std::exp(host[static_cast<size_t>(h)]);
            for (int k = 0; k < K; ++k) {
                neg_a[static_cast<size_t>(h * K + k)] = a;
            }
        }
        ggml_backend_tensor_set(ly.neg_a, neg_a.data(), 0, neg_a.size() * sizeof(float));
    }
    if (hp.final_norm) {
        if (!upload(p.final_w, { t.final_norm_w }) || !upload(p.final_b, { t.final_norm_b })) {
            return TRANSCRIBE_ERR_GGUF;
        }
    }
    return TRANSCRIBE_OK;
}

// ---------------------------------------------------------------------------
// Per-decode step graph
// ---------------------------------------------------------------------------

KdaGraph::~KdaGraph() {
    if (buf != nullptr) {
        safe_buffer_free(buf);
    }
    if (ctx != nullptr) {
        ggml_free(ctx);
    }
}

bool build_kda_graph(KdaGraph & g, const KdaPredictor & p, ggml_backend_t backend) {
    const int    H  = p.hidden;
    const int    K  = p.hp.head_dim;
    const int    NH = p.hp.num_heads;
    const int    Kd = p.hp.key_dim();
    const int    C  = 3 * Kd;
    const int    W  = p.hp.conv_size;
    const int    L  = static_cast<int>(p.layers.size());
    const size_t fs = sizeof(float);
    if (backend == nullptr || L == 0 || H == 0) {
        return false;
    }

    ggml_init_params ip{};
    ip.mem_size   = ggml_tensor_overhead() * static_cast<size_t>(64 * L + 16) + ggml_graph_overhead();
    ip.mem_buffer = nullptr;
    ip.no_alloc   = true;
    g.ctx         = ggml_init(ip);
    if (g.ctx == nullptr) {
        return false;
    }
    ggml_context * ctx = g.ctx;

    g.x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, H);
    ggml_set_input(g.x);
    g.s_in.assign(static_cast<size_t>(L), nullptr);
    g.conv_in.assign(static_cast<size_t>(L), nullptr);
    g.gdn.assign(static_cast<size_t>(L), nullptr);
    g.conv_out.assign(static_cast<size_t>(L), nullptr);
    g.h.assign(static_cast<size_t>(L), nullptr);
    for (int l = 0; l < L; ++l) {
        g.s_in[l] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, K, K, NH, 1);
        ggml_set_input(g.s_in[l]);
        g.conv_in[l] = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, W - 1, C);
        ggml_set_input(g.conv_in[l]);
    }

    // fla's l2norm, x / sqrt(sum(x^2) + 1e-6), is rms_norm with eps 1e-6/K
    // scaled by 1/sqrt(K). (ggml_l2_norm clamps sqrt(sum) at eps instead.)
    const float l2_eps     = 1e-6f / static_cast<float>(K);
    const float inv_sqrt_k = 1.0f / std::sqrt(static_cast<float>(K));

    ggml_tensor * x = g.x;
    for (int l = 0; l < L; ++l) {
        const KdaPredictor::Layer & ly = p.layers[static_cast<size_t>(l)];

        ggml_tensor * xn = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, p.hp.norm_eps), ly.norm_w), ly.norm_b);

        // q/k/v projections, then the depthwise causal conv over the W-1
        // previous inputs and this one: y = SiLU(sum_j w[j] * x[t-W+1+j]).
        ggml_tensor * qkv  = ggml_mul_mat(ctx, ly.w_qkv, xn);                                     // [C]
        ggml_tensor * cx   = ggml_concat(ctx, g.conv_in[l], ggml_reshape_2d(ctx, qkv, 1, C), 0);  // [W, C]
        g.conv_out[l]      = ggml_cont(ctx, ggml_view_2d(ctx, cx, W - 1, C, cx->nb[1], fs));
        ggml_tensor * conv = ggml_silu(ctx, ggml_sum_rows(ctx, ggml_mul(ctx, cx, ly.conv_w)));    // [1, C]
        conv               = ggml_reshape_1d(ctx, conv, C);

        auto heads = [&](size_t offset_floats) {
            return ggml_view_4d(ctx, conv, K, NH, 1, 1, K * fs, Kd * fs, Kd * fs, offset_floats * fs);
        };
        ggml_tensor * q = ggml_scale(ctx, ggml_rms_norm(ctx, heads(0), l2_eps), inv_sqrt_k);
        ggml_tensor * k = ggml_scale(ctx, ggml_rms_norm(ctx, heads(Kd), l2_eps), inv_sqrt_k);
        ggml_tensor * v = heads(2 * static_cast<size_t>(Kd));

        // Low-rank gate input, output gate input and beta share one matmul.
        ggml_tensor * fgb  = ggml_mul_mat(ctx, ly.w_fgb, xn);  // [2K + NH]
        ggml_tensor * f_lo = ggml_view_1d(ctx, fgb, K, 0);
        ggml_tensor * g_lo = ggml_view_1d(ctx, fgb, K, K * fs);
        ggml_tensor * b_lo = ggml_view_1d(ctx, fgb, NH, 2 * K * fs);

        ggml_tensor * gate =
            ggml_mul(ctx, ggml_softplus(ctx, ggml_add(ctx, ggml_mul_mat(ctx, ly.w_f1, f_lo), ly.dt_b)), ly.neg_a);
        gate              = ggml_reshape_4d(ctx, gate, K, NH, 1, 1);
        ggml_tensor * bet = ggml_reshape_4d(ctx, ggml_sigmoid(ctx, b_lo), 1, NH, 1, 1);

        // Result: o [K*NH] for the token, then the new state [K, K, NH].
        ggml_tensor * r = ggml_gated_delta_net(ctx, q, k, v, gate, bet, g.s_in[l], 1);
        g.gdn[l]        = r;

        ggml_tensor * o  = ggml_view_2d(ctx, r, K, NH, K * fs, 0);
        o                = ggml_mul(ctx, ggml_rms_norm(ctx, o, p.hp.o_norm_eps), ly.o_nw);
        ggml_tensor * og = ggml_sigmoid(ctx, ggml_add(ctx, ggml_mul_mat(ctx, ly.w_g1, g_lo), ly.g1_b));
        o                = ggml_mul(ctx, o, ggml_reshape_2d(ctx, og, K, NH));
        o                = ggml_mul_mat(ctx, ly.w_o, ggml_reshape_1d(ctx, o, Kd));  // [H]
        x                = ggml_add(ctx, x, o);

        if (l == L - 1 && p.final_w != nullptr) {
            x = ggml_add(ctx, ggml_mul(ctx, ggml_norm(ctx, x, p.hp.norm_eps), p.final_w), p.final_b);
        }
        g.h[l] = x;
        ggml_set_output(g.h[l]);
        ggml_set_output(g.gdn[l]);
        ggml_set_output(g.conv_out[l]);
    }

    g.buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (g.buf == nullptr) {
        return false;
    }
    g.graph = ggml_new_graph(ctx);
    for (int l = 0; l < L; ++l) {
        ggml_build_forward_expand(g.graph, g.h[l]);
        ggml_build_forward_expand(g.graph, g.gdn[l]);
        ggml_build_forward_expand(g.graph, g.conv_out[l]);
    }
    g.ready = true;
    return true;
}

const float * kda_predictor_step(const KdaPredictor & p,
                                 KdaGraph &           g,
                                 ggml_backend_t       backend,
                                 const float *        x,
                                 const LstmState &    prev,
                                 LstmState &          next) {
    const size_t L     = p.layers.size();
    const size_t H     = static_cast<size_t>(p.hidden);
    const size_t rec   = p.recurrent_floats();
    const size_t conv  = p.conv_floats();
    const size_t layer = rec + conv;
    const size_t fs    = sizeof(float);
    if (!g.ready || next.h.size() != L || (!prev.kda.empty() && prev.kda.size() != layer * L)) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet kda: predictor state has the wrong shape");
        return nullptr;
    }

    ggml_backend_tensor_set(g.x, x, 0, H * fs);
    for (size_t l = 0; l < L; ++l) {
        if (prev.kda.empty()) {  // start of sequence: zero memory, zero conv history
            ggml_backend_tensor_memset(g.s_in[l], 0, 0, rec * fs);
            ggml_backend_tensor_memset(g.conv_in[l], 0, 0, conv * fs);
        } else {
            ggml_backend_tensor_set(g.s_in[l], prev.kda.data() + l * layer, 0, rec * fs);
            ggml_backend_tensor_set(g.conv_in[l], prev.kda.data() + l * layer + rec, 0, conv * fs);
        }
    }

    if (const ggml_status gs = ggml_backend_graph_compute(backend, g.graph); gs != GGML_STATUS_SUCCESS) {
        log_msg(TRANSCRIBE_LOG_LEVEL_ERROR, "parakeet kda: predictor compute failed (%d)", static_cast<int>(gs));
        return nullptr;
    }

    next.kda.resize(layer * L);
    const size_t o_floats = static_cast<size_t>(p.hp.key_dim());  // the token's o precedes the state
    for (size_t l = 0; l < L; ++l) {
        if (next.h[l].size() != H) {
            next.h[l].resize(H);
        }
        ggml_backend_tensor_get(g.gdn[l], next.kda.data() + l * layer, o_floats * fs, rec * fs);
        ggml_backend_tensor_get(g.conv_out[l], next.kda.data() + l * layer + rec, 0, conv * fs);
        ggml_backend_tensor_get(g.h[l], next.h[l].data(), 0, H * fs);
    }
    return next.h.back().data();
}

}  // namespace transcribe::parakeet
