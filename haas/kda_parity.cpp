// haas/kda_parity.cpp - dump what the C++ KDA predictor computes, for
// haas/check_parity.py to compare against the PyTorch reference.
//
//   haas-kda-parity steps <model.gguf> <tokens.txt> <out.f32>
//       Predictor outputs for a token sequence (-1 = start embedding),
//       one [pred_hidden] fp32 row per step, carrying the state across
//       steps like the decoders do (kda_predictor_step).
//
//   haas-kda-parity tdt <model.gguf> <enc.f32> <T> <out.txt>
//       decode_tdt_greedy over a given encoder output [T, d_enc]; writes
//       "id step_at_emit duration" per emitted token. Goes through
//       decoder.cpp's predictor dispatch and state commit logic.

#include "arch/parakeet/decoder.h"
#include "arch/parakeet/parakeet.h"
#include "transcribe.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

namespace tp = transcribe::parakeet;

namespace {

const tp::ParakeetModel * load(const char * path, transcribe_model ** out) {
    transcribe_model_load_params mp;
    transcribe_model_load_params_init(&mp);
    if (transcribe_model_load_file(path, &mp, out) != TRANSCRIBE_OK || *out == nullptr) {
        std::fprintf(stderr, "load failed: %s\n", path);
        return nullptr;
    }
    const auto * pm = static_cast<const tp::ParakeetModel *>(*out);
    if (pm->host_decoder.predictor.kda == nullptr) {
        std::fprintf(stderr, "%s has no KDA predictor\n", path);
        return nullptr;
    }
    return pm;
}

int run_steps(const char * model_path, const char * tokens_path, const char * out_path) {
    transcribe_model *        model = nullptr;
    const tp::ParakeetModel * pm    = load(model_path, &model);
    if (pm == nullptr) {
        return 1;
    }
    std::vector<int> tokens;
    {
        std::ifstream in(tokens_path);
        int           t;
        while (in >> t) {
            tokens.push_back(t);
        }
    }
    const auto &       w       = pm->host_decoder;
    const auto &       kda     = *w.predictor.kda;
    const int          H       = w.predictor.pred_hidden;
    const int          L       = static_cast<int>(kda.layers.size());
    ggml_backend_t     backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    std::vector<float> outs;
    {
        tp::KdaGraph g;
        if (!tp::build_kda_graph(g, kda, backend)) {
            std::fprintf(stderr, "graph build failed\n");
            return 1;
        }
        tp::LstmState      state, next;
        std::vector<float> x(static_cast<size_t>(H));
        state.reset(L, H);
        next.reset(L, H);
        for (int t : tokens) {
            if (t < 0) {
                std::fill(x.begin(), x.end(), 0.0f);
            } else {
                std::memcpy(x.data(), w.predictor.embed_w.data() + static_cast<size_t>(t) * H, H * sizeof(float));
            }
            const float * o = tp::kda_predictor_step(kda, g, backend, x.data(), state, next);
            if (o == nullptr) {
                std::fprintf(stderr, "step failed\n");
                return 1;
            }
            outs.insert(outs.end(), o, o + H);
            std::swap(state, next);
        }
    }
    ggml_backend_free(backend);
    std::FILE * f = std::fopen(out_path, "wb");
    std::fwrite(outs.data(), sizeof(float), outs.size(), f);
    std::fclose(f);
    std::printf("%zu steps x %d\n", tokens.size(), w.predictor.pred_hidden);
    transcribe_model_free(model);
    return 0;
}

int run_tdt(const char * model_path, const char * enc_path, int T, const char * out_path) {
    transcribe_model *        model = nullptr;
    const tp::ParakeetModel * pm    = load(model_path, &model);
    if (pm == nullptr) {
        return 1;
    }
    const int          d_enc = pm->host_decoder.joint.d_enc;
    std::vector<float> enc(static_cast<size_t>(T) * static_cast<size_t>(d_enc));
    std::FILE *        f = std::fopen(enc_path, "rb");
    if (f == nullptr || std::fread(enc.data(), sizeof(float), enc.size(), f) != enc.size()) {
        std::fprintf(stderr, "short read: %s\n", enc_path);
        return 1;
    }
    std::fclose(f);
    std::vector<tp::TdtToken> toks;
    if (tp::decode_tdt_greedy(pm->host_decoder, enc.data(), T, d_enc, 4, toks) != TRANSCRIBE_OK) {
        std::fprintf(stderr, "decode failed\n");
        return 1;
    }
    std::FILE * o = std::fopen(out_path, "w");
    for (const auto & t : toks) {
        std::fprintf(o, "%d %d %d\n", t.id, t.step_at_emit, t.duration_frames);
    }
    std::fclose(o);
    std::printf("%zu tokens\n", toks.size());
    transcribe_model_free(model);
    return 0;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc == 5 && std::strcmp(argv[1], "steps") == 0) {
        return run_steps(argv[2], argv[3], argv[4]);
    }
    if (argc == 6 && std::strcmp(argv[1], "tdt") == 0) {
        return run_tdt(argv[2], argv[3], std::atoi(argv[4]), argv[5]);
    }
    std::fprintf(stderr,
                 "usage: %s steps <model.gguf> <tokens.txt> <out.f32>\n"
                 "       %s tdt <model.gguf> <enc.f32> <T> <out.txt>\n",
                 argv[0], argv[0]);
    return 2;
}
