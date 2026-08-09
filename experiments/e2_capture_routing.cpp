// E2 (capture half) — record the REAL expert routing of a MoE model, per token, per layer.
//
// This is the decisive measurement of the Pollard Weights idea (README §6.1). The design rests on
// one unproven claim: that real traffic reuses a SMALL number of routing patterns out of the
// C(256,8) ~= 4e14 possible per layer. High reuse -> the online index converges and stays tiny.
// Low reuse -> it grows unbounded and the whole thing is dead. Nothing else matters until this
// number exists.
//
// We do NOT predict anything here. The router is deterministic: experts = top-k(W_router . h).
// llama.cpp already names that tensor "ffn_moe_topk-<layer>", so we attach a backend eval
// callback and read the selection straight out of the compute graph. No patch to llama.cpp, no
// re-implementation of the router that could silently disagree with the real one.
//
// Weights stay flash-resident by default (--ngl 0, mmap on): that is the regime the idea targets,
// so measuring in any other regime would be measuring a different machine.
//
// Emits JSONL, one line per (prompt, position, layer):
//   {"prompt":0,"tag":"code","pos":12,"layer":7,"experts":[3,17,...]}
//
// Build: see e2_build.sh   Analyse: see e2_analyse_routing.py

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cstdlib>
#include <cstring>
#include <cstdio>

// --rpc host:port,host:port — shard the capture model across machines the way
// llama.cpp itself does (the K3-on-Sparks pattern). Uses the backend registry,
// so this compiles against any build; at runtime it needs GGML_RPC=ON.
static void add_rpc_devices(const char * servers_csv) {
    ggml_backend_reg_t rpc_reg = ggml_backend_reg_by_name("RPC");
    if (!rpc_reg) {
        fprintf(stderr, "E2: this llama.cpp build has no RPC backend — rebuild with -DGGML_RPC=ON\n");
        exit(1);
    }
    typedef ggml_backend_dev_t (*add_fn_t)(const char * endpoint);
    add_fn_t add_fn = (add_fn_t) ggml_backend_reg_get_proc_address(rpc_reg, "ggml_backend_rpc_add_device");
    if (!add_fn) {
        fprintf(stderr, "E2: RPC backend lacks ggml_backend_rpc_add_device\n");
        exit(1);
    }
    char buf[1024];
    snprintf(buf, sizeof(buf), "%s", servers_csv);
    for (char * tok = strtok(buf, ","); tok; tok = strtok(nullptr, ",")) {
        ggml_backend_dev_t dev = add_fn(tok);
        if (!dev) { fprintf(stderr, "E2: failed to add RPC device %s\n", tok); exit(1); }
        ggml_backend_device_register(dev);
        fprintf(stderr, "E2: RPC device added: %s\n", tok);
    }
}

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <chrono>
#include <map>

struct capture_state {
    FILE * out          = nullptr;
    int    prompt_id    = 0;
    const char * tag    = "";
    long   lines        = 0;
    std::vector<int32_t> buf;
    // Absolute token position emitted SO FAR, per layer. llama.cpp splits a long prompt into
    // several ubatches, firing this callback once per ubatch per layer; a single per-prompt
    // "pos_base" therefore restarts at 0 on every chunk and silently aliases token 512 onto
    // token 0. Counting per layer is correct however the batch is chunked.
    std::map<int, int> layer_pos;
};

// Backend scheduler callback. Called for every node in the graph: first with ask=true to ask
// whether we want the data, then (if we said yes) with ask=false once the data is valid.
static bool cb_eval(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = static_cast<capture_state *>(user_data);
    static const char * PREFIX = "ffn_moe_topk-";
    const size_t plen = strlen(PREFIX);

    if (ask) {
        // Returning false here just means "don't bother copying this one back" — it never
        // suppresses the node's execution, so this cannot change what the model computes.
        return strncmp(t->name, PREFIX, plen) == 0;
    }

    if (strncmp(t->name, PREFIX, plen) != 0) {
        return true;
    }
    const int layer = atoi(t->name + plen);

    // ggml_argsort_top_k emits expert INDICES, shape [n_expert_used, n_tokens].
    if (t->type != GGML_TYPE_I32) {
        fprintf(stderr, "E2: expected I32 for %s, got %s — aborting rather than writing junk\n",
                t->name, ggml_type_name(t->type));
        exit(1);
    }

    const int64_t n_used   = t->ne[0];
    const int64_t n_tokens = t->ne[1];

    st->buf.resize((size_t) n_used * n_tokens);
    ggml_backend_tensor_get(t, st->buf.data(), 0, ggml_nbytes(t));

    const int base = st->layer_pos[layer];
    for (int64_t j = 0; j < n_tokens; ++j) {
        fprintf(st->out, "{\"prompt\":%d,\"tag\":\"%s\",\"pos\":%d,\"layer\":%d,\"experts\":[",
                st->prompt_id, st->tag, base + (int) j, layer);
        for (int64_t i = 0; i < n_used; ++i) {
            fprintf(st->out, "%s%d", i ? "," : "", st->buf[(size_t) j * n_used + i]);
        }
        fprintf(st->out, "]}\n");
        st->lines++;
    }
    st->layer_pos[layer] = base + (int) n_tokens;
    return true;
}

// Prompts are grouped by domain on purpose: README §6.4 asks whether "converged" is one global
// state or per-domain. Answering that needs traffic that actually differs in domain.
struct prompt_spec { const char * tag; const char * text; };

static const std::vector<prompt_spec> DEFAULT_PROMPTS = {
    {"code", "def compute_watering_time(soil_moisture, sun_hours, drip_rate):\n"
             "    round_trip = 2 * (sun_hours + drip_rate) / 10000.0\n"
             "    return soil_moisture * (1.0 + round_trip)\n\n"
             "class OrderRouter:\n"
             "    def __init__(self, venues):\n"
             "        self.venues = sorted(venues, key=lambda v: v.taker_fee)\n"
             "    def best_venue(self, size):\n"
             "        for v in self.venues:\n"
             "            if v.depth_at(size) > size:\n"
             "                return v\n"
             "        raise NoLiquidity(size)\n"},
    {"code", "impl Router {\n"
             "    pub fn select(&self, hidden: &[f32]) -> Vec<usize> {\n"
             "        let mut scored: Vec<(usize, f32)> = self.gate\n"
             "            .iter().enumerate()\n"
             "            .map(|(i, w)| (i, dot(w, hidden)))\n"
             "            .collect();\n"
             "        scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());\n"
             "        scored.into_iter().take(self.top_k).map(|(i, _)| i).collect()\n"
             "    }\n"
             "}\n"},
    {"prose", "The harbour at dawn was the colour of cold tea, and the fishermen worked without "
              "speaking, because everything worth saying had been said the night before. Nets came "
              "up heavy or came up light, and either way the boats turned home before the wind "
              "changed. A boy counted gulls on the breakwater and lost his place twice.\n"},
    {"prose", "Memory is not a recording. Each time a thing is recalled it is rebuilt from parts, "
              "and the rebuilding leaves fingerprints, so that a story told often enough becomes "
              "smooth where it was once rough, and confident where it was once uncertain. What "
              "feels like clarity is often only repetition.\n"},
    {"math",  "Let p be an odd prime and let g be a primitive root modulo p. For the discrete "
              "logarithm x with g^x = h (mod p), Pollard's rho method finds a collision in "
              "expected O(sqrt(p)) group operations by iterating a pseudorandom walk and storing "
              "only distinguished points, those whose representation ends in k zero bits.\n"},
};

int main(int argc, char ** argv) {
    const char * model_path = nullptr;
    const char * out_path   = "routing.jsonl";
    const char * file_path  = nullptr;   // single-domain corpus: the workload that actually matters
    const char * file_tag   = "file";
    int  ngl   = 0;      // flash-resident by default — the regime this project is about
    int  n_ctx = 4096;
    int  n_gen = 0;       // decode-phase capture: greedy-generate N tokens after the corpus
    float temp = 0.0f;    // 0 = greedy; >0 = temperature sampling (kills repetition inflation)
    uint32_t seed = 42;
    const char * rpc_csv = nullptr;
    bool list_only = false;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-m") && i + 1 < argc)          model_path = argv[++i];
        else if (!strcmp(argv[i], "-o") && i + 1 < argc)     out_path   = argv[++i];
        else if (!strcmp(argv[i], "-f") && i + 1 < argc)     file_path  = argv[++i];
        else if (!strcmp(argv[i], "--tag") && i + 1 < argc)  file_tag   = argv[++i];
        else if (!strcmp(argv[i], "--ngl") && i + 1 < argc)  ngl        = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ctx") && i + 1 < argc)  n_ctx      = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--gen") && i + 1 < argc)  n_gen      = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--temp") && i + 1 < argc) temp       = (float) atof(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed       = (uint32_t) atoi(argv[++i]);
        else if (!strcmp(argv[i], "--rpc") && i + 1 < argc)  rpc_csv    = argv[++i];
        else if (!strcmp(argv[i], "--dry-run"))              list_only  = true;
        else { fprintf(stderr, "usage: %s -m model.gguf [-o out.jsonl] [-f corpus.txt --tag name] "
                               "[--ngl N] [--ctx N]\n", argv[0]); return 1; }
    }
    if (!model_path) { fprintf(stderr, "need -m <model.gguf>\n"); return 1; }

    // A corpus file replaces the built-in mixed prompts. The built-ins deliberately span
    // code/prose/math to expose distribution shift; that makes them the WRONG sample for the
    // question "does one long session concentrate on a few experts?", which is what a hot cache
    // actually has to serve.
    std::vector<prompt_spec> prompts = DEFAULT_PROMPTS;
    std::string corpus;
    if (file_path) {
        FILE * fp = fopen(file_path, "rb");
        if (!fp) { fprintf(stderr, "cannot read %s\n", file_path); return 1; }
        char buf[65536];
        size_t n;
        while ((n = fread(buf, 1, sizeof(buf), fp)) > 0) corpus.append(buf, n);
        fclose(fp);
        if (corpus.empty()) { fprintf(stderr, "%s is empty\n", file_path); return 1; }
        prompts.clear();
        prompts.push_back({file_tag, corpus.c_str()});
        fprintf(stderr, "E2: corpus %s (%zu chars) as one continuous session\n", file_path, corpus.size());
    }

    if (list_only) {
        for (size_t i = 0; i < prompts.size(); ++i)
            printf("%zu  %-6s %zu chars\n", i, prompts[i].tag, strlen(prompts[i].text));
        return 0;
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = ngl;
    if (rpc_csv) add_rpc_devices(rpc_csv);
    mparams.load_mode    = LLAMA_LOAD_MODE_MMAP;   // big weights against small RAM: mmap or nothing
    // CRITICAL on a memory-tight machine: the CPU backend otherwise repacks Q4_K into q4_K_8x8
    // for faster ARM matmul, allocating a SECOND full copy of the weights
    // (CPU_Mapped 19,026 MiB + CPU_REPACK 19,142 MiB = ~38 GB against 16 GB of RAM). That is what
    // killed earlier captures mid-graph and what ballooned swap to ~25 GB. Slower matmul, but it
    // actually completes.
    mparams.use_extra_bufts = false;

    fprintf(stderr, "E2: loading %s (ngl=%d, mmap=on)\n", model_path, ngl);
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }

    capture_state st;
    st.out = fopen(out_path, "w");
    if (!st.out) { fprintf(stderr, "cannot write %s\n", out_path); return 1; }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx             = n_ctx;
    // Batch is capped independently of context: compute buffers scale with BATCH, and a
    // 4096-token batch on a 256-expert MoE is what made the buffers unaffordable. Chunking is safe
    // now that positions are tracked per layer rather than per prompt.
    cparams.n_batch           = n_ctx < 512 ? n_ctx : 512;
    cparams.n_ubatch          = cparams.n_batch;
    cparams.cb_eval           = cb_eval;
    cparams.cb_eval_user_data = &st;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    long total_tokens = 0;
    const auto t_start = std::chrono::steady_clock::now();

    for (size_t p = 0; p < prompts.size(); ++p) {
        const auto & spec = prompts[p];

        // Size the buffer to the WHOLE corpus (llama_tokenize returns -needed if
        // the buffer is short); the decode loop below feeds it in n_batch slices,
        // and the KV window slides, so a corpus far longer than n_ctx is fine.
        std::vector<llama_token> toks((size_t) strlen(spec.text) + 16);
        int n = llama_tokenize(vocab, spec.text, (int32_t) strlen(spec.text),
                               toks.data(), (int32_t) toks.size(), true, false);
        if (n < 0) { toks.resize(-n); n = llama_tokenize(vocab, spec.text,
                     (int32_t) strlen(spec.text), toks.data(), (int32_t) toks.size(),
                     true, false); }
        if (n < 0) { fprintf(stderr, "tokenize failed on prompt %zu\n", p); return 1; }
        toks.resize(n);

        // Each prompt is an independent conversation; a shared KV cache would make prompt N's
        // routing depend on prompt N-1 and quietly contaminate the cross-domain comparison.
        llama_memory_clear(llama_get_memory(ctx), true);

        st.prompt_id = (int) p;
        st.tag       = spec.tag;
        st.layer_pos.clear();

        // llama_decode asserts n_tokens <= n_batch and does NOT chunk for you, so a corpus longer
        // than the batch must be fed in slices. Safe for the capture because positions are tracked
        // per layer, so a token's recorded position is its true offset in the session regardless of
        // which slice carried it.
        const int nb = n_ctx < 512 ? n_ctx : 512;
        for (int off = 0; off < n; off += nb) {
            const int cur = (n - off) < nb ? (n - off) : nb;
            if (llama_decode(ctx, llama_batch_get_one(toks.data() + off, cur)) != 0) {
                fprintf(stderr, "decode failed on prompt %zu at offset %d\n", p, off);
                return 1;
            }
        }
        total_tokens += n;
        fprintf(stderr, "E2: prompt %zu (%s) %d tokens\n", p, spec.tag, n);

        // ---- decode-phase capture: real generation, one token at a time.
        // Prefill reads foreign text in bulk; DECODE is the regime users live in
        // (the model looping on its own topic). Records are tagged "decode" so
        // the analysis can compare routing concentration between the regimes.
        if (n_gen > 0) {
            st.tag = "decode";
            llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
            if (temp > 0.0f) {
                llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp));
                llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));
            } else {
                llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
            }
            int produced = 0;
            for (; produced < n_gen; ++produced) {
                llama_token tok = llama_sampler_sample(smpl, ctx, -1);
                if (llama_vocab_is_eog(vocab, tok)) break;
                if (llama_decode(ctx, llama_batch_get_one(&tok, 1)) != 0) {
                    fprintf(stderr, "decode-gen failed at %d\n", produced);
                    break;
                }
            }
            llama_sampler_free(smpl);
            total_tokens += produced;
            fprintf(stderr, "E2: decode capture %d generated tokens\n", produced);
        }
    }

    const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();

    fclose(st.out);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    fprintf(stderr, "\nE2: %ld tokens, %ld routing records -> %s\n", total_tokens, st.lines, out_path);
    fprintf(stderr, "E2: prefill wall clock %.1fs (%.1f tok/s, ngl=%d) — indicative only, not the\n"
                    "    generation benchmark: prefill is compute-bound and batched.\n",
            secs, total_tokens / (secs > 0 ? secs : 1), ngl);
    return 0;
}
