#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <stdexcept>
#include <cstring>

#include "llama.h"
#include "common.h"

namespace py = pybind11;

// 将 token 序列转换成字符串
static std::string tokens_to_string(const llama_vocab *vocab, const std::vector<llama_token> &tokens)
{
    if (!vocab)
        return "";

    std::string result;
    char piece_buf[256];
    for (auto t : tokens)
    {
        int n = llama_token_to_piece(vocab, t, piece_buf, sizeof(piece_buf), 0, false);
        if (n > 0)
            result.append(piece_buf, n);
    }
    return result;
}

// Llama 引擎类
class LlamaEngine
{
public:
    LlamaEngine(const std::string &model_path, int n_gpu_layers, int n_ctx)
    {
        llama_backend_init();
        llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);

        auto mparams = llama_model_default_params();
        mparams.n_gpu_layers = n_gpu_layers;

        model = llama_model_load_from_file(model_path.c_str(), mparams);
        if (!model)
            throw std::runtime_error("C++ Error: Failed to load model.");

        auto cparams = llama_context_default_params();
        cparams.n_ctx = n_ctx;

        ctx = llama_init_from_model(model, cparams);
        if (!ctx)
            throw std::runtime_error("C++ Error: Failed to create context.");
    }

    ~LlamaEngine()
    {
        if (ctx)
            llama_free(ctx);
        if (model)
            llama_model_free(model);
        llama_backend_free();
    }

    std::string generate(const std::string &prompt, int max_tokens = 128)
    {
        if (!model || !ctx)
            return "C++ Error: Model or context not initialized.";

        const llama_vocab *vocab = llama_model_get_vocab(model);
        if (!vocab)
            return "C++ Error: Failed to get vocab.";

        std::vector<llama_token> tokens(llama_n_ctx(ctx));
        int n_prompt = llama_tokenize(vocab, prompt.c_str(), (int)prompt.size(),
                                      tokens.data(), (int)tokens.size(), true, false);
        if (n_prompt <= 0)
            return "C++ Error: Tokenization failed.";
        tokens.resize(n_prompt);

        // 将 prompt 送入模型
        for (auto &t : tokens)
        {
            llama_decode(ctx, llama_batch_get_one(&t, 1));
        }

        std::string result;
        int n_ctx = llama_n_ctx(ctx);
        int n_cur = n_prompt;

        llama_token eos_tok = llama_vocab_eos(vocab);

        while (n_cur < n_ctx && max_tokens-- > 0)
        {
            const float *logits = llama_get_logits(ctx);
            if (!logits)
                break;

            int n_vocab = llama_vocab_n_tokens(vocab);
            int best_id = 0;
            float best_val = logits[0];
            for (int i = 1; i < n_vocab; ++i)
            {
                if (logits[i] > best_val)
                {
                    best_val = logits[i];
                    best_id = i;
                }
            }

            llama_token new_tok = best_id;

            if (new_tok == eos_tok)
                break;

            if (llama_decode(ctx, llama_batch_get_one(&new_tok, 1)))
                break;

            result += tokens_to_string(vocab, std::vector<llama_token>{new_tok});
            ++n_cur;
        }

        return result;
    }

private:
    llama_model *model = nullptr;
    llama_context *ctx = nullptr;
};

// Pybind11 绑定
PYBIND11_MODULE(my_bridge, m)
{
    m.doc() = "A minimal llama.cpp bridge module using vocab API";

    py::class_<LlamaEngine>(m, "LlamaEngine")
        // 修改后的 C++ 绑定代码
        .def(py::init<const std::string &, int, int>(),
             py::arg("model_path"),
             py::arg("n_gpu_layers"),
             py::arg("n_ctx"))
        .def("generate", &LlamaEngine::generate,
             py::arg("prompt"),
             py::arg("max_tokens") = 128,
             py::call_guard<py::gil_scoped_release>());
}
