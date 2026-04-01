// ==========================================================
//     src/bridge.cpp (V22.0 - 终极修正版)
// ==========================================================
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <string>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <limits>

// 【【【 终极修正 1: 补上缺失的头文件！ 】】】
#include <random>

#include "llama.h"
#include "common.h"

namespace py = pybind11;

// --- tokens_to_string 函数 (来自我们测试成功的版本) ---
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

class LlamaEngine
{
public:
    // 构造函数：只加载模型
    LlamaEngine(const std::string &model_path, int n_gpu_layers)
    {
        llama_backend_init();
        auto mparams = llama_model_default_params();
        mparams.n_gpu_layers = n_gpu_layers;
        model = llama_load_model_from_file(model_path.c_str(), mparams);
        if (!model)
            throw std::runtime_error("C++: Failed to load model.");
        rng = std::mt19937(std::random_device{}());
    }

    // 析构函数：清理模型和可能存在的上下文
    ~LlamaEngine()
    {
        if (ctx)
            llama_free(ctx);
        if (model)
            llama_model_free(model);
        llama_backend_free();
    }

    // generate 函数：终极核心，每次都重建一个干净的上下文
    std::string generate(
        const std::string &prompt, int max_tokens = 2048, float temp = 0.8f,
        float repeat_penalty = 1.1f, int top_k = 40, float top_p = 0.95f)
    {
        // 核心改造：先销毁旧大脑，再创建新大脑
        if (ctx)
        {
            llama_free(ctx);
            ctx = nullptr;
        }
        auto cparams = llama_context_default_params();
        cparams.n_ctx = 4096; // 您可以根据需要调整这个值
        ctx = llama_init_from_model(model, cparams);
        if (!ctx)
            return "C++ Error: Failed to create new context.";

        const llama_vocab *vocab = llama_model_get_vocab(model);
        if (!vocab)
            return "C++ Error: Failed to get vocab.";

        // 【【【 终极修正 2: 使用正确的 tokenize 调用方式！ 】】】
        std::vector<llama_token> tokens(cparams.n_ctx);
        int n_prompt = llama_tokenize(vocab, prompt.c_str(), (int)prompt.size(), tokens.data(), (int)tokens.size(), true, false);
        if (n_prompt <= 0)
            return "C++ Error: Tokenization failed.";
        tokens.resize(n_prompt);

        for (auto &t : tokens)
        {
            llama_decode(ctx, llama_batch_get_one(&t, 1));
        }

        std::string result;
        llama_token eos_tok = llama_vocab_eos(vocab);
        int n_gen = 0;

        while (n_gen < max_tokens)
        {
            // --- 这里是完整的、我们测试成功的手动采样逻辑 (V18) ---
            const float *logits = llama_get_logits(ctx);
            if (!logits)
                break;
            int n_vocab = llama_vocab_n_tokens(vocab);

            std::vector<llama_token_data> candidates;
            candidates.reserve(n_vocab);
            for (llama_token token_id = 0; token_id < n_vocab; ++token_id)
            {
                candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
            }

            const int last_n = 64;
            float inv_penalty = 1.0f / repeat_penalty;
            for (int i = 0; i < (int)candidates.size(); ++i)
            {
                for (int j = 0; j < last_n && j < (int)tokens.size(); ++j)
                {
                    if (candidates[i].id == tokens[tokens.size() - 1 - j])
                    {
                        if (candidates[i].logit > 0)
                            candidates[i].logit *= inv_penalty;
                        else
                            candidates[i].logit /= inv_penalty;
                    }
                }
            }

            if (top_k > 0 && top_k < n_vocab)
            {
                std::sort(candidates.begin(), candidates.end(), [](const llama_token_data &a, const llama_token_data &b)
                          { return a.logit > b.logit; });
                candidates.resize(top_k);
            }

            float max_l = -std::numeric_limits<float>::infinity();
            for (const auto &c : candidates)
                max_l = std::max(max_l, c.logit);
            float current_sum = 0.0f;
            for (auto &c : candidates)
            {
                c.logit = expf((c.logit - max_l) / temp);
                current_sum += c.logit;
            }
            if (current_sum > 0)
                for (auto &c : candidates)
                    c.logit /= current_sum;

            if (top_p > 0.0f && top_p < 1.0f)
            {
                std::sort(candidates.begin(), candidates.end(), [](const llama_token_data &a, const llama_token_data &b)
                          { return a.logit > b.logit; });
                float cumulative_prob = 0.0f;
                int last_idx = 0;
                for (size_t j = 0; j < candidates.size(); ++j)
                {
                    cumulative_prob += candidates[j].logit;
                    if (cumulative_prob > top_p)
                    {
                        last_idx = j;
                        break;
                    }
                }
                candidates.resize(last_idx + 1);
                float new_sum = 0.0f;
                for (const auto &c : candidates)
                    new_sum += c.logit;
                if (new_sum > 0)
                    for (auto &c : candidates)
                        c.logit /= new_sum;
            }

            std::uniform_real_distribution<> dist(0.0f, 1.0f);
            float p_roll = dist(rng);
            float cumulative_p = 0.0f;
            llama_token new_tok = candidates.empty() ? eos_tok : candidates[0].id;
            for (const auto &c : candidates)
            {
                cumulative_p += c.logit;
                if (p_roll < cumulative_p)
                {
                    new_tok = c.id;
                    break;
                }
            }

            if (new_tok == eos_tok)
                break;

            if (llama_decode(ctx, llama_batch_get_one(&new_tok, 1)))
                break;

            result += tokens_to_string(vocab, std::vector<llama_token>{new_tok});
            tokens.push_back(new_tok);
            n_gen++;
        }
        return result;
    }

private:
    llama_model *model = nullptr;
    llama_context *ctx = nullptr;
    std::mt19937 rng;
};

// Pybind11 绑定 (终极版)
PYBIND11_MODULE(my_bridge, m)
{
    m.doc() = "A stable, stateless LlamaEngine bridge";
    py::class_<LlamaEngine>(m, "LlamaEngine")
        .def(py::init<const std::string &, int>(),
             py::arg("model_path"), py::arg("n_gpu_layers"))
        .def("generate", &LlamaEngine::generate,
             py::arg("prompt"), py::arg("max_tokens") = 2048, py::arg("temp") = 0.8f,
             py::arg("repeat_penalty") = 1.1f, py::arg("top_k") = 40, py::arg("top_p") = 0.95f,
             py::call_guard<py::gil_scoped_release>());
}