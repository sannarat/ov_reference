#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <regex>
#include <openvino/genai/llm_pipeline.hpp>

namespace fs = std::filesystem;

struct ScenarioResult {
    std::string model;
    std::string scenario_id;
    int max_new_tokens = 128;
    bool do_sample = false;
    float temperature = 0.0f;
    std::vector<std::string> replies;
    std::vector<std::vector<int64_t>> prompt_token_ids;
    std::vector<std::vector<int64_t>> reply_token_ids;
};

std::string escape_json(const std::string& str) {
    std::string res;
    for (char c : str) {
        switch (c) {
            case '"':  res += "\\\""; break;
            case '\\': res += "\\\\"; break;
            case '\b': res += "\\b"; break;
            case '\f': res += "\\f"; break;
            case '\n': res += "\\n"; break;
            case '\r': res += "\\r"; break;
            case '\t': res += "\\t"; break;
            default:   res += c; break;
        }
    }
    return res;
}

std::vector<int64_t> tensor_to_vec(const ov::Tensor& tensor) {
    std::vector<int64_t> vec;
    if (!tensor) return vec;

    size_t size = tensor.get_size();
    vec.reserve(size);

    if (tensor.get_element_type() == ov::element::i64) {
        const int64_t* data = tensor.data<int64_t>();
        vec.assign(data, data + size);
    } else if (tensor.get_element_type() == ov::element::i32) {
        const int32_t* data = tensor.data<int32_t>();
        for (size_t i = 0; i < size; ++i) {
            vec.push_back(data[i]);
        }
    }
    return vec;
}

std::string clean_think_block(const std::string& input) {
    std::regex think_regex(R"(<think>[\s\S]*?<\/think>\s*)");
    std::string cleaned = std::regex_replace(input, think_regex, "");

    if (cleaned.rfind("<think>", 0) == 0) {
        size_t last_break = cleaned.rfind('\n');
        if (last_break != std::string::npos && last_break + 1 < cleaned.size()) {
            cleaned = cleaned.substr(last_break + 1);
        }
    }

    size_t first = cleaned.find_first_not_of(" \t\n\r");
    if (first == std::string::npos) return "";
    size_t last = cleaned.find_last_not_of(" \t\n\r");
    return cleaned.substr(first, (last - first + 1));
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Diagnostic Error: Missing arguments.\nUsage: "
                  << argv[0] << " <models_dir> <output_json_path>" << std::endl;
        return 2;
    }

    std::string models_dir = argv[1];
    std::string output_file = argv[2];

    if (!fs::exists(models_dir) || !fs::is_directory(models_dir)) {
        std::cerr << "Diagnostic Error: Models directory does not exist or is not a directory: "
                  << models_dir << std::endl;
        return 1;
    }

    const std::vector<std::string> target_models = {
        "qwen3-0.6b-int4-ov",
        "llama-3.2-1b-int4-ov"
    };

    ov::genai::GenerationConfig config;
    config.max_new_tokens = 128;
    config.do_sample = false;
    config.temperature = 0.0f;

    std::vector<ScenarioResult> records;

    for (const auto& model_name : target_models) {
        fs::path model_path = fs::path(models_dir) / model_name;
        if (!fs::exists(model_path)) {
            std::cerr << "Diagnostic Error: Target model path not found: " << model_path << std::endl;
            return 1;
        }

        try {
            ov::genai::LLMPipeline pipe(model_path.string(), "CPU");
            auto tokenizer = pipe.get_tokenizer();

            {
                std::string prompt = "List 5 primary colors separated by commas.";
                ScenarioResult r{model_name, "scenario_en_colors"};
                r.prompt_token_ids.push_back(tensor_to_vec(tokenizer.encode(prompt).input_ids));

                std::string raw_reply = pipe.generate(prompt, config);
                std::string reply = clean_think_block(raw_reply);
                r.replies.push_back(reply);
                r.reply_token_ids.push_back(tensor_to_vec(tokenizer.encode(reply).input_ids));
                records.push_back(r);
            }

            {
                std::string prompt = "Назови 5 основных цветов через запятую.";
                ScenarioResult r{model_name, "scenario_ru_colors"};
                r.prompt_token_ids.push_back(tensor_to_vec(tokenizer.encode(prompt).input_ids));

                std::string raw_reply = pipe.generate(prompt, config);
                std::string reply = clean_think_block(raw_reply);
                r.replies.push_back(reply);
                r.reply_token_ids.push_back(tensor_to_vec(tokenizer.encode(reply).input_ids));
                records.push_back(r);
            }

            {
                std::string prompt = "What does this emoji mean: 🚀?";
                ScenarioResult r{model_name, "scenario_emoji"};
                r.prompt_token_ids.push_back(tensor_to_vec(tokenizer.encode(prompt).input_ids));

                std::string raw_reply = pipe.generate(prompt, config);
                std::string reply = clean_think_block(raw_reply);
                r.replies.push_back(reply);
                r.reply_token_ids.push_back(tensor_to_vec(tokenizer.encode(reply).input_ids));
                records.push_back(r);
            }
            
            {
                ScenarioResult r{model_name, "scenario_recall_memory"};
                pipe.start_chat();

                std::string turn1 = "Remember this exact secret word: 'Cobalt'. Reply with 'ACK' only.";
                r.prompt_token_ids.push_back(tensor_to_vec(tokenizer.encode(turn1).input_ids));
                std::string raw_rep1 = pipe.generate(turn1, config);
                std::string rep1 = clean_think_block(raw_rep1);
                r.replies.push_back(rep1);
                r.reply_token_ids.push_back(tensor_to_vec(tokenizer.encode(rep1).input_ids));

                std::string turn2 = "What was the exact secret word I asked you to remember?";
                r.prompt_token_ids.push_back(tensor_to_vec(tokenizer.encode(turn2).input_ids));
                std::string raw_rep2 = pipe.generate(turn2, config);
                std::string rep2 = clean_think_block(raw_rep2);
                r.replies.push_back(rep2);
                r.reply_token_ids.push_back(tensor_to_vec(tokenizer.encode(rep2).input_ids));

                pipe.finish_chat();
                records.push_back(r);
            }

        } catch (const std::exception& e) {
            std::cerr << "Diagnostic Exception for " << model_name << ": " << e.what() << std::endl;
            return 1;
        }
    }

    std::ofstream out(output_file);
    out << "[\n";
    for (size_t i = 0; i < records.size(); ++i) {
        const auto& r = records[i];
        out << "  {\n";
        out << "    \"model\": \"" << r.model << "\",\n";
        out << "    \"scenario_id\": \"" << r.scenario_id << "\",\n";
        out << "    \"parameters\": {\"max_new_tokens\": " << r.max_new_tokens << ", \"do_sample\": false, \"temperature\": 0.0},\n";

        out << "    \"replies\": [";
        for (size_t j = 0; j < r.replies.size(); ++j) {
            out << "\"" << escape_json(r.replies[j]) << "\"" << (j + 1 < r.replies.size() ? ", " : "");
        }
        out << "],\n";

        out << "    \"prompt_token_ids\": [";
        for (size_t j = 0; j < r.prompt_token_ids.size(); ++j) {
            out << "[";
            for (size_t k = 0; k < r.prompt_token_ids[j].size(); ++k) {
                out << r.prompt_token_ids[j][k] << (k + 1 < r.prompt_token_ids[j].size() ? "," : "");
            }
            out << "]" << (j + 1 < r.prompt_token_ids.size() ? ", " : "");
        }
        out << "],\n";

        out << "    \"reply_token_ids\": [";
        for (size_t j = 0; j < r.reply_token_ids.size(); ++j) {
            out << "[";
            for (size_t k = 0; k < r.reply_token_ids[j].size(); ++k) {
                out << r.reply_token_ids[j][k] << (k + 1 < r.reply_token_ids[j].size() ? "," : "");
            }
            out << "]" << (j + 1 < r.reply_token_ids.size() ? ", " : "");
        }
        out << "]\n";

        out << "  }" << (i + 1 < records.size() ? ",\n" : "\n");
    }
    out << "]\n";

    return 0;
}