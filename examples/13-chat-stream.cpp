#include "openai.hpp"

#include <iostream>
#include <string>

int main() {
    // Requires OPENAI_API_KEY (and optionally OPENAI_API_BASE) in the env.
    openai::start();

    nlohmann::json request = {
        {"model", "gpt-4.1-mini"},
        {"messages",
         {{{"role", "user"},
           {"content", "Stream a short paragraph about SSE;"}}}},
        {"stream", true}};

    std::string assembled;
    bool paused_once = false;

    openai::chat().stream(
        request,
        openai::ChatStreamCallbacks{
            // on_data
            [&](const nlohmann::json &chunk) -> openai::StreamControl {
                if (!chunk.contains("choices") ||
                    !chunk["choices"].is_array() || chunk["choices"].empty()) {
                    return openai::StreamControl::Continue;
                }
                const auto &delta = chunk["choices"][0]["delta"];
                if (delta.contains("content") && delta["content"].is_string()) {
                    const auto piece = delta["content"].get<std::string>();
                    assembled += piece;
                    std::cout << piece << std::flush;
                    /*if (!paused_once) {
                        paused_once = true;
                        std::cout << " [pause]\n";
                        return openai::StreamControl::Pause;
                    }*/
                    if (assembled.size() > 180) {
                        std::cout << " [stop]\n";
                        return openai::StreamControl::Stop; // demo early stop
                    }
                }
                return openai::StreamControl::Continue;
            },
            // on_done
            [&] { std::cout << "\n\nFull message: " << assembled << "\n"; },
            // on_error
            [&](const std::string &err) {
                std::cerr << "\n[stream error] " << err << "\n";
            },
            // control hook (not used here)
            {}});

    return 0;
}
