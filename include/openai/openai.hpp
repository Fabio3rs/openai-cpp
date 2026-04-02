// The MIT License (MIT)
// 
// Copyright (c) 2023 Olrea, Florian Dang
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef OPENAI_HPP_
#define OPENAI_HPP_


#include <cstdio>
#if OPENAI_VERBOSE_OUTPUT
#pragma message ("OPENAI_VERBOSE_OUTPUT is ON")
#endif

#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <sstream>
#include <mutex>
#include <chrono>
#include <algorithm>
#include <cstdlib>
#include <map>
#include <memory>
#include <functional>
#include <utility>
#include <string_view>
#include <atomic>
#include <optional>
#include <memory>

#ifndef CURL_STATICLIB
#include <curl/curl.h>
#else 
#include "curl/curl.h"
#endif

#include <nlohmann/json.hpp>  // nlohmann/json

namespace openai {

namespace _detail {

// Json alias
using Json = nlohmann::json;

struct Response {
    std::string text;
    bool        is_error;
    std::string error_message;
};

enum class StreamControl { Continue, Pause, Stop };

struct SseEvent {
    std::string event;
    std::string data;
};

struct ChatStreamCallbacks {
    std::function<StreamControl(const Json&)> on_data;
    std::function<void()> on_done;
    std::function<void(const std::string&)> on_error;
    // Optional external control hook polled before processing each chunk
    std::function<StreamControl()> control;
};

// Tiny SSE parser: accumulates bytes, splits on blank line, tolerates \r\n.
class SseParser {
  public:
    using EventCallback = std::function<void(const SseEvent&)>;

    explicit SseParser(EventCallback cb) : callback_{std::move(cb)} {}

    // Feed raw bytes from HTTP body; returns false if callback signals stop.
    bool feed(std::string_view chunk,
              const std::function<bool()> &should_stop = {}) {
        buffer_.append(chunk.data(), chunk.size());
        while (true) {
            const auto pos = find_separator();
            if (pos == std::string::npos) break;
            const std::string event_block = buffer_.substr(0, pos);
            if (!consume_event(event_block)) return false;
            if (should_stop && should_stop()) return false;
            buffer_.erase(0, pos + separator_len_);
        }
        return true;
    }

  private:
    size_t find_separator() {
        // Check for \r\n\r\n first, then \n\n.
        const auto crlf = buffer_.find("\r\n\r\n");
        if (crlf != std::string::npos) {
            separator_len_ = 4;
            return crlf;
        }
        const auto lf = buffer_.find("\n\n");
        if (lf != std::string::npos) {
            separator_len_ = 2;
            return lf;
        }
        return std::string::npos;
    }

    bool consume_event(const std::string& block) {
        std::string current_event;
        std::string current_data;

        std::istringstream iss(block);
        std::string line;
        while (std::getline(iss, line)) {
            // trim trailing carriage return
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.rfind("event:", 0) == 0) {
                auto value = line.substr(6);
                if (!value.empty() && value.front() == ' ') value.erase(0, 1);
                current_event = std::move(value);
            } else if (line.rfind("data:", 0) == 0) {
                auto value = line.substr(5);
                if (!value.empty() && value.front() == ' ') value.erase(0, 1);
                if (!current_data.empty()) current_data.push_back('\n');
                current_data += value;
            }
        }

        if (current_event.empty() && current_data.empty()) {
            return true; // ignore keep-alives / comments
        }

        callback_(SseEvent{std::move(current_event), std::move(current_data)});
        return true;
    }

    std::string buffer_;
    size_t separator_len_{2};
    EventCallback callback_;
};

// Simple curl Session inspired by CPR
class Session {
public:
    Session(bool throw_exception) : throw_exception_{throw_exception} {
        initCurl();
    }

    Session(bool throw_exception, std::string proxy_url) : throw_exception_{ throw_exception } {
        initCurl();
        setProxyUrl(proxy_url);
    }

    ~Session() { 
        curl_easy_cleanup(curl_); 
        curl_global_cleanup();
        if (mime_form_ != nullptr) {
            curl_mime_free(mime_form_);
        }
    }

    void initCurl() {
        curl_global_init(CURL_GLOBAL_ALL);
        curl_ = curl_easy_init();
        if (curl_ == nullptr) {
            throw std::runtime_error("curl cannot initialize"); // here we throw it shouldn't happen
        }
        curl_easy_setopt(curl_, CURLOPT_NOSIGNAL, 1);
        applyTimeouts();
        applyTlsOptions();
    }

    void ignoreSSL() {
        curl_easy_setopt(curl_, CURLOPT_SSL_VERIFYPEER, 0L);
    }
 
    void setUrl(const std::string& url) { url_ = url; }

    void setToken(const std::string& token, const std::string& organization) {
        token_ = token;
        organization_ = organization;
    }
    void setProxyUrl(const std::string& url) {
        proxy_url_ = url; 
        curl_easy_setopt(curl_, CURLOPT_PROXY, proxy_url_.c_str());
        
    }

    void setBeta(const std::string& beta) { beta_ = beta; }
    void setTlsOptions(bool verify_peer, bool verify_host, const std::string& ca_info, const std::string& ca_path);
    void setTimeouts(std::chrono::milliseconds connect_timeout, std::chrono::milliseconds total_timeout);

    void setBody(const std::string& data);
    void setMultiformPart(const std::pair<std::string, std::string>& filefield_and_filepath, const std::map<std::string, std::string>& fields);
    
    Response getPrepare();
    Response postPrepare(const std::string& contentType = "");
    Response deletePrepare();
    Response makeRequest(const std::string& contentType = "");

    // Streaming (SSE) -------------------------------------------------------
    // Handler returns StreamControl for this write chunk.
    using StreamHandler = std::function<StreamControl(std::string_view)>;

    Response streamRequest(std::string_view contentType, StreamHandler handler);
    std::string easyEscape(const std::string& text);

private:
    void applyTlsOptions();
    void applyTimeouts();

    static size_t writeFunction(void* ptr, size_t size, size_t nmemb, std::string* data) {
        data->append((char*) ptr, size * nmemb);
        return size * nmemb;
    }

private:
    CURL*       curl_;
    CURLcode    res_;
    curl_mime   *mime_form_ = nullptr;
    std::string url_;
    std::string proxy_url_;
    std::string token_;
    std::string organization_;
    std::string beta_;
    bool verify_peer_ = true;
    bool verify_host_ = true;
    std::string ca_info_;
    std::string ca_path_;
    std::chrono::milliseconds connect_timeout_{std::chrono::milliseconds{5000}};
    std::chrono::milliseconds total_timeout_{std::chrono::milliseconds{30000}};

    bool        throw_exception_;
    std::mutex  mutex_request_;
    std::shared_ptr<StreamHandler> active_stream_handler_;
    bool stop_requested_{false};
};

inline void Session::applyTimeouts() {
    if (curl_) {
        const auto connect_ms = std::max<long>(0L, static_cast<long>(connect_timeout_.count()));
        const auto total_ms   = std::max<long>(0L, static_cast<long>(total_timeout_.count()));
        curl_easy_setopt(curl_, CURLOPT_CONNECTTIMEOUT_MS, connect_ms);
        curl_easy_setopt(curl_, CURLOPT_TIMEOUT_MS, total_ms);
    }
}

inline void Session::setTimeouts(std::chrono::milliseconds connect_timeout, std::chrono::milliseconds total_timeout) {
    connect_timeout_ = std::max(connect_timeout, std::chrono::milliseconds{0});
    total_timeout_   = std::max(total_timeout, std::chrono::milliseconds{0});
    applyTimeouts();
}

inline void Session::applyTlsOptions() {
    if (curl_) {
        curl_easy_setopt(curl_, CURLOPT_SSL_VERIFYPEER, verify_peer_ ? 1L : 0L);
        curl_easy_setopt(curl_, CURLOPT_SSL_VERIFYHOST, verify_host_ ? 2L : 0L);
        if (!ca_info_.empty()) {
            curl_easy_setopt(curl_, CURLOPT_CAINFO, ca_info_.c_str());
        }
        if (!ca_path_.empty()) {
            curl_easy_setopt(curl_, CURLOPT_CAPATH, ca_path_.c_str());
        }
    }
}

inline void Session::setTlsOptions(bool verify_peer, bool verify_host, const std::string& ca_info, const std::string& ca_path) {
    verify_peer_ = verify_peer;
    verify_host_ = verify_host;
    ca_info_ = ca_info;
    ca_path_ = ca_path;
    applyTlsOptions();
}

inline void Session::setBody(const std::string& data) { 
    if (curl_) {
        curl_easy_setopt(curl_, CURLOPT_POSTFIELDSIZE, data.length());
        curl_easy_setopt(curl_, CURLOPT_POSTFIELDS, data.data());
    }
}

inline void Session::setMultiformPart(const std::pair<std::string, std::string>& fieldfield_and_filepath, const std::map<std::string, std::string>& fields) {
    // https://curl.se/libcurl/c/curl_mime_init.html
    if (curl_) {
        if (mime_form_ != nullptr) {
            curl_mime_free(mime_form_);
            mime_form_ = nullptr;
        }
        curl_mimepart *field = nullptr;

        mime_form_ = curl_mime_init(curl_);
    
        field = curl_mime_addpart(mime_form_);
        curl_mime_name(field, fieldfield_and_filepath.first.c_str());
        curl_mime_filedata(field, fieldfield_and_filepath.second.c_str());

        for (const auto &field_pair : fields) {
            field = curl_mime_addpart(mime_form_);
            curl_mime_name(field, field_pair.first.c_str());
            curl_mime_data(field, field_pair.second.c_str(), CURL_ZERO_TERMINATED);
        }
        
        curl_easy_setopt(curl_, CURLOPT_MIMEPOST, mime_form_);
    }
}

inline Response Session::getPrepare() {
    if (curl_) {
        curl_easy_setopt(curl_, CURLOPT_HTTPGET, 1L);
        curl_easy_setopt(curl_, CURLOPT_POST, 0L);
        curl_easy_setopt(curl_, CURLOPT_NOBODY, 0L);
        curl_easy_setopt(curl_, CURLOPT_CUSTOMREQUEST, nullptr);
    }
    return makeRequest();
}

inline Response Session::postPrepare(const std::string& contentType) {
    if (curl_) {
        curl_easy_setopt(curl_, CURLOPT_HTTPGET, 0L);
        curl_easy_setopt(curl_, CURLOPT_POST, 1L);
        curl_easy_setopt(curl_, CURLOPT_NOBODY, 0L);
        curl_easy_setopt(curl_, CURLOPT_CUSTOMREQUEST, nullptr);
    }
    return makeRequest(contentType);
}

inline Response Session::deletePrepare() {
    if (curl_) {
        curl_easy_setopt(curl_, CURLOPT_HTTPGET, 0L);
        curl_easy_setopt(curl_, CURLOPT_NOBODY, 0L);
        curl_easy_setopt(curl_, CURLOPT_POST, 0L);
        curl_easy_setopt(curl_, CURLOPT_CUSTOMREQUEST, "DELETE");
    }
    return makeRequest();
}

inline Response Session::makeRequest(const std::string& contentType) {
    std::lock_guard<std::mutex> lock(mutex_request_);
    active_stream_handler_.reset();
    
    struct SListFreeAll {
        void operator()(curl_slist* list) const noexcept {
            if (list) {
                curl_slist_free_all(list);
            }
        }
    };
    using slistptr_t = std::unique_ptr<curl_slist, SListFreeAll>;

    slistptr_t headers{nullptr};
    if (!contentType.empty()) {
        headers.reset(curl_slist_append(headers.release(), std::string{"Content-Type: " + contentType}.c_str()));
        if (contentType == "multipart/form-data") {
            headers.reset(curl_slist_append(headers.release(), "Expect:"));
        }
    }
    headers.reset(curl_slist_append(headers.release(), std::string{"Authorization: Bearer " + token_}.c_str()));
    if (!organization_.empty()) {
        headers.reset(curl_slist_append(headers.release(), std::string{"OpenAI-Organization: " + organization_}.c_str()));
    }
    if (!beta_.empty()) {
        headers.reset(curl_slist_append(headers.release(), std::string{"OpenAI-Beta: " + beta_}.c_str()));
    }
    curl_easy_setopt(curl_, CURLOPT_HTTPHEADER, headers.get());
    curl_easy_setopt(curl_, CURLOPT_URL, url_.c_str());
    
    std::string response_string;
    std::string header_string;
    curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION, writeFunction);
    curl_easy_setopt(curl_, CURLOPT_WRITEDATA, &response_string);
    curl_easy_setopt(curl_, CURLOPT_HEADERDATA, &header_string);

    res_ = curl_easy_perform(curl_);
    active_stream_handler_.reset();

    bool is_error = false;
    std::string error_msg{};
    if(res_ != CURLE_OK) {
        is_error = true;
        error_msg = "OpenAI curl_easy_perform() failed: " + std::string{curl_easy_strerror(res_)};
        if (throw_exception_) {
            throw std::runtime_error(error_msg);
        }
        else {
            std::cerr << error_msg << '\n';
        }
    }

    return { response_string, is_error, error_msg };
}

inline Response Session::streamRequest(std::string_view contentType,
                                       StreamHandler handler) {
    std::lock_guard<std::mutex> lock(mutex_request_);
    stop_requested_ = false;
    if (curl_) {
        curl_easy_setopt(curl_, CURLOPT_HTTPGET, 0L);
        curl_easy_setopt(curl_, CURLOPT_POST, 1L);
        curl_easy_setopt(curl_, CURLOPT_NOBODY, 0L);
        curl_easy_setopt(curl_, CURLOPT_CUSTOMREQUEST, nullptr);
    }

    struct SListFreeAll {
        void operator()(curl_slist* list) const noexcept {
            if (list) {
                curl_slist_free_all(list);
            }
        }
    };
    using slistptr_t = std::unique_ptr<curl_slist, SListFreeAll>;

    slistptr_t headers{nullptr};
    if (!contentType.empty()) {
        headers.reset(curl_slist_append(headers.release(),
                                        (std::string("Content-Type: ").append(contentType)).c_str()));
        if (contentType == "multipart/form-data") {
            headers.reset(curl_slist_append(headers.release(), "Expect:"));
        }
    }
    headers.reset(curl_slist_append(headers.release(),
                                    std::string{"Authorization: Bearer " + token_}.c_str()));
    if (!organization_.empty()) {
        headers.reset(curl_slist_append(headers.release(),
                                        std::string{"OpenAI-Organization: " + organization_}.c_str()));
    }
    if (!beta_.empty()) {
        headers.reset(curl_slist_append(headers.release(),
                                        std::string{"OpenAI-Beta: " + beta_}.c_str()));
    }
    curl_easy_setopt(curl_, CURLOPT_HTTPHEADER, headers.get());
    curl_easy_setopt(curl_, CURLOPT_URL, url_.c_str());

    active_stream_handler_ = std::make_shared<StreamHandler>(std::move(handler));

    // Streaming write callback: forward raw bytes to handler, which can abort.
    std::string header_string;
    auto write_fn = +[](char* ptr, size_t size, size_t nmemb, void* userdata) -> size_t {
        const size_t total = size * nmemb;
        auto* self = static_cast<Session*>(userdata);
        if (!self) {
            return total;
        }
        auto fn_ptr = self->active_stream_handler_;
        if (!fn_ptr) {
            return total;
        }
        StreamControl decision = StreamControl::Continue;
        try {
            decision = (*fn_ptr)(std::string(ptr, total));
        } catch (...) {
            decision = StreamControl::Stop;
        }
        switch (decision) {
        case StreamControl::Continue:
            return total;
        case StreamControl::Pause:
            curl_easy_pause(self->curl_, CURLPAUSE_RECV);
            curl_easy_pause(self->curl_, CURLPAUSE_CONT);
            return CURL_WRITEFUNC_PAUSE;
        case StreamControl::Stop:
        default:
            self->stop_requested_ = true;
            return 0; // abort transfer
        }
    };

    auto headerfunc = +[](char *ptr, size_t size, size_t nmemb, void *userdata) -> size_t {
        try {
            auto *s = static_cast<std::string *>(userdata);
            s->append(ptr, size * nmemb);
        } catch (...) {
            // Handle any exceptions that may occur
            // NOLINT
        }
        return size * nmemb;
    };
    curl_easy_setopt(curl_, CURLOPT_HEADERFUNCTION, headerfunc);
    curl_easy_setopt(curl_, CURLOPT_HEADERDATA, &header_string);

    curl_easy_setopt(curl_, CURLOPT_WRITEFUNCTION, write_fn);
    curl_easy_setopt(curl_, CURLOPT_WRITEDATA, this);

    res_ = curl_easy_perform(curl_);

    bool is_error = false;
    std::string error_msg{};
    const bool stopped_intentionally = stop_requested_;
    stop_requested_ = false;
    active_stream_handler_.reset();
    if (res_ != CURLE_OK) {
        if (res_ == CURLE_WRITE_ERROR && stopped_intentionally) {
            is_error = false;
        } else {
            is_error = true;
            error_msg = "OpenAI stream curl_easy_perform() failed: " +
                        std::string{curl_easy_strerror(res_)};
            if (throw_exception_) {
                throw std::runtime_error(error_msg);
            } else {
                std::cerr << error_msg << '\n';
            }
        }
    }

    // No aggregated body in streaming mode; handler consumes data. We still
    // return a Response for uniform error shape.
    return {std::string{}, is_error, error_msg};
}

inline std::string Session::easyEscape(const std::string& text) {
    char *encoded_output = curl_easy_escape(curl_, text.c_str(), static_cast<int>(text.length()));
    const auto str = std::string{ encoded_output };
    curl_free(encoded_output);
    return str;
}

// forward declaration for category structures
class  OpenAI;

// https://platform.openai.com/docs/api-reference/models
// List and describe the various models available in the API. You can refer to the Models documentation to understand what models are available and the differences between them.
struct CategoryModel {
    Json list();
    Json retrieve(const std::string& model);

    CategoryModel(OpenAI& openai) : openai_{openai} {}
private:
    OpenAI& openai_;
};

// https://platform.openai.com/docs/api-reference/assistants
// Build assistants that can call models and use tools to perform tasks.
struct CategoryAssistants {
    Json create(Json input);
    Json retrieve(const std::string& assistants);
    Json modify(const std::string& assistants, Json input);
    Json del(const std::string& assistants);
    Json list();
    Json createFile(const std::string& assistants, Json input);
    Json retrieveFile(const std::string& assistants, const std::string& files);
    Json delFile(const std::string& assistants, const std::string& files);
    Json listFile(const std::string& assistants);

    CategoryAssistants(OpenAI& openai) : openai_{openai} {}
private:
    OpenAI& openai_;
};

// https://platform.openai.com/docs/api-reference/threads
// Create threads that assistants can interact with.
struct CategoryThreads {
    Json create();
    Json retrieve(const std::string& threads);
    Json modify(const std::string& threads, Json input);
    Json del(const std::string& threads);
    Json list();

    // https://platform.openai.com/docs/api-reference/messages
    // Create messages within threads
    Json createMessage(const std::string& threads, Json input);
    Json retrieveMessage(const std::string& threads, const std::string& messages);
    Json modifyMessage(const std::string& threads, const std::string& messages, Json input);
    Json listMessage(const std::string& threads);
    Json retrieveMessageFile(const std::string& threads, const std::string& messages, const std::string& files);
    Json listMessageFile(const std::string& threads, const std::string& messages);

    // https://platform.openai.com/docs/api-reference/runs
    // Represents an execution run on a thread.
    Json createRun(const std::string& threads, Json input);
    Json retrieveRun(const std::string& threads, const std::string& runs);
    Json modifyRun(const std::string& threads, const std::string& runs, Json input);
    Json listRun(const std::string& threads);
    Json submitToolOutputsToRun(const std::string& threads, const std::string& runs, Json input);
    Json cancelRun(const std::string& threads, const std::string& runs);
    Json createThreadAndRun(Json input);
    Json retrieveRunStep(const std::string& threads, const std::string& runs, const std::string& steps);
    Json listRunStep(const std::string& threads, const std::string& runs);

    CategoryThreads(OpenAI& openai) : openai_{openai} {}
private:
    OpenAI& openai_;
};

// https://platform.openai.com/docs/api-reference/completions
// Given a prompt, the model will return one or more predicted completions, and can also return the probabilities of alternative tokens at each position.
struct CategoryCompletion {
    Json create(Json input);

    // Streaming alias to chat streaming for parity with legacy API.
    void stream(Json input, const ChatStreamCallbacks &cb);

    CategoryCompletion(OpenAI& openai) : openai_{openai} {}

private:
    OpenAI& openai_;
};

// https://platform.openai.com/docs/api-reference/chat
// Given a prompt, the model will return one or more predicted chat completions.
struct CategoryChat {
    Json create(Json input);

    using StreamCallbacks = ChatStreamCallbacks;

    void stream(Json input, const StreamCallbacks& cb);

    CategoryChat(OpenAI& openai) : openai_{openai} {}

private:
    OpenAI& openai_;
};

// https://platform.openai.com/docs/api-reference/audio
// Learn how to turn audio into text.
struct CategoryAudio {
    Json transcribe(Json input);
    Json translate(Json input);

    CategoryAudio(OpenAI& openai) : openai_{openai} {}

private:
    OpenAI& openai_;
};

// https://platform.openai.com/docs/api-reference/edits
// Given a prompt and an instruction, the model will return an edited version of the prompt.
struct CategoryEdit {
    Json create(Json input);

    CategoryEdit(OpenAI& openai) : openai_{openai} {}

private:
    OpenAI& openai_;
};


// https://platform.openai.com/docs/api-reference/images
// Given a prompt and/or an input image, the model will generate a new image.
struct CategoryImage {
    Json create(Json input);
    Json edit(Json input);
    Json variation(Json input);

    CategoryImage(OpenAI& openai) : openai_{openai} {}

private:
    OpenAI& openai_;
};

// https://platform.openai.com/docs/api-reference/embeddings
// Get a vector representation of a given input that can be easily consumed by machine learning models and algorithms.
struct CategoryEmbedding {
    Json create(Json input);
    CategoryEmbedding(OpenAI& openai) : openai_{openai} {}

private:
    OpenAI& openai_;
};

struct FileRequest {
    std::string file;
    std::string purpose;
};

// https://platform.openai.com/docs/api-reference/files
// Files are used to upload documents that can be used with features like Fine-tuning.
struct CategoryFile {
    Json list();
    Json upload(Json input);
    Json del(const std::string& file); // TODO
    Json retrieve(const std::string& file_id);
    Json content(const std::string& file_id);

    CategoryFile(OpenAI& openai) : openai_{openai} {}

private:
    OpenAI& openai_;
};

// https://platform.openai.com/docs/api-reference/fine-tunes
// Manage fine-tuning jobs to tailor a model to your specific training data.
struct CategoryFineTune {
    Json create(Json input);
    Json list();
    Json retrieve(const std::string& fine_tune_id);
    Json content(const std::string& fine_tune_id);
    Json cancel(const std::string& fine_tune_id);
    Json events(const std::string& fine_tune_id);
    Json del(const std::string& model);

    CategoryFineTune(OpenAI& openai) : openai_{openai} {}

private:
    OpenAI& openai_;
};

// https://platform.openai.com/docs/api-reference/moderations
// Given a input text, outputs if the model classifies it as violating OpenAI's content policy.
struct CategoryModeration {
    Json create(Json input);

    CategoryModeration(OpenAI& openai) : openai_{openai} {}

private:
    OpenAI& openai_;
};


// OpenAI
class OpenAI {
public:
    OpenAI(const std::string& token = "", const std::string& organization = "", bool throw_exception = true, const std::string& api_base_url = "", const std::string& beta = "") 
        : session_{throw_exception}, token_{token}, organization_{organization}, throw_exception_{throw_exception} {
            if (token.empty()) {
                if(const char* env_p = std::getenv("OPENAI_API_KEY")) {
                    token_ = std::string{env_p};
                }
            }
            if (api_base_url.empty()) {
                if(const char* env_p = std::getenv("OPENAI_API_BASE")) {
                    base_url = std::string{env_p} + "/";
                }
                else {
                    base_url = "https://api.openai.com/v1/";
                }
            }
            else {
                base_url = api_base_url;
            }
            session_.setUrl(base_url);
            session_.setToken(token_, organization_);
            session_.setBeta(beta);
            session_.setTimeouts(std::chrono::milliseconds{5000}, std::chrono::milliseconds{30000});
            session_.setTlsOptions(true, true, "", "");
        }
    
    OpenAI(const OpenAI&)               = delete;
    OpenAI& operator=(const OpenAI&)    = delete;
    OpenAI(OpenAI&&)                    = delete;
    OpenAI& operator=(OpenAI&&)         = delete;

    auto& setToken(const std::string& token = "", const std::string& organization = "") { session_.setToken(token, organization); return *this; }

    auto& setProxy(const std::string& url) { session_.setProxyUrl(url); return *this; }

    auto& setBeta(const std::string& beta) { session_.setBeta(beta); return *this; }

    // Configure TLS verification (defaults: verify_peer = true, verify_host = true).
    auto& setTlsOptions(bool verify_peer = true, bool verify_host = true, const std::string& ca_info = "", const std::string& ca_path = "") { session_.setTlsOptions(verify_peer, verify_host, ca_info, ca_path); return *this; }

    // Convenience to disable TLS verification (use only for debugging).
    auto& setInsecure() { setTlsOptions(false, false); return *this; }

    // Configure request timeouts.
    auto& setTimeouts(std::chrono::milliseconds connect_timeout, std::chrono::milliseconds total_timeout) { session_.setTimeouts(connect_timeout, total_timeout); return *this; }

    // void change_token(const std::string& token) { token_ = token; };
    auto& setThrowException(bool throw_exception) { throw_exception_ = throw_exception; return *this; }

    auto& setMultiformPart(const std::pair<std::string, std::string>& filefield_and_filepath, const std::map<std::string, std::string>& fields) { session_.setMultiformPart(filefield_and_filepath, fields); return *this; }

    Json post(const std::string& suffix, const std::string& data, const std::string& contentType) {
        setParameters(suffix, data, contentType);
        auto response = session_.postPrepare(contentType);
        if (response.is_error){ 
            trigger_error(response.error_message);
        }

        Json json{};
        if (isJson(response.text)){

            json = Json::parse(response.text); 
            checkResponse(json);
        }
        else{
          #if OPENAI_VERBOSE_OUTPUT
            std::cerr << "Response is not a valid JSON";
            std::cout << "<< " << response.text << "\n";
          #endif
        }
       
        return json;
    }

    Json get(const std::string& suffix, const std::string& data = "") {
        setParameters(suffix, data);
        auto response = session_.getPrepare();
        if (response.is_error) { trigger_error(response.error_message); }

        Json json{};
        if (isJson(response.text)) {
            json = Json::parse(response.text);
            checkResponse(json);
        }
        else {
          #if OPENAI_VERBOSE_OUTPUT
            std::cerr << "Response is not a valid JSON\n";
            std::cout << "<< " << response.text<< "\n";
          #endif
            json = Json{{"Result", response.text}};
        }
        return json;
    }

    Json post(const std::string& suffix, const Json& json, const std::string& contentType="application/json") {
        return post(suffix, json.dump(), contentType);
    }

    Json del(const std::string& suffix) {
        setParameters(suffix, "");
        auto response = session_.deletePrepare();
        if (response.is_error) { trigger_error(response.error_message); }

        Json json{};
        if (isJson(response.text)) {
            json = Json::parse(response.text);
            checkResponse(json);
        }
        else {
          #if OPENAI_VERBOSE_OUTPUT
            std::cerr << "Response is not a valid JSON\n";
            std::cout << "<< " << response.text<< "\n";
          #endif
        }
        return json;
    }

    std::string easyEscape(const std::string& text) { return session_.easyEscape(text); }

    void debug() const { std::cout << token_ << '\n'; }

    void setBaseUrl(const std::string &url) {
        base_url = url;
    }

    std::string getBaseUrl() const {
        return base_url;
    }

    // Streaming POST helper (SSE) used by high-level categories.
    void stream(const std::string &suffix, const Json &json,
                const ChatStreamCallbacks &cb);

private:
    std::string base_url;

    void setParameters(const std::string& suffix, const std::string& data, const std::string& contentType = "") {
        auto complete_url =  base_url+ suffix;
        session_.setUrl(complete_url);

        if (contentType != "multipart/form-data") {
            session_.setBody(data);
        }

        #if OPENAI_VERBOSE_OUTPUT
            std::cout << "<< request: "<< complete_url << "  " << data << '\n';
        #endif
    }

    void checkResponse(const Json& json) {
        if (json.count("error")) {
            auto reason = json["error"].dump();
            trigger_error(reason);

            #if OPENAI_VERBOSE_OUTPUT
                std::cerr << ">> response error :\n" << json.dump(2) << "\n";
            #endif
        } 
    }

    // as of now the only way
    bool isJson(const std::string &data){
        bool rc = true;
        try {
            auto json = Json::parse(data); // throws if no json 
        }
        catch (std::exception &){
            rc = false;
        }
        return(rc);
    }

    void trigger_error(const std::string& msg) {
        if (throw_exception_) {
            throw std::runtime_error(msg);
        }
        else {
            std::cerr << "[OpenAI] error. Reason: " << msg << '\n';
        }
    }

public:
    CategoryModel           model     {*this};
    CategoryAssistants      assistant {*this};
    CategoryThreads         thread    {*this};
    CategoryCompletion      completion{*this};
    CategoryEdit            edit      {*this};
    CategoryImage           image     {*this};
    CategoryEmbedding       embedding {*this};
    CategoryFile            file      {*this};
    CategoryFineTune        fine_tune {*this};
    CategoryModeration      moderation{*this};
    CategoryChat            chat      {*this};
    CategoryAudio           audio     {*this};
    // CategoryEngine          engine{*this}; // Not handled since deprecated (use Model instead)

private:
    Session                 session_;
    std::string             token_;
    std::string             organization_;
    bool                    throw_exception_;
};

inline std::string bool_to_string(const bool b) {
    std::ostringstream ss;
    ss << std::boolalpha << b;
    return ss.str();
}

inline void OpenAI::stream(const std::string &suffix, const Json &json,
                           const ChatStreamCallbacks &cb) {
    // Prepare request
    const auto body = json.dump();
    const auto complete_url = base_url + suffix;
    session_.setUrl(complete_url);
    session_.setBody(body);

    std::atomic<StreamControl> control{StreamControl::Continue};
    bool error_signalled = false;
    bool saw_done = false;

    auto on_event = [&](const _detail::SseEvent &ev) {
        static constexpr std::string_view done_token = "[DONE]";
        if (ev.data == done_token) {
            saw_done = true;
            if (cb.on_done) cb.on_done();
            return;
        }
        StreamControl decision = StreamControl::Continue;
        try {
            auto parsed = Json::parse(ev.data);
            if (cb.on_data) {
                decision = cb.on_data(parsed);
            }
        } catch (const std::exception &e) {
            decision = StreamControl::Stop;
            if (cb.on_error && !error_signalled) {
                cb.on_error(e.what());
                error_signalled = true;
            }
        }
        if (cb.control) {
            auto ext = cb.control();
            if (ext != StreamControl::Continue) {
                decision = ext;
            }
        }
        if (decision == StreamControl::Pause) {
            decision = StreamControl::Continue; // No safe pause/resume without external driver
        }
        control.store(decision);
    };

    _detail::SseParser parser(on_event);

    auto handler = [&](std::string_view raw) -> StreamControl {
        // External control hook takes priority; if it wants to pause/stop we
        // honor it without touching parser state.
        if (cb.control) {
            auto ext = cb.control();
            if (ext != StreamControl::Continue) {
                control.store(ext);
                return ext == StreamControl::Pause ? StreamControl::Continue : ext;
            }
        }

        auto current = control.load();
        if (current == StreamControl::Pause) {
            control.store(StreamControl::Continue);
            current = StreamControl::Continue;
        }
        if (current == StreamControl::Stop) {
            return StreamControl::Stop;
        }

        try {
            if (!parser.feed(raw, [&] { return control.load() == StreamControl::Stop; })) {
                control.store(StreamControl::Stop);
                return StreamControl::Stop;
            }
        } catch (const std::exception &e) {
            control.store(StreamControl::Stop);
            if (cb.on_error && !error_signalled) {
                cb.on_error(e.what());
                error_signalled = true;
            }
            return StreamControl::Stop;
        }

        // If on_data set Pause, make it transient unless an external control
        // callback exists to explicitly manage the paused state.
        current = control.load();
        if (current == StreamControl::Pause) {
            control.store(StreamControl::Continue);
            current = StreamControl::Continue;
        }
        return current;
    };

    try {
        const auto resp = session_.streamRequest("application/json", handler);
        const bool stopped_by_user =
            !resp.is_error && !error_signalled &&
            control.load() == StreamControl::Stop;

        if (resp.is_error && cb.on_error && !error_signalled) {
            cb.on_error(resp.error_message);
            error_signalled = true;
        } else if (stopped_by_user) {
            if (cb.on_done) cb.on_done();
            saw_done = true;
        } else if (!saw_done && cb.on_done &&
                   control.load() == StreamControl::Continue) {
            cb.on_done();
        }
    } catch (const std::exception &e) {
        if (cb.on_error) {
            cb.on_error(e.what());
        } else {
            throw;
        }
    }
}

inline OpenAI& start(const std::string& token = "", const std::string& organization = "", bool throw_exception = true, const std::string& api_base_url = "")  {
    static OpenAI instance{token, organization, throw_exception, api_base_url};
    return instance;
}

inline OpenAI& instance() {
    return start();
}

inline Json post(const std::string& suffix, const Json& json) {
    return instance().post(suffix, json);
}

inline Json get(const std::string& suffix/*, const Json& json*/) {
    return instance().get(suffix);
}

// Helper functions to get category structures instance()

inline CategoryModel& model() {
    return instance().model;
}

inline CategoryAssistants& assistant() {
    return instance().assistant;
}

inline CategoryThreads& thread() {
    return instance().thread;
}

inline CategoryCompletion& completion() {
    return instance().completion;
}

inline CategoryChat& chat() {
    return instance().chat;
}

inline CategoryAudio& audio() {
    return instance().audio;
}

inline CategoryEdit& edit() {
    return instance().edit;
}

inline CategoryImage& image() {
    return instance().image;
}

inline CategoryEmbedding& embedding() {
    return instance().embedding;
}

inline CategoryFile& file() {
    return instance().file;
}

inline CategoryFineTune& fineTune() {
    return instance().fine_tune;
}

inline CategoryModeration& moderation() {
    return instance().moderation;
}

// Definitions of category methods

// GET https://api.openai.com/v1/models
// Lists the currently available models, and provides basic information about each one such as the owner and availability.
inline Json CategoryModel::list() {
    return openai_.get("models");
}

// GET https://api.openai.com/v1/models/{model}
// Retrieves a model instance, providing basic information about the model such as the owner and permissioning.
inline Json CategoryModel::retrieve(const std::string& model) {
    return openai_.get("models/" + model);
}

// POST https://api.openai.com/v1/assistants 
// Create an assistant with a model and instructions.
inline Json CategoryAssistants::create(Json input) {
    return openai_.post("assistants", input);
}

// GET https://api.openai.com/v1/assistants/{assistant_id}
// Retrieves an assistant.
inline Json CategoryAssistants::retrieve(const std::string& assistants) {
    return openai_.get("assistants/" + assistants);
}

// POST https://api.openai.com/v1/assistants/{assistant_id}
// Modifies an assistant.
inline Json CategoryAssistants::modify(const std::string& assistants, Json input) {
    return openai_.post("assistants/" + assistants, input);
}

// DELETE https://api.openai.com/v1/assistants/{assistant_id}
// Delete an assistant.
inline Json CategoryAssistants::del(const std::string& assistants) {
    return openai_.del("assistants/" + assistants);
}

// GET https://api.openai.com/v1/assistants
// Returns a list of assistants.
inline Json CategoryAssistants::list() {
    return openai_.get("assistants");
}

// POST https://api.openai.com/v1/assistants/{assistant_id}/files
// Create an assistant file by attaching a File to an assistant.
inline Json CategoryAssistants::createFile(const std::string& assistants, Json input) {
    return openai_.post("assistants/" + assistants + "/files", input);
}

// GET https://api.openai.com/v1/assistants/{assistant_id}/files/{file_id}
// Retrieves an AssistantFile.
inline Json CategoryAssistants::retrieveFile(const std::string& assistants, const std::string& files) {
    return openai_.get("assistants/" + assistants + "/files/" + files);
}

// DELETE https://api.openai.com/v1/assistants/{assistant_id}/files/{file_id}
// Delete an assistant file.
inline Json CategoryAssistants::delFile(const std::string& assistants, const std::string& files) {
    return openai_.del("assistants/" + assistants + "/files/" + files);
}

// GET https://api.openai.com/v1/assistants/{assistant_id}/files
// Returns a list of assistant files.
inline Json CategoryAssistants::listFile(const std::string& assistants) {
    return openai_.get("assistants/" + assistants + "/files");
}

// POST https://api.openai.com/v1/threads
// Create a thread.
inline Json CategoryThreads::create() {
    Json input;
    return openai_.post("threads", input);
}

// GET https://api.openai.com/v1/threads/{thread_id}
// Retrieves a thread.
inline Json CategoryThreads::retrieve(const std::string& threads) {
    return openai_.get("threads/" + threads);
}

// POST https://api.openai.com/v1/threads/{thread_id}
// Modifies a thread.
inline Json CategoryThreads::modify(const std::string& threads, Json input) {
    return openai_.post("threads/" + threads, input);
}

// DELETE https://api.openai.com/v1/threads/{thread_id}
// Delete a thread.
inline Json CategoryThreads::del(const std::string& threads) {
    return openai_.del("threads/" + threads);
}

// POST https://api.openai.com/v1/threads/{thread_id}/messages
// Create a message.
inline Json CategoryThreads::createMessage(const std::string& threads, Json input) {
    return openai_.post("threads/" + threads + "/messages", input);
}

// GET https://api.openai.com/v1/threads/{thread_id}/messages/{message_id}
// Retrieve a message.
inline Json CategoryThreads::retrieveMessage(const std::string& threads, const std::string& messages) {
    return openai_.get("threads/" + threads + "/messages/" + messages);
}

// POST https://api.openai.com/v1/threads/{thread_id}/messages/{message_id}
// Modifies a message.
inline Json CategoryThreads::modifyMessage(const std::string& threads, const std::string& messages, Json input) {
    return openai_.post("threads/" + threads + "/messages/" + messages, input);
}

// GET https://api.openai.com/v1/threads/{thread_id}/messages
// Returns a list of messages for a given thread.
inline Json CategoryThreads::listMessage(const std::string& threads) {
    return openai_.get("threads/" + threads + "/messages");
}

// GET https://api.openai.com/v1/threads/{thread_id}/messages/{message_id}/files/{file_id}
// Retrieves a message file.
inline Json CategoryThreads::retrieveMessageFile(const std::string& threads, const std::string& messages, const std::string& files) {
    return openai_.get("threads/" + threads + "/messages/" + messages + "/files/" + files);
}

// GET https://api.openai.com/v1/threads/{thread_id}/messages/{message_id}/files
// Returns a list of message files.
inline Json CategoryThreads::listMessageFile(const std::string& threads, const std::string& messages) {
    return openai_.get("threads/" + threads + "/messages/" + messages + "/files");
}

// POST https://api.openai.com/v1/threads/{thread_id}/runs
// Create a run.
inline Json CategoryThreads::createRun(const std::string& threads, Json input) {
    return openai_.post("threads/" + threads + "/runs", input);
}

// GET https://api.openai.com/v1/threads/{thread_id}/runs/{run_id}
// Retrieves a run.
inline Json CategoryThreads::retrieveRun(const std::string& threads, const std::string& runs) {
    return openai_.get("threads/" + threads + "/runs/" + runs);
}

// POST https://api.openai.com/v1/threads/{thread_id}/runs/{run_id}
// Modifies a run.
inline Json CategoryThreads::modifyRun(const std::string& threads, const std::string& runs, Json input) {
    return openai_.post("threads/" + threads + "/runs/" + runs, input);
}

// GET https://api.openai.com/v1/threads/{thread_id}/runs
// Returns a list of runs belonging to a thread.
inline Json CategoryThreads::listRun(const std::string& threads) {
    return openai_.get("threads/" + threads + "/runs");
}

// POST https://api.openai.com/v1/threads/{thread_id}/runs/{run_id}/submit_tool_outputs
// When a run has the status: "requires_action" and required_action.type is submit_tool_outputs, this endpoint can be used to submit the outputs from the tool calls once they're all completed. All outputs must be submitted in a single request.
inline Json CategoryThreads::submitToolOutputsToRun(const std::string& threads, const std::string& runs, Json input) {
    return openai_.post("threads/" + threads + "/runs/" + runs + "submit_tool_outputs", input);
}

// POST https://api.openai.com/v1/threads/{thread_id}/runs/{run_id}/cancel
// Cancels a run that is in_progress.
inline Json CategoryThreads::cancelRun(const std::string& threads, const std::string& runs) {
    Json input;
    return openai_.post("threads/" + threads + "/runs/" + runs + "/cancel", input);
}

// POST https://api.openai.com/v1/threads/runs
// Create a thread and run it in one request.
inline Json CategoryThreads::createThreadAndRun(Json input) {
    return openai_.post("threads/runs", input);
}

// GET https://api.openai.com/v1/threads/{thread_id}/runs/{run_id}/steps/{step_id}
// Retrieves a run step.
inline Json CategoryThreads::retrieveRunStep(const std::string& threads, const std::string& runs, const std::string& steps) {
    return openai_.get("threads/" + threads + "/runs/" + runs + "/steps/" + steps);
}

// GET https://api.openai.com/v1/threads/{thread_id}/runs/{run_id}/steps
// Returns a list of run steps belonging to a run.
inline Json CategoryThreads::listRunStep(const std::string& threads, const std::string& runs) {
    return openai_.get("threads/" + threads + "/runs/" + runs + "/steps");
}

// POST https://api.openai.com/v1/completions
// Creates a completion for the provided prompt and parameters
inline Json CategoryCompletion::create(Json input) {
    return openai_.post("completions", input);
}

inline void CategoryCompletion::stream(Json input,
                                       const ChatStreamCallbacks &cb) {
    if (!input.contains("stream")) {
        input["stream"] = true;
    }
    openai_.stream("completions", std::move(input), cb);
}

// POST https://api.openai.com/v1/chat/completions
// Creates a chat completion for the provided prompt and parameters
inline Json CategoryChat::create(Json input) {
    return openai_.post("chat/completions", input);
}

inline void CategoryChat::stream(Json input, const StreamCallbacks &cb) {
    if (!input.contains("stream")) {
        input["stream"] = true;
    }
    openai_.stream("chat/completions", input, cb);
}

// POST https://api.openai.com/v1/audio/transcriptions
// Transcribes audio into the input language.
inline Json CategoryAudio::transcribe(Json input) {
    auto lambda = [input]() -> std::map<std::string, std::string> {
        std::map<std::string, std::string> temp;
        temp.insert({"model", input["model"].get<std::string>()});
        if (input.contains("language")) {
            temp.insert({"language", input["language"].get<std::string>()});
        }
        if (input.contains("prompt")) {
            temp.insert({"prompt", input["prompt"].get<std::string>()});
        }
        if (input.contains("response_format")) {
            temp.insert({"response_format", input["response_format"].get<std::string>()});
        }
        if (input.contains("temperature")) {
            temp.insert({"temperature", std::to_string(input["temperature"].get<float>())});
        }
        return temp;
    };
    openai_.setMultiformPart({"file", input["file"].get<std::string>()}, 
        lambda()
    );

    return openai_.post("audio/transcriptions", std::string{""}, "multipart/form-data"); 
}

// POST https://api.openai.com/v1/audio/translations
// Translates audio into into English..
inline Json CategoryAudio::translate(Json input) {
    auto lambda = [input]() -> std::map<std::string, std::string> {
        std::map<std::string, std::string> temp;
        temp.insert({"model", input["model"].get<std::string>()});
        if (input.contains("language")) {
            temp.insert({"language", input["language"].get<std::string>()});
        }
        if (input.contains("prompt")) {
            temp.insert({"prompt", input["prompt"].get<std::string>()});
        }
        if (input.contains("response_format")) {
            temp.insert({"response_format", input["response_format"].get<std::string>()});
        }
        if (input.contains("temperature")) {
            temp.insert({"temperature", std::to_string(input["temperature"].get<float>())});
        }
        return temp;
    };
    openai_.setMultiformPart({"file", input["file"].get<std::string>()}, 
        lambda()
    );

    return openai_.post("audio/translations", std::string{""}, "multipart/form-data"); 
}

// POST https://api.openai.com/v1/translations
// Creates a new edit for the provided input, instruction, and parameters
inline Json CategoryEdit::create(Json input) {
    return openai_.post("edits", input);
}

// POST https://api.openai.com/v1/images/generations
// Given a prompt and/or an input image, the model will generate a new image.
inline Json CategoryImage::create(Json input) {
    return openai_.post("images/generations", input);
}

// POST https://api.openai.com/v1/images/edits
// Creates an edited or extended image given an original image and a prompt.
inline Json CategoryImage::edit(Json input) {
    std::string prompt = input["prompt"].get<std::string>(); // required
    // Default values
    std::string mask = "";
    int n = 1;
    std::string size = "1024x1024";
    std::string response_format = "url";
    std::string user = "";
    
    if (input.contains("mask")) {
        mask = input["mask"].get<std::string>();
    }
    if (input.contains("n")) {
        n = input["n"].get<int>();
    }
    if (input.contains("size")) {
        size = input["size"].get<std::string>();
    }
    if (input.contains("response_format")) {
        response_format = input["response_format"].get<std::string>();
    }
    if (input.contains("user")) {
        user = input["user"].get<std::string>();
    }
    openai_.setMultiformPart({"image",input["image"].get<std::string>()}, 
        std::map<std::string, std::string>{
            {"prompt", prompt},
            {"mask", mask},
            {"n", std::to_string(n)},
            {"size", size},
            {"response_format", response_format},
            {"user", user}
        }
    );

    return openai_.post("images/edits", std::string{""}, "multipart/form-data"); 
}

// POST https://api.openai.com/v1/images/variations
// Creates a variation of a given image.
inline Json CategoryImage::variation(Json input) {
    // Default values
    int n = 1;
    std::string size = "1024x1024";
    std::string response_format = "url";
    std::string user = "";
    
    if (input.contains("n")) {
        n = input["n"].get<int>();
    }
    if (input.contains("size")) {
        size = input["size"].get<std::string>();
    }
    if (input.contains("response_format")) {
        response_format = input["response_format"].get<std::string>();
    }
    if (input.contains("user")) {
        user = input["user"].get<std::string>();
    }
    openai_.setMultiformPart({"image",input["image"].get<std::string>()}, 
        std::map<std::string, std::string>{
            {"n", std::to_string(n)},
            {"size", size},
            {"response_format", response_format},
            {"user", user}
        }
    );

    return openai_.post("images/variations", std::string{""}, "multipart/form-data"); 
}

inline Json CategoryEmbedding::create(Json input) { 
    return openai_.post("embeddings", input); 
}

inline Json CategoryFile::list() { 
    return openai_.get("files"); 
}

inline Json CategoryFile::upload(Json input) {
    openai_.setMultiformPart({"file", input["file"].get<std::string>()}, 
        std::map<std::string, std::string>{{"purpose", input["purpose"].get<std::string>()}}
    );

    return openai_.post("files", std::string{""}, "multipart/form-data"); 
}

inline Json CategoryFile::del(const std::string& file_id) { 
    return openai_.del("files/" + file_id); 
}

inline Json CategoryFile::retrieve(const std::string& file_id) { 
    return openai_.get("files/" + file_id); 
}

inline Json CategoryFile::content(const std::string& file_id) { 
    return openai_.get("files/" + file_id + "/content"); 
}

inline Json CategoryFineTune::create(Json input) { 
    return openai_.post("fine-tunes", input); 
}

inline Json CategoryFineTune::list() { 
    return openai_.get("fine-tunes"); 
}

inline Json CategoryFineTune::retrieve(const std::string& fine_tune_id) { 
    return openai_.get("fine-tunes/" + fine_tune_id); 
}

inline Json CategoryFineTune::content(const std::string& fine_tune_id) { 
    return openai_.get("fine-tunes/" + fine_tune_id + "/content"); 
}

inline Json CategoryFineTune::cancel(const std::string& fine_tune_id) { 
    return openai_.post("fine-tunes/" + fine_tune_id + "/cancel", Json{}); 
}

inline Json CategoryFineTune::events(const std::string& fine_tune_id) { 
    return openai_.get("fine-tunes/" + fine_tune_id + "/events"); 
}

inline Json CategoryFineTune::del(const std::string& model) { 
    return openai_.del("models/" + model); 
}

inline Json CategoryModeration::create(Json input) { 
    return openai_.post("moderations", input); 
}

} // namespace _detail

// Public interface

using _detail::OpenAI;

// instance
using _detail::start;
using _detail::instance;

// Generic methods
using _detail::post;
using _detail::get;

// Helper categories access
using _detail::model;
using _detail::assistant;
using _detail::thread;
using _detail::completion;
using _detail::CategoryCompletion;
using _detail::edit;
using _detail::image;
using _detail::embedding;
using _detail::file;
using _detail::fineTune;
using _detail::moderation;
using _detail::chat;
using _detail::CategoryChat;
using _detail::audio;
using _detail::ChatStreamCallbacks;
using _detail::StreamControl;

using _detail::Json;

} // namespace openai

#endif // OPENAI_HPP_
