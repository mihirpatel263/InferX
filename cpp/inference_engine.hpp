#pragma once
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>

class InferenceEngine {
public:
    explicit InferenceEngine(const std::string& model_path);
    std::vector<float> run(const std::vector<float>& input);

private:
    Ort::Env env;
    Ort::Session session;
};
