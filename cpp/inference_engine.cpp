#include "inference_engine.hpp"
#include <iostream>

InferenceEngine::InferenceEngine(const std::string& model_path)
    : env(ORT_LOGGING_LEVEL_WARNING, "cifar"),
      session(nullptr)
{
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

    session = Ort::Session(env, model_path.c_str(), session_options);
}

std::vector<float> InferenceEngine::run(const std::vector<float>& input)
{
    // Input shape: [1, 3, 32, 32]
    std::vector<int64_t> input_shape = {1, 3, 32, 32};

    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float*>(input.data()),
        input.size(),
        input_shape.data(),
        input_shape.size()
    );

    const char* input_names[]  = {"input"};
    const char* output_names[] = {"output"};

    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_names,
        &input_tensor,
        1,
        output_names,
        1
    );

    float* output_data = output_tensors[0].GetTensorMutableData<float>();
    return std::vector<float>(output_data, output_data + 10);
}
