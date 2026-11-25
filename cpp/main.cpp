#include "inference_engine.hpp"
#include <iostream>
#include <vector>
#include <chrono>

int main() {
    // Load ONNX model
    InferenceEngine engine("../models/cifar_tinynet.onnx");

    // Dummy input: 1 x 3 x 32 x 32
    std::vector<float> input(3 * 32 * 32, 0.5f);

    // -------------------------------
    // Warm-up (important!)
    // -------------------------------
    for (int i = 0; i < 50; i++) {
        engine.run(input);
    }

    // -------------------------------
    // Steady-state measurement
    // -------------------------------
    const int runs = 1000;
    double total_ms = 0.0;

    for (int i = 0; i < runs; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        engine.run(input);
        auto end = std::chrono::high_resolution_clock::now();

        total_ms += std::chrono::duration<double, std::milli>(end - start).count();
    }

    std::cout << "C++ Avg Latency (steady-state): "
              << (total_ms / runs) << " ms" << std::endl;

    return 0;
}
