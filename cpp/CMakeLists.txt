cmake_minimum_required(VERSION 3.10)
project(cifar_infer)

set(CMAKE_CXX_STANDARD 17)

set(ONNXRUNTIME_DIR /opt/onnxruntime)

include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib)

add_executable(cifar_infer
    main.cpp
    inference_engine.cpp
)

target_link_libraries(cifar_infer onnxruntime)
