cmake_minimum_required(VERSION 3.22)
project(ov_reference_runner CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OPENVINO_INSTALL_DIR "/home/sannarat/openvino-android/openvino-install")
set(JNILIBS_DIR "/home/sannarat/openvino-android/jniLibs/arm64-v8a")

add_executable(ov_reference_runner
        src/main.cpp
)

target_include_directories(ov_reference_runner PRIVATE
        ${OPENVINO_INSTALL_DIR}/runtime/include
)

target_link_directories(ov_reference_runner PRIVATE
        ${JNILIBS_DIR}
)

target_link_libraries(ov_reference_runner PRIVATE
        openvino_genai
        openvino
        openvino_tokenizers
        tbb
        c++_shared
)