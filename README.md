cat << 'EOF' > ~/ov_reference/README.md
# OpenVINO GenAI Reference Runner (Android ARM64)

Minimal C++ reference runner for evaluating offline INT4-quantized LLMs on Android ARM64 devices via Intel OpenVINO GenAI runtime.

## Supported Models
- `qwen3-0.6b-int4-ov`
- `llama-3.2-1b-int4-ov`

## Prerequisites
- Android NDK (r28c+)
- Android device (ARM64-v8a, USB debugging enabled)
- Prebuilt OpenVINO GenAI Android libraries (`arm64-v8a`)

## Build
```bash
cmake -B build -S . \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-35 \
  -DANDROID_STL=c++_shared

cmake --build build -j$(nproc)
```
