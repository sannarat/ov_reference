#!/usr/bin/env bash
set -euo pipefail

DEV_DIR="/data/local/tmp/ov_reference"
JNILIBS_PATH="/home/sannarat/openvino-android/jniLibs/arm64-v8a"
BUILD_BIN="./build/ov_reference_runner"

echo "==> Подготовка каталогов на Android..."
adb shell "mkdir -p ${DEV_DIR}/models"

echo "==> Отправка библиотек и бинарника..."
adb push ${JNILIBS_PATH}/* ${DEV_DIR}/
adb push ${BUILD_BIN} ${DEV_DIR}/
adb shell "chmod +x ${DEV_DIR}/ov_reference_runner"

echo "==> Отправка моделей на устройство..."
adb push ./models/qwen3-0.6b-int4-ov ${DEV_DIR}/models/
adb push ./models/llama-3.2-1b-int4-ov ${DEV_DIR}/models/

echo "==> Запуск 8 сценариев на CPU..."
adb shell "LD_LIBRARY_PATH=${DEV_DIR} ${DEV_DIR}/ov_reference_runner ${DEV_DIR}/models ${DEV_DIR}/results.json"

echo "==> Скачивание результатов на хост..."
mkdir -p ./reference_outputs
adb pull ${DEV_DIR}/results.json ./reference_outputs/results.json

echo "==> Проверка негативного теста (несуществующий путь)..."
set +e
adb shell "LD_LIBRARY_PATH=${DEV_DIR} ${DEV_DIR}/ov_reference_runner ${DEV_DIR}/nonexistent ${DEV_DIR}/err.json"
CODE=$?
set -e

if [ $CODE -ne 0 ]; then
    echo "Негативный тест пройден: получен ненулевой код ошибки ($CODE)."
else
    echo "Ошибка: раннер вернул код 0 на несуществующей директории!"
    exit 1
fi

echo "Выполнено успешно. Сформирован файл ./reference_outputs/results.json."