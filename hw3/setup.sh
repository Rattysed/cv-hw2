#!/bin/bash

# Setup script - первичная настройка проекта

echo "=========================================="
echo "DETR Object Detection - Setup"
echo "=========================================="
echo ""

# Проверка Python
if ! command -v python3.10 &> /dev/null; then
    echo "❌ Python 3 не найден! Установите Python 3.8+"
    exit 1
fi

PYTHON_VERSION=$(python3.10 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✓ Python версия: $PYTHON_VERSION"

# Проверка GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU обнаружена:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠ NVIDIA GPU не обнаружена. Обучение будет на CPU (медленно!)"
fi

echo ""

if [ -d "venv" ]; then
    echo "⚠ venv уже существует. Пересоздать? (y/n)"
    read -r response
    if [ "$response" = "y" ]; then
        rm -rf venv
        python3 -m venv venv
    fi
else
    python3 -m venv venv
fi

echo "✓ Виртуальное окружение создано"

# Активация
source venv/bin/activate
echo "✓ Виртуальное окружение активировано"

echo ""
echo "=========================================="
echo "Установка зависимостей"
echo "=========================================="
echo ""

# Обновление pip
pip3 install --upgrade pip

# Установка PyTorch (с CUDA если доступна)
if command -v nvidia-smi &> /dev/null; then
    echo "Установка PyTorch с CUDA поддержкой..."
    pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
else
    echo "Установка PyTorch (CPU версия)..."
    pip3 install torch torchvision
fi

# Установка остальных зависимостей
echo ""
echo "Установка остальных пакетов..."
pip3 install -r requirements.txt

echo ""
echo "✓ Все зависимости установлены"


echo ""
echo "Проверка установки"
echo ""

python3 << EOF
import sys
print("Python:", sys.version)

try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"  CUDA доступна: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA версия: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"❌ PyTorch: {e}")

try:
    import transformers
    print(f"✓ Transformers: {transformers.__version__}")
except ImportError as e:
    print(f"❌ Transformers: {e}")

try:
    import diffusers
    print(f"✓ Diffusers: {diffusers.__version__}")
except ImportError as e:
    print(f"❌ Diffusers: {e}")

try:
    from pycocotools import coco
    print(f"✓ pycocotools установлен")
except ImportError as e:
    print(f"❌ pycocotools: {e}")

try:
    import tensorboard
    print(f"✓ TensorBoard установлен")
except ImportError as e:
    print(f"❌ TensorBoard: {e}")
EOF
