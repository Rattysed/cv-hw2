# Домашнее задание 2.5: Синтетические данные через Stable Diffusion + ControlNet
## Результаты экспериментов
* Логи tensorboard доступны в директории `runs`
* Синтетический датасет `synthetic_cifar10` доступен в Releases под тэгом `hw3`
* Чекпоинты обучения доступны в Releases под тэгом `hw3`

## Эксперименты
### Датасет
Во всех экспериментах использовался датасет `CIFAR10` с различными модификациями.  
Аугметации применялись на лету с использованием библиотеки `Albumentations`, применяемые аугментации:
  `HorizontalFlip(p=0.5)`
  `RandomBrightnessContrast(p=0.5)`
  `HueSaturationValue(p=0.5)`
  `Rotate(limit=10, p=0.5)`
  `GaussianBlur(p=0.5)`
  `GaussNoise(p=0.5)`

### Модель
Во всех экспериментах Обучению подвергался кастомный ViT со следующими гиперпараметрами:
```python
ViTConfig(
    image_size=224,
    patch_size=16,
    num_channels=3,
    num_labels=num_labels,
    hidden_size=128,
    num_hidden_layers=4,
    num_attention_heads=2,
    intermediate_size=256,
)
```

### Обучение
Во всех экспериментах обучение происходило в 10 эпох, валидация и логгирование происходили каждую эпоху.  
**Шедулер:** `LinearLR(learning_rate=1e-5)`  
**Оптимизатор:** `AdamW(weight_decay=1e-4)` 

**1. Обучение на неравномерном датасете**
Для эксперимента классы датасета CIFAR10 были произвольным образом уменены до `0.1, 0.2, ..., 0.9, 1.0` части исходного размера.

**2. Обучение на синтетически выровненном датасете**
Для эксперимента классы датасета CIFAR10 были выровнены с использованием модели `stabilityai/stabilityai/sdxl-turbo` из репозиториев HuggingFace.  
Примеры изображений из синтетического датасета (полный датасет `synthetic_cifar10` доступен в Releases под тэгом `hw3`): 
![Example of synthetic_cifar10](assets/synth_example.png)

### Результаты
|       EXP           | Test Accuracy | Test F1 (macro) |
|:-------------------:|:-------------:|:---------------:|
| Baseline            | 0.4701        | 0.4320          |
| With Synthetic Data | 0.5236        | 0.5631          |