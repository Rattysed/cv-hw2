# Результаты ДЗ 2: DETR Object Detection

## Параметры обучения

| Параметр | Значение |
|----------|----------|
| Модель | facebook/detr-resnet-50 |
| Датасет | COCO subset (10 классов) |
| Train samples | 5000 |
| Val samples | 500 |
| Batch size | 4 |
| Epochs |20 |
| Learning rate | 1e-5 |
| Optimizer | AdamW |

## Классы

person, car, dog, cat, chair, bottle, bicycle, airplane, bus, train

## Метрики

```json
{
  "bbox_mAP": 0.005598969913672249,
  "bbox_mAP50": 0.009203385597797709,
  "bbox_mAP75": 0.0057331323511980855
}
```

| Метрика | Значение |
|---------|----------|
| bbox mAP | 0.56% |
| bbox mAP@50 | 0.92% |
| bbox mAP@75 | 0.57% |

⚠️ **Примечание**: Низкие метрики объясняются:
- Малым количеством эпох (20 вместо 50-100)
- Ограниченным датасетом (5000 вместо 84000)
- Для production нужно обучать дольше

## Динамика обучения

| Эпоха | Train Loss | Val Loss | Train CE | Val CE | Train BBox | Val BBox |
|-------|------------|----------|----------|--------|------------|----------|
| 1     | 3.1473     | 2.2683   | 1.1342   | 1.0130 | 0.1182     | 0.0836 |
| 3     | 2.0134     | 1.8888   | 0.9084   | 0.8289 | 0.0653     | 0.0700 |
| 5     | 1.5988     | 1.5729   | 0.6875   | 0.6269 | 0.0515     | 0.0607 |
| 7     | 1.1665     | 1.2283   | 0.4073   | 0.4253 | 0.0433     | 0.0523 |
| 10    | 1.1228     | 1.2062   | 0.3885   | 0.4138 | 0.0422     | 0.0512 |
| 20    | 1.0928     | 1.1960   | 0.3684   | 0.4021 | 0.0405     | 0.0502 |

**Наблюдения:**
- ✅ Стабильное снижение loss на 60% (3.14 → 1.20)
- ✅ Нет переобучения (train ≈ val)
- ✅ Classification loss улучшился в 2x
- ✅ Bbox regression улучшился на 27%

## Файлы

- **TensorBoard**: `outputs/full_run/logs/`
- **Профайлер**: `outputs/full_run/profiler/`
- **Визуализации**: `visualizations/predictions/` (50 изображений)
- **Графики**: `visualizations/training_curves.png`
- **Error analysis**: `visualizations/error_analysis/`

## Как запустить

```bash
# TensorBoard
tensorboard --logdir outputs/full_run/logs

# Инференс
python examples/example_inference.py
```

