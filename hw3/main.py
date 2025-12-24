import os
import random

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

from collections import Counter
from functools import partial

from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, f1_score
from PIL import Image as PILImage

from datasets import load_dataset, concatenate_datasets, Dataset, Features, Value, Image, ClassLabel
from transformers import (
    ViTConfig,
    ViTForImageClassification,
    AutoImageProcessor,
    TrainingArguments,
    Trainer
)
from diffusers import AutoPipelineForText2Image

from tqdm.auto import tqdm
from IPython.display import display

SEED = 42
NUM_PROC=32
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def show_distribution(ds, title="Распределение классов"):
    labels_train = [ex["label"] for ex in ds]
    label_counts = Counter(labels_train)

    id2label = ds.features["label"].int2str
    label_names = [id2label(i) for i in sorted(label_counts.keys())]
    counts = [label_counts[i] for i in sorted(label_counts.keys())]

    plt.figure(figsize=(8,4))
    plt.bar(label_names, counts)
    plt.title(title)
    plt.xlabel("Класс")
    plt.ylabel("Количество образцов")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def show_some_examples(ds, n=8, seed=42, suptitle="Примеры элементов датасета", id2label=None):
    plt.figure(figsize=(12,4))
    
    for i, example in enumerate(ds.shuffle(seed=seed).select(range(n))):
        plt.subplot(1, n, i+1)
        plt.imshow(example['img'])
        plt.axis("off")
        if (id2label):
            plt.title(id2label[example['label']])
    plt.suptitle(suptitle)
    plt.tight_layout()
    plt.show()

dataset_name = "cifar10"
ds = load_dataset(dataset_name)


print(ds)
print(ds["train"][0])

# input("тык")

train, test = ds['train'], ds['test']
num_labels = ds["train"].features["label"].num_classes
id2label = {i: ds["train"].features["label"].int2str(i) for i in range(num_labels)}
label2id = {v: k for k, v in id2label.items()}

show_distribution(train, title="Распределение классов в train")


def make_it_worse(ds, p: list[float]):
    return ds.filter(lambda item: bool(np.random.binomial(n=1, p=p[item['label']])), num_proc=NUM_PROC)

p = [i / 10 for i in range(1, 11)]
train = make_it_worse(train, random.sample(p, k=len(p)))

show_distribution(train, title='Распределение классов в "сломанном" train')



def preprocess(transform, examples):
    images = [transform(image=np.array(img.convert('RGB')))['image'] for img in examples['img']]
    return {
        "pixel_values": torch.stack(images),
        "labels": torch.tensor(list(examples['label']), dtype=torch.long)
    }


train_transform = A.Compose(
    [
        A.Resize(224, 224),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),
        A.Rotate(limit=10, p=0.5),
        A.GaussianBlur(p=0.5),
        A.GaussNoise(p=0.5),
        A.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        ),
        ToTensorV2(),
    ],
)

train_preprocess = partial(preprocess, train_transform)

test_transform = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        ),
        ToTensorV2(),
    ],
)

test_preprocess = partial(preprocess, test_transform)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1_macro = f1_score(labels, preds, average="macro")
    return {"accuracy": acc, "f1_macro": f1_macro}

def train_pipeline(path: str, train, valid):
    config = ViTConfig(
        image_size=224,
        patch_size=16,
        num_channels=3,
        num_labels=num_labels,
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=2,
        intermediate_size=256,
        id2label=id2label,
        label2id=label2id,
    )
    model = ViTForImageClassification(config)
    
    training_args = TrainingArguments(
        output_dir=f"./{path}",
        per_device_train_batch_size=256,
        per_device_eval_batch_size=256,
        num_train_epochs=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        report_to=["tensorboard"],
        logging_dir=f"./runs/{path}",
        logging_strategy="steps",
        logging_steps=100,
        remove_unused_columns=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train.with_transform(train_preprocess),
        eval_dataset=valid.with_transform(test_preprocess),
        compute_metrics=compute_metrics
    )
    
    trainer.train()
    
    return model, trainer


vit, trainer = train_pipeline('vit_cifar10_baseline', train, valid=test)
print("Тут был трейн")

# АУГМЕНТАЦИЯ

label_counts = Counter([example["label"] for example in train])
max_count = max(label_counts.values())

underrepresented_labels = [
    (label, max_count - count)
    for label, count in label_counts.items() if count < max_count
]

def build_prompt(label_id: int, label_name: str):
    return f"high quality photo of {label_name}, ultra detailed, realistic"

sdxl = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sdxl-turbo",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")


synthetic_images = []
synthetic_labels = []

generator = torch.Generator(device="cuda").manual_seed(SEED)

for label_id, need in underrepresented_labels:
    continue
    label_name = id2label[label_id]
    prompt = build_prompt(label_id, label_name)
    
    print(f"Генерируем {need} изображений для класса '{label_name}'")
    for i in tqdm(range(need)):
        image = sdxl(prompt, num_inference_steps=20, guidance_scale=7.5, generator=generator).images[0]
        image = image.resize((32, 32), PILImage.BICUBIC)

        synthetic_images.append(image)
        synthetic_labels.append(label_id)

synthetic_train = Dataset.from_dict({ 'img': synthetic_images, 'label': synthetic_labels })

show_some_examples(synthetic_train, seed=123, suptitle="Примеры элементов синтетического датасета", id2label=id2label)

casted_synthetic_train = synthetic_train.cast(Features({
    "img": Image(mode=None, decode=True),
    "label": ClassLabel(names=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']),
}))

train_with_synthetic = concatenate_datasets([train, casted_synthetic_train])
show_distribution(train_with_synthetic, title='Распределение классов в "синтетическом" train')

train_with_synthetic.save_to_disk("./data/synthetic_cifar10")

vit_synth, trainer_synth = train_pipeline('vit_cifar10_synth', train_with_synthetic, valid=test)

metrics_baseline = trainer.evaluate(test.with_transform(test_preprocess))

metrics_synth = trainer_synth.evaluate(test.with_transform(test_preprocess))


results = {
    "Setting": ["Baseline", "With Synthetic Data"],
    "Test Accuracy": [
        metrics_baseline["eval_accuracy"],
        metrics_synth["eval_accuracy"]
    ],
    "Test F1 (macro)": [
        metrics_baseline["eval_f1_macro"],
        metrics_synth["eval_f1_macro"]
    ]
}

df_ablation = pd.DataFrame(results)
print(df_ablation)

# display(df_ablation.style.format({
#     "Test Accuracy": "{:.4f}",
#     "Test F1 (macro)": "{:.4f}"
# }))