import os
import json
import random
import subprocess

import numpy as np
import torch
import matplotlib.pyplot as plt

from datasets import load_from_disk

from transformers import (
    DebertaV2Tokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error
)

from scipy.stats import (
    pearsonr,
    spearmanr
)

# =====================================================
# Config
# =====================================================

MODEL_NAME = "DeBERTa-v3-small"

MODEL_PATH = (
    "./LLMs/DeBERTa-v3-small"
)

DATASET_PATH = (
    "./hf_ReTraC"
)

OUTPUT_DIR = (
    "./"
)

MAX_LENGTH = 512
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5
WEIGHT_DECAY = 0.01
SEED = 42

# =====================================================
# Create Output Dir
# =====================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# Matplotlib Style
# =====================================================

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# =====================================================
# Seed
# =====================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# =====================================================
# Auto GPU Selection
# =====================================================

def get_free_gpu():
    if not torch.cuda.is_available():
        return torch.device("cpu")

    try:
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,nounits,noheader"
            ]
        )

        memory_free = result.decode("utf-8").strip().split("\n")
        memory_free = [int(x) for x in memory_free]

        best_gpu = memory_free.index(max(memory_free))

        print("\n========== GPU INFO ==========")
        print(f"GPU Free Memory: {memory_free}")
        print(f"Using GPU {best_gpu}")
        print("================================\n")

        return best_gpu

    except Exception as e:
        print(f"[WARNING] GPU selection failed: {e}")
        return 0

# =====================================================
# GPU
# =====================================================

gpu_id = get_free_gpu()

device = torch.device(
    f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
)

# =====================================================
# Load Dataset
# =====================================================

print("Loading dataset...")

dataset = load_from_disk(DATASET_PATH)

train_dataset = dataset["train"]
valid_dataset = dataset["validation"]
test_dataset = dataset["test"]

print(f"Train Size: {len(train_dataset)}")
print(f"Valid Size: {len(valid_dataset)}")
print(f"Test Size: {len(test_dataset)}")

# =====================================================
# Load Tokenizer
# =====================================================

print("\nLoading tokenizer...")

tokenizer = DebertaV2Tokenizer.from_pretrained(MODEL_PATH)

# =====================================================
# Build Input
# =====================================================

def build_input(example):

    text = (
        "Question:\n"
        + example["question"]
        + "\n\n"
        + "Current Reasoning:\n"
        + example["prefix"]
    )

    tokenized = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH
    )

    tokenized["labels"] = float(example["recoverability"])

    return tokenized

# =====================================================
# Tokenize Dataset
# =====================================================

print("\nTokenizing dataset...")

train_dataset = train_dataset.map(build_input)
valid_dataset = valid_dataset.map(build_input)
test_dataset = test_dataset.map(build_input)

# =====================================================
# Data Collator
# =====================================================

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)

# =====================================================
# Load Model
# =====================================================

print("\nLoading model...")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH,
    num_labels=1,
    problem_type="regression"
)

model.to(device)

print(f"Model loaded on {device}")

# =====================================================
# Metrics
# =====================================================

def compute_metrics(eval_pred):

    predictions, labels = eval_pred

    predictions = predictions.squeeze(-1)

    mse = mean_squared_error(labels, predictions)

    mae = mean_absolute_error(labels, predictions)

    try:
        pearson_corr = pearsonr(labels, predictions)[0]
    except:
        pearson_corr = 0.0

    try:
        spearman_corr = spearmanr(labels, predictions)[0]
    except:
        spearman_corr = 0.0

    return {
        "mse": mse,
        "mae": mae,
        "pearson": pearson_corr,
        "spearman": spearman_corr
    }

# =====================================================
# Training Args
# =====================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,

    learning_rate=LEARNING_RATE,

    num_train_epochs=NUM_EPOCHS,

    weight_decay=WEIGHT_DECAY,

    logging_steps=20,

    eval_strategy="epoch",

    report_to="none",

    fp16=False
)

# =====================================================
# Trainer
# =====================================================

trainer = Trainer(
    model=model,

    args=training_args,

    train_dataset=train_dataset,
    eval_dataset=valid_dataset,

    tokenizer=tokenizer,

    data_collator=data_collator,

    compute_metrics=compute_metrics
)

# =====================================================
# Train
# =====================================================

print("\n========== START TRAINING ==========\n")

trainer.train()

# =====================================================
# Save Loss History
# =====================================================

log_history = trainer.state.log_history

train_loss = [
    x['loss']
    for x in log_history
    if 'loss' in x
]

eval_loss = [
    x['eval_loss']
    for x in log_history
    if 'eval_loss' in x
]

loss_history_path = os.path.join(
    OUTPUT_DIR,
    "loss_history.json"
)

with open(loss_history_path, "w", encoding="utf-8") as f:

    json.dump(
        {
            "train_loss": train_loss,
            "eval_loss": eval_loss
        },
        f,
        indent=2
    )

print(f"Loss values saved to {loss_history_path}")

# =====================================================
# Plot Loss Curve
# =====================================================

plt.figure(figsize=(8, 5))

# 训练损失曲线
plt.plot(
    train_loss,
    label='Train Loss',
    linewidth=2.8
)

# 如果想画验证集loss，可取消注释
# plt.plot(
#     np.linspace(
#         0,
#         len(train_loss) - 1,
#         len(eval_loss)
#     ),
#     eval_loss,
#     label='Validation Loss',
#     linewidth=2.8
# )

# 标题和坐标轴字体增大
plt.xlabel(
    'Logging Step',
    fontsize=18
)

plt.ylabel(
    'Loss',
    fontsize=18
)

plt.title(
    'Training Loss',
    fontsize=20
)

# 坐标轴刻度字体
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# 图例字体
plt.legend(fontsize=15)

# 网格
plt.grid(True)

# 自动紧凑布局
plt.tight_layout()

loss_plot_path = os.path.join(
    OUTPUT_DIR,
    "loss_curve.pdf"
)

# 去除额外留白
plt.savefig(
    loss_plot_path,
    bbox_inches='tight',
    pad_inches=0
)

plt.close()

print(f"Loss curve saved to {loss_plot_path}")

# =====================================================
# Final Evaluation
# =====================================================

print("\n========== TEST EVALUATION ==========\n")

metrics = trainer.evaluate(test_dataset)

for k, v in metrics.items():
    print(f"{k}: {v}")

# =====================================================
# Save Test Metrics
# =====================================================

metrics_path = os.path.join(
    OUTPUT_DIR,
    "test_metrics.json"
)

with open(metrics_path, "w", encoding="utf-8") as f:

    json.dump(
        metrics,
        f,
        indent=2
    )

print(f"Test metrics saved to {metrics_path}")

# =====================================================
# Save Model
# =====================================================

print("\nSaving model...")

trainer.save_model(OUTPUT_DIR)

tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\nModel Saved To: {OUTPUT_DIR}")