#!/usr/bin/env python3
import os, json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from typing import Dict
import inspect

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
import transformers as tf
from sklearn.metrics import (
    f1_score, accuracy_score, classification_report, confusion_matrix
)

# ===============================
#  Paths & Columns
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_CSV = os.path.join(BASE_DIR, "train_clean.csv")
VALID_CSV = os.path.join(BASE_DIR, "valid_clean.csv")
TEST_CSV  = os.path.join(BASE_DIR, "test_clean.csv")

TEXT_COL  = "clean_statement"
LABEL_COL = "label"

# ===============================
#  Load CSV
# ===============================
train_df = pd.read_csv(TRAIN_CSV)
valid_df = pd.read_csv(VALID_CSV)
test_df  = pd.read_csv(TEST_CSV)

labels = sorted(train_df[LABEL_COL].unique().tolist())
label2id: Dict[str,int] = {y:i for i,y in enumerate(labels)}
id2label: Dict[int,str] = {i:y for y,i in label2id.items()}
num_labels = len(labels)

train_df["label_id"] = train_df[LABEL_COL].map(label2id)
valid_df["label_id"] = valid_df[LABEL_COL].map(label2id)
test_df["label_id"]  = test_df[LABEL_COL].map(label2id)

# ===============================
#  Tokenizer + Dataset
# ===============================
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
MAX_LEN = 64

class TextClsDS(Dataset):
    def __init__(self, df):
        self.texts = df[TEXT_COL].astype(str).tolist()
        self.labels = df["label_id"].astype(int).tolist()
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN
        )
        enc["labels"] = self.labels[idx]
        return {k: torch.tensor(v) for k, v in enc.items()}

ds_train = TextClsDS(train_df)
ds_valid = TextClsDS(valid_df)
ds_test  = TextClsDS(test_df)

# ===============================
#  Class Weights
# ===============================
counts = train_df["label_id"].value_counts().sort_index().values
class_weights = torch.tensor(
    (counts.sum() / (counts * len(counts))), dtype=torch.float
)
use_class_weights = True

class WeightedTrainer(Trainer):
    # 兼容 >=4.57 传入的 num_items_in_batch；**kwargs 兜底未来参数
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        # 取出 labels，其余全部传给模型
        labels = inputs.pop("labels")
        outputs = model(**inputs)  # 包含 logits
        logits = outputs.logits

        if use_class_weights:
            loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(logits.device))
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ===============================
#  Model
# ===============================
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
)

# ===============================
#  Metrics
# ===============================
def compute_metrics(pred):
    logits, labels = pred
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro")
    }

# ===============================
#  TrainingArguments（自动兼容 & 成对一致策略）
# ===============================
print(f"[INFO] transformers version = {tf.__version__}")

def param_names(klass):
    try:
        return set(inspect.signature(klass.__init__).parameters.keys())
    except Exception:
        return set()

ta_params = param_names(TrainingArguments)

def supports(name: str) -> bool:
    return name in ta_params

OUT_DIR = os.path.join(BASE_DIR, "distilbert_out")

# 基础参数（各版本通用）
ta_kwargs = dict(
    output_dir=OUT_DIR,
    num_train_epochs=5,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=128,
    learning_rate=5e-5,
    weight_decay=0.01,
    seed=42,
)

# 可选：fp16 / warmup_ratio / report_to
if supports("fp16"):
    ta_kwargs["fp16"] = torch.cuda.is_available()
if supports("warmup_ratio"):
    ta_kwargs["warmup_ratio"] = 0.06
if supports("report_to"):
    ta_kwargs["report_to"] = "none"
if supports("dataloader_pin_memory"):
    ta_kwargs["dataloader_pin_memory"] = False


# —— 关键：评估/保存策略成对处理 —— #
# 优先使用 evaluation_strategy；若没有，尝试 eval_strategy；都没有则放弃策略并关闭 load_best
eval_key = None
if supports("evaluation_strategy"):
    eval_key = "evaluation_strategy"
elif supports("eval_strategy"):
    eval_key = "eval_strategy"

can_set_eval = eval_key is not None
can_set_save = supports("save_strategy")

if can_set_eval and can_set_save:
    # 成对设置 eval/save 为 epoch，并启用 load_best
    ta_kwargs[eval_key] = "epoch"
    ta_kwargs["save_strategy"] = "epoch"
    if supports("logging_strategy"):
        ta_kwargs["logging_strategy"] = "steps"
    if supports("logging_steps"):
        ta_kwargs["logging_steps"] = 50
    if supports("save_total_limit"):
        ta_kwargs["save_total_limit"] = 2
    if supports("load_best_model_at_end"):
        ta_kwargs["load_best_model_at_end"] = True
    if supports("metric_for_best_model"):
        ta_kwargs["metric_for_best_model"] = "f1"
    if supports("greater_is_better"):
        ta_kwargs["greater_is_better"] = True
else:
    # 无法设置 eval 策略：不设置 save_strategy，并确保不启用 load_best
    print("[WARN] Eval strategy not supported in this environment; running without in-training eval/save strategy.")
    if "save_strategy" in ta_kwargs:
        ta_kwargs.pop("save_strategy", None)
    # 确保这些与最佳模型有关的参数不被设置
    for k in ("load_best_model_at_end", "metric_for_best_model", "greater_is_better"):
        ta_kwargs.pop(k, None)
    # logging_steps（如果能设）仍然可以保留
    if supports("logging_steps"):
        ta_kwargs["logging_steps"] = 50

args = TrainingArguments(**ta_kwargs)

# ===============================
#  Trainer
# ===============================
trainer = WeightedTrainer(
    model=model,
    args=args,
    train_dataset=ds_train,
    eval_dataset=ds_valid,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# ===============================
#  Train
# ===============================
trainer.train()
trainer.save_model()
tokenizer.save_pretrained(OUT_DIR)

with open(os.path.join(OUT_DIR, "label_mapping.json"), "w") as f:
    json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)

# ===============================
#  VALID Evaluate
# ===============================
print("\n=== VALID RESULTS ===")
print(trainer.evaluate(ds_valid))

# ===============================
#  TEST Evaluate + Confusion Matrix
# ===============================
pred = trainer.predict(ds_test)
y_true = test_df["label_id"].to_numpy()
y_pred = pred.predictions.argmax(axis=-1)

print("\n=== TEST REPORT ===")
print(classification_report(y_true, y_pred, target_names=labels, zero_division=0))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(7,6))
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix — DistilBERT")
plt.colorbar()
ticks = np.arange(len(labels))
plt.xticks(ticks, labels, rotation=45, ha="right")
plt.yticks(ticks, labels)

th = cm.max() / 2 if cm.size else 0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        val = cm[i, j]
        plt.text(j, i, str(val), ha="center", va="center",
                 color="white" if val > th else "black")

plt.tight_layout()
save_path = os.path.join(OUT_DIR, "confusion_matrix.png")
plt.savefig(save_path, dpi=600)
plt.close()

print(f"[SAVED] confusion matrix → {save_path}")
