import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 使用 GPU 0

from datasets import load_dataset
from transformers import RobertaTokenizerFast
from transformers import RobertaForMaskedLM, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling
import torch

# 清理显存碎片
torch.cuda.empty_cache()

# 加载数据集
dataset = load_dataset("json", data_files="data/functions.jsonl", split="train")

# 加载预训练的 GraphCodeBERT tokenizer
tokenizer = RobertaTokenizerFast.from_pretrained("microsoft/graphcodebert-base")

# 对数据进行 tokenize
def tokenize_function(examples):
    return tokenizer(examples["code"], truncation=True, max_length=256)  # max_length改成256

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["code"])

# 打印预处理后的数据样本
print(tokenized_dataset[0])

# 数据整理器（MLM Masking）
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.15
)

# 训练参数
training_args = TrainingArguments(
    output_dir="./graphcodebert-mlm",
    num_train_epochs=3,
    per_device_train_batch_size=8,  # batch size减小
    gradient_accumulation_steps=2,  # 累积梯度模拟大 batch
    fp16=True,  # 半精度训练
    logging_dir="./logs",
    save_steps=5000,
    save_total_limit=2,
    prediction_loss_only=True
)

# 加载预训练 GraphCodeBERT
model = RobertaForMaskedLM.from_pretrained("microsoft/graphcodebert-base")

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

# 开始训练
trainer.train()

# 保存最终模型
model.save_pretrained("./graphcodebert-mlm-final")
tokenizer.save_pretrained("./graphcodebert-mlm-final")






