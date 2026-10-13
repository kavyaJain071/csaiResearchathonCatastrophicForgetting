import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import argparse, json
import torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, Trainer,
                          TrainingArguments, DataCollatorForSeq2Seq)
from peft import LoraConfig, get_peft_model

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--modules", default="all", choices=["all", "attn", "mlp"])
p.add_argument("--lr", type=float, default=2e-4)
p.add_argument("--epochs", type=float, default=2)
p.add_argument("--rank", type=int, default=16)
p.add_argument("--alpha", type=int, default=32)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--out_root", default="/kaggle/working/runs")
a = p.parse_args()

MODEL = "Qwen/Qwen2.5-1.5B"
MODULES = {"attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
           "mlp": ["gate_proj", "up_proj", "down_proj"]}
MODULES["all"] = MODULES["attn"] + MODULES["mlp"]

out_dir = f"{a.out_root}/{a.name}"
os.makedirs(out_dir, exist_ok=True)
os.makedirs("results/train_logs", exist_ok=True)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to("cuda")
model = get_peft_model(model, LoraConfig(
    r=a.rank, lora_alpha=a.alpha, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=MODULES[a.modules]))
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.print_trainable_parameters()

# full GSM8K training split (no subset); the test split is never used
ds = load_dataset("openai/gsm8k", "main", split="train")
print("training examples:", len(ds))

def prep(ex):
    prompt = f"Question: {ex['question']}\nAnswer:"
    answer = " " + ex["answer"] + tok.eos_token
    pi, ai = tok(prompt)["input_ids"], tok(answer)["input_ids"]
    ids = (pi + ai)[:512]
    labels = ([-100] * len(pi) + ai)[:512]       # learn only from the answer
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}

ds = ds.map(prep, remove_columns=ds.column_names)

args = TrainingArguments(
    output_dir=out_dir, per_device_train_batch_size=4, gradient_accumulation_steps=8,
    learning_rate=a.lr, num_train_epochs=a.epochs, lr_scheduler_type="cosine",
    warmup_steps=15, fp16=True, logging_steps=10, save_steps=50, save_total_limit=None,
    save_only_model=True, report_to="none", seed=a.seed)

trainer = Trainer(model=model, args=args, train_dataset=ds,
                  data_collator=DataCollatorForSeq2Seq(tok, padding=True))
trainer.train()
model.save_pretrained(f"{out_dir}/final")

cfg = {**vars(a), "target_modules": MODULES[a.modules], "dropout": 0.05,
       "per_device_batch": 4, "grad_accum": 8, "effective_batch": 32,
       "lr_schedule": "cosine", "warmup_steps": 15, "precision": "fp16",
       "max_seq_len": 512, "train_examples": len(ds), "save_every_steps": 50,
       "trainable_params": sum(q.numel() for q in model.parameters() if q.requires_grad),
       "total_params": sum(q.numel() for q in model.parameters())}
json.dump(cfg, open(f"results/train_logs/{a.name}_config.json", "w"), indent=2)
json.dump(trainer.state.log_history, open(f"results/train_logs/{a.name}_loss.json", "w"))
