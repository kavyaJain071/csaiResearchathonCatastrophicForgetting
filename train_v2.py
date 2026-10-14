import os, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import argparse, json, random
import torch
from datasets import load_dataset, Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, Trainer,
                          TrainingArguments, DataCollatorForSeq2Seq)
from peft import LoraConfig, get_peft_model, PeftModel

p = argparse.ArgumentParser()
p.add_argument("--name", required=True)
p.add_argument("--mix", default="gsm8k:all", help="e.g. gsm8k:all,alpaca:3200")
p.add_argument("--schedule", default="shuffle", choices=["shuffle", "interleave", "block"])
p.add_argument("--modules", default="all", choices=["all", "attn", "mlp"])
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--epochs", type=float, default=1)
p.add_argument("--rank", type=int, default=16)
p.add_argument("--alpha", type=int, default=32)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--init_adapter", default=None, help="continue training from this adapter")
p.add_argument("--save_steps", type=int, default=100)
p.add_argument("--out_root", default="/kaggle/working/runs")
a = p.parse_args()

MODEL = "Qwen/Qwen2.5-1.5B"
MODS = {"attn": ["q_proj", "k_proj", "v_proj", "o_proj"], "mlp": ["gate_proj", "up_proj", "down_proj"]}
MODS["all"] = MODS["attn"] + MODS["mlp"]
out_dir = f"{a.out_root}/{a.name}"
os.makedirs(out_dir, exist_ok=True); os.makedirs("results/train_logs", exist_ok=True)

def load_source(name):
    if name == "gsm8k":
        return [(x["question"], x["answer"]) for x in load_dataset("openai/gsm8k", "main", split="train")]
    if name == "alpaca":
        return [(x["instruction"] + ("\n" + x["input"] if x["input"] else ""), x["output"])
                for x in load_dataset("yahma/alpaca-cleaned", split="train")]
    if name == "sciq":
        return [(x["question"], x["correct_answer"]) for x in load_dataset("allenai/sciq", split="train")]
    raise ValueError(name)

rng = random.Random(a.seed)
parts = []
for item in a.mix.split(","):
    src, n = item.split(":")
    ex = load_source(src); rng.shuffle(ex)
    parts.append(ex if n == "all" else ex[:int(n)])

CH = 32                                     # effective batch size
if a.schedule == "shuffle":
    data = [e for x in parts for e in x]; rng.shuffle(data)
else:
    assert len(parts) == 2, "interleave/block need exactly two sources: domain first, general second"
    parts = [x[: len(x) // CH * CH] for x in parts]
    chunks = [[x[i:i + CH] for i in range(0, len(x), CH)] for x in parts]
    if a.schedule == "block":
        order = chunks[0] + chunks[1]
    else:                                   # alternate whole batches in proportion to each source's size
        order, taken = [], [0, 0]
        while taken[0] < len(chunks[0]) or taken[1] < len(chunks[1]):
            fr = [taken[i] / max(len(chunks[i]), 1) for i in range(2)]
            i = 0 if fr[0] <= fr[1] else 1
            if taken[i] >= len(chunks[i]): i = 1 - i
            order.append(chunks[i][taken[i]]); taken[i] += 1
    data = [e for ch in order for e in ch]

tok = AutoTokenizer.from_pretrained(MODEL)
def enc(q, ans):
    pi = tok(f"Question: {q}\nAnswer:")["input_ids"]
    ai = tok(" " + ans + tok.eos_token)["input_ids"]
    ids = (pi + ai)[:512]; labels = ([-100] * len(pi) + ai)[:512]   # loss on the answer only
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}
ds = Dataset.from_list([enc(q, ans) for q, ans in data])
print("training examples:", len(ds), "| mix:", a.mix, "| schedule:", a.schedule)

model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to("cuda")
if a.init_adapter:
    model = PeftModel.from_pretrained(model, a.init_adapter, is_trainable=True)
else:
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.05,
                           task_type="CAUSAL_LM", target_modules=MODS[a.modules]))
model.gradient_checkpointing_enable(); model.enable_input_require_grads()
model.print_trainable_parameters()

class OrderedTrainer(Trainer):              # keep the dataset order we built (needed for interleave/block)
    def _get_train_sampler(self, *args, **kwargs):
        return torch.utils.data.SequentialSampler(self.train_dataset)

args = TrainingArguments(output_dir=out_dir, per_device_train_batch_size=4, gradient_accumulation_steps=8,
    learning_rate=a.lr, num_train_epochs=a.epochs, lr_scheduler_type="cosine", warmup_steps=15,
    fp16=True, logging_steps=10, save_steps=a.save_steps, save_only_model=True, report_to="none", seed=a.seed)
trainer = OrderedTrainer(model=model, args=args, train_dataset=ds,
                         data_collator=DataCollatorForSeq2Seq(tok, padding=True))

if a.schedule != "shuffle":                 # safety check: is the order really preserved?
    b = next(iter(trainer.get_train_dataloader()))
    row = b["input_ids"][0][b["attention_mask"][0].bool()].tolist()
    assert row == ds[0]["input_ids"], "Trainer is shuffling, so the schedule would be meaningless"

t0 = time.time(); trainer.train(); secs = time.time() - t0
model.save_pretrained(f"{out_dir}/final")
cfg = {**vars(a), "target_modules": MODS[a.modules], "train_examples": len(ds),
       "max_steps": trainer.state.max_steps, "train_seconds": round(secs),
       "trainable_params": sum(q.numel() for q in model.parameters() if q.requires_grad)}
json.dump(cfg, open(f"results/train_logs/{a.name}_config.json", "w"), indent=2)
json.dump(trainer.state.log_history, open(f"results/train_logs/{a.name}_loss.json", "w"))
