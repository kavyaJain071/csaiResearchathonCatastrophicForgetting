import argparse, json, torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer
p = argparse.ArgumentParser()
p.add_argument("--adapters", required=True, help="comma-separated adapter folders")
p.add_argument("--method", required=True, choices=["am", "gm", "hm", "sum"])
p.add_argument("--out", required=True)
a = p.parse_args()
paths = a.adapters.split(",")
MODEL = "Qwen/Qwen2.5-1.5B"
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to("cuda")
cfgs = [json.load(open(f"{d}/adapter_config.json")) for d in paths]
scal = [c["lora_alpha"] / c["r"] for c in cfgs]
W = [load_file(f"{d}/adapter_model.safetensors", device="cuda") for d in paths]
params = dict(model.named_parameters())

def combine(D):                                   # D: (K, out, in)
    if a.method == "am":  return D.mean(0)
    if a.method == "sum": return D.sum(0)
    pos, neg = (D > 0).all(0), (D < 0).all(0)     # keep an entry only if all adapters agree on its sign
    M = D.abs().clamp_min(1e-12)
    mag = torch.exp(torch.log(M).mean(0)) if a.method == "gm" else len(D) / (1.0 / M).sum(0)
    return torch.where(pos, mag, torch.where(neg, -mag, torch.zeros_like(mag)))

for key in W[0]:
    if ".lora_A.weight" not in key: continue
    kb = key.replace("lora_A", "lora_B")
    name = key.replace("base_model.model.", "").replace(".lora_A.weight", ".weight")
    D = torch.stack([s * (w[kb].float() @ w[key].float()) for s, w in zip(scal, W)])
    params[name].data = (params[name].data.float() + combine(D)).half()

model.save_pretrained(a.out); AutoTokenizer.from_pretrained(MODEL).save_pretrained(a.out)
print("merged", a.method, "->", a.out)
