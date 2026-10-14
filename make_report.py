import json, glob, os, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS = [("base", "Base model"), ("attn_only", "Attention only"),
          ("mlp_only", "MLP only"), ("both", "Both (all linear)")]
ADAPTERS = MODELS[1:]
BENCH = [("GSM8K", "gsm8k", "exact_match,flexible-extract", "gsm8k"),
         ("MMLU", "mmlu", "acc,none", "mmlu"),
         ("ARC-Challenge", "arc_challenge", "acc_norm,none", "fast"),
         ("HellaSwag", "hellaswag", "acc_norm,none", "fast"),
         ("Winogrande", "winogrande", "acc,none", "fast")]
GEN = [b for b in BENCH if b[0] != "GSM8K"]

def get(model, kind, task, metric):
    files = sorted(glob.glob(f"results/{model}_{kind}_full/**/results_*.json", recursive=True))
    if not files: return None
    r = json.load(open(files[-1]))["results"][task]
    return r[metric], r[metric.replace(",", "_stderr,")]

S = {m: {b[0]: get(m, b[3], b[1], b[2]) for b in BENCH} for m, _ in MODELS}

def delta(m, b):
    base, mod = S["base"][b[0]], S[m][b[0]]
    if not base or not mod: return None
    return mod[0] - base[0], math.hypot(base[1], mod[1])

def cfg(name):
    p = f"results/train_logs/{name}_config.json"
    return json.load(open(p)) if os.path.exists(p) else {}

# ---------- figures ----------
fig, ax = plt.subplots(figsize=(11, 4.5)); w = 0.2
for i, (m, label) in enumerate(MODELS):
    pts = [(j + (i - 1.5) * w, *S[m][b[0]]) for j, b in enumerate(BENCH) if S[m][b[0]]]
    if pts: ax.bar([p[0] for p in pts], [p[1] for p in pts], w, yerr=[p[2] for p in pts], capsize=3, label=label)
ax.set_xticks(range(len(BENCH))); ax.set_xticklabels([b[0] for b in BENCH])
ax.set_ylabel("score"); ax.set_title("Scores on the full test sets (error bars = standard error)")
ax.legend(); fig.tight_layout(); fig.savefig("results/fig_scores.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(11, 4.5)); w = 0.25
for i, (m, label) in enumerate(ADAPTERS):
    pts = [(j + (i - 1) * w, *delta(m, b)) for j, b in enumerate(BENCH) if delta(m, b)]
    if pts: ax.bar([p[0] for p in pts], [p[1] for p in pts], w, yerr=[p[2] for p in pts], capsize=3, label=label)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(range(len(BENCH))); ax.set_xticklabels([b[0] for b in BENCH])
ax.set_ylabel("score change vs base"); ax.set_title("Change relative to the base model")
ax.legend(); fig.tight_layout(); fig.savefig("results/fig_change.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4))
for m, label in ADAPTERS:
    p = f"results/train_logs/{m}_loss.json"
    if os.path.exists(p):
        h = [x for x in json.load(open(p)) if "loss" in x]
        ax.plot([x["step"] for x in h], [x["loss"] for x in h], label=label)
ax.set_xlabel("training step"); ax.set_ylabel("training loss"); ax.set_title("Training loss")
ax.legend(); fig.tight_layout(); fig.savefig("results/fig_loss.png", dpi=150); plt.close(fig)

def avg_general(m):
    ds = [delta(m, b) for b in GEN]; ds = [d[0] for d in ds if d]
    return sum(ds) / len(ds) if ds else None

fig, ax = plt.subplots(figsize=(5.5, 4.5))
for m, label in ADAPTERS:
    g, a = delta(m, BENCH[0]), avg_general(m)
    if g and a is not None:
        ax.scatter(g[0], a); ax.annotate(label, (g[0], a), textcoords="offset points", xytext=(5, 5))
ax.axhline(0, color="grey", lw=0.6); ax.axvline(0, color="grey", lw=0.6)
ax.set_xlabel("GSM8K change (specialization gain)"); ax.set_ylabel("average change on the 4 general benchmarks")
ax.set_title("Gain vs. forgetting"); fig.tight_layout(); fig.savefig("results/fig_tradeoff.png", dpi=150); plt.close(fig)

# ---------- README ----------
def fmt(r): return f"{r[0]:.3f} ± {r[1]:.3f}" if r else "-"
c = cfg("both")
L = []
L += ["# Specialization Without Catastrophic Forgetting in Small Language Models", "",
      "We fine-tune `Qwen/Qwen2.5-1.5B` on GSM8K (math) with LoRA, once with adapters on the attention layers only, "
      "once on the MLP layers only, and once on both, then measure what each configuration gains on math and what it "
      "loses on general benchmarks (full test sets, no sample limits).", "",
      "## Benchmarks", "",
      "| Benchmark | What it tests | Questions | Metric | Role |", "|---|---|---:|---|---|",
      "| GSM8K | Grade-school math word problems; model writes a solution | 1,319 | exact match (flexible-extract) | Target skill |",
      "| MMLU | Knowledge across 57 subjects | 14,042 | accuracy | General knowledge |",
      "| ARC-Challenge | Hard science questions | 1,172 | acc_norm | Science reasoning |",
      "| HellaSwag | Commonsense sentence completion | 10,042 | acc_norm | Commonsense |",
      "| Winogrande | Pronoun resolution | 1,267 | accuracy | Language understanding |", "",
      "Chance level: 25% for MMLU, ARC and HellaSwag; 50% for Winogrande.", "",
      "## Evaluation settings", "",
      "[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), base-model prompting (no chat template), "
      "`float16`, greedy decoding. GSM8K: 5-shot, `max_gen_toks=256`. All other benchmarks: 0-shot. No `--limit` anywhere.", "",
      "## Training setup", "",
      f"Data: GSM8K train split, all {c.get('train_examples', 7473):,} examples. Prompt `Question: ...\\nAnswer:`, "
      f"loss on answer tokens only, max {c.get('max_seq_len', 512)} tokens.", "",
      "| Setting | Value |", "|---|---|",
      f"| LoRA rank / alpha / dropout | {c.get('rank', 16)} / {c.get('alpha', 32)} / {c.get('dropout', 0.05)} |",
      f"| Learning rate | {c.get('lr', 2e-4)} ({c.get('lr_schedule', 'cosine')}, {c.get('warmup_steps', 15)} warmup steps) |",
      f"| Epochs | {c.get('epochs', 2)} |",
      f"| Batch size | {c.get('per_device_batch', 4)} x {c.get('grad_accum', 8)} accumulation = {c.get('effective_batch', 32)} |",
      f"| Precision | {c.get('precision', 'fp16')} |",
      f"| Checkpoints | every {c.get('save_every_steps', 50)} steps |",
      f"| Seed | {c.get('seed', 42)} |", "",
      "| Configuration | Target modules | Trainable parameters |", "|---|---|---:|"]
for m, label in ADAPTERS:
    k = cfg(m)
    L.append(f"| {label} | {', '.join(k.get('target_modules', []))} | {k.get('trainable_params', 0) / 1e6:.1f}M |")
L += ["", "![Training loss](results/fig_loss.png)", "", "## Results", "",
      "![Scores](results/fig_scores.png)", "",
      "| Model | GSM8K | MMLU | ARC-Challenge | HellaSwag | Winogrande |", "|---|---:|---:|---:|---:|---:|"]
for m, label in MODELS:
    L.append(f"| {label} | " + " | ".join(fmt(S[m][b[0]]) for b in BENCH) + " |")
L += ["", "Values are score ± standard error.", "", "### Change relative to the base model", "",
      "![Change](results/fig_change.png)", "",
      "| Model | GSM8K | MMLU | ARC-Challenge | HellaSwag | Winogrande | Avg (4 general) |", "|---|---:|---:|---:|---:|---:|---:|"]
flags = []
for m, label in ADAPTERS:
    cells = []
    for b in BENCH:
        d = delta(m, b)
        if not d: cells.append("-"); continue
        big = abs(d[0]) > 2 * d[1]
        cells.append(f"{d[0]:+.3f}{' *' if big else ''}")
        if big: flags.append(f"{label}: {b[0]} {d[0]:+.3f}")
    a = avg_general(m)
    L.append(f"| {label} | " + " | ".join(cells) + f" | {a:+.3f} |" if a is not None else f"| {label} | " + " | ".join(cells) + " | - |")
L += ["", "\\* larger than twice the combined standard error (unpaired), so unlikely to be noise.", "",
      "### Gain vs. forgetting", "", "![Trade-off](results/fig_tradeoff.png)", "",
      "## Key observations", "",
      "Changes beyond noise: " + ("; ".join(flags) if flags else "none."), "",
      "*Add your interpretation here.*", "",
      "## Reproducing", "",
      "1. `python train.py --name attn_only --modules attn` (likewise `mlp_only --modules mlp`, `both --modules all`)",
      "2. Evaluate with `lm_eval` using the settings above, writing to `results/<model>_<kind>_full/`",
      "3. `python make_report.py` regenerates the figures and this README.", ""]
open("README.md", "w").write("\n".join(L))
print("README.md and figures written")
