import os, json, glob, math
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.system("python collect.py > /dev/null")
df = pd.read_csv("results/summary.csv").set_index("label")
NAMES = {"gsm8k": "GSM8K", "mmlu": "MMLU", "mmlu_stem": "MMLU STEM", "mmlu_humanities": "MMLU humanities",
         "mmlu_social": "MMLU social sci.", "mmlu_other": "MMLU other", "arc": "ARC", "hellaswag": "HellaSwag",
         "winogrande": "Winogrande", "sciq": "SciQ"}
GEN = ["mmlu", "arc", "hellaswag", "winogrande"]
KEYS = ["gsm8k", "mmlu", "mmlu_stem", "arc", "hellaswag", "winogrande", "sciq"]

def val(l, k):
    if l not in df.index or k not in df.columns: return None
    v, s = df.loc[l, k], df.loc[l, k + "_se"]
    return None if pd.isna(v) else (float(v), 0.0 if pd.isna(s) else float(s))
def chg(l, k):                                   # change vs base, with combined standard error
    a, b = val(l, k), val("base", k)
    return None if not a or not b else (a[0] - b[0], math.hypot(a[1], b[1]))
def rel(l, k):                                   # relative change in %
    c, b = chg(l, k), val("base", k)
    return None if not c or not b else (c[0] / b[0] * 100, c[1] / b[0] * 100)
def star(c): return " *" if c and abs(c[0]) > 2 * c[1] else ""
def gen_avg(l):
    c = [chg(l, k) for k in GEN]; c = [x[0] for x in c if x]
    return sum(c) / len(c) if c else None
def has(l): return any(val(l, k) for k in KEYS)

# ---------------- figures ----------------
CAPS = ["mmlu_stem", "mmlu_humanities", "mmlu_social", "mmlu_other", "arc", "hellaswag", "winogrande"]
def fig_q1():
    ck = {int(l.split("ckpt")[1]): l for l in df.index if str(l).startswith("A_ckpt")}
    cp = "results/train_logs/A_config.json"
    if "A_final" in df.index and os.path.exists(cp): ck[json.load(open(cp))["max_steps"]] = "A_final"
    if not ck: return False
    steps = sorted(ck); last = ck[steps[-1]]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for k in CAPS:
        if all(rel(ck[s], k) for s in steps):
            ax[0].plot([0] + steps, [0] + [rel(ck[s], k)[0] for s in steps], marker="o", label=NAMES[k])
    ax[0].axhline(0, color="k", lw=.7); ax[0].set_xlabel("training step"); ax[0].set_ylabel("% change vs base")
    ax[0].set_title("General capabilities during training (run A)"); ax[0].legend(fontsize=8)
    order = sorted([k for k in CAPS if rel(last, k)], key=lambda k: rel(last, k)[0])
    ax[1].barh([NAMES[k] for k in order], [rel(last, k)[0] for k in order], xerr=[rel(last, k)[1] for k in order], capsize=3)
    ax[1].axvline(0, color="k", lw=.7); ax[1].set_xlabel("% change vs base"); ax[1].set_title("Final change (most fragile at top)")
    fig.tight_layout(); fig.savefig("results/fig_q1_fragility.png", dpi=150); plt.close(fig); return True

Q2 = [("A_final", "GSM8K only"), ("M1_mix_final", "Mixed (shuffled)"), ("M2_interleave_final", "Interleaved batches"),
      ("M3_targeted_final", "Targeted mix (SciQ)")]
def fig_q2():
    runs = [(l, n) for l, n in Q2 if has(l)]
    if len(runs) < 2: return False
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    g = [(i, val(l, "gsm8k")) for i, (l, _) in enumerate(runs) if val(l, "gsm8k")]
    ax[0].bar([i for i, _ in g], [v[0] for _, v in g], yerr=[v[1] for _, v in g], capsize=4)
    ax[0].axhline(val("base", "gsm8k")[0], color="k", ls="--", lw=.8, label="base model"); ax[0].legend()
    ax[0].set_xticks(range(len(runs))); ax[0].set_xticklabels([n for _, n in runs], rotation=15); ax[0].set_title("GSM8K (higher is better)")
    ks = ["mmlu", "mmlu_stem", "arc", "hellaswag", "winogrande"]; w = 0.8 / len(runs)
    for i, (l, n) in enumerate(runs):
        cs = [(j, chg(l, k)) for j, k in enumerate(ks) if chg(l, k)]
        ax[1].bar([j + i * w for j, _ in cs], [c[0] * 100 for _, c in cs], w, yerr=[c[1] * 100 for _, c in cs], capsize=2, label=n)
    ax[1].set_xticks([j + 0.4 - w / 2 for j in range(len(ks))]); ax[1].set_xticklabels([NAMES[k] for k in ks])
    ax[1].axhline(0, color="k", lw=.7); ax[1].set_ylabel("change vs base (points)"); ax[1].set_title("General capabilities"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig("results/fig_q2_data_strategies.png", dpi=150); plt.close(fig); return True

METRICS = ["gsm8k", "sciq", "mmlu_stem", "mmlu", "arc", "hellaswag", "winogrande"]
def heat(rows, title, fname):
    rows = [(l, n) for l, n in rows if has(l)]
    if not rows: return False
    M = np.array([[chg(l, k)[0] * 100 if chg(l, k) else np.nan for k in METRICS] for l, _ in rows])
    v = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1
    fig, ax = plt.subplots(figsize=(9.5, 0.6 * len(rows) + 2))
    im = ax.imshow(M, cmap="RdBu", vmin=-v, vmax=v, aspect="auto")
    ax.set_xticks(range(len(METRICS))); ax.set_xticklabels([NAMES[k] for k in METRICS], rotation=30, ha="right")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([n for _, n in rows]); ax.set_title(title)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]): ax.text(j, i, f"{M[i, j]:+.1f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, label="change vs base (points)"); fig.tight_layout(); fig.savefig(fname, dpi=150); plt.close(fig); return True

Q3 = [("A_final", "Stage 1: math"), ("S2_final", "Stage 2: + science"), ("S3_final", "Stage 3: + general")]
Q4 = [("A_final", "Math adapter"), ("Ds_final", "Science adapter"), ("Dg_final", "General adapter"),
      ("merge_am", "Merge: arithmetic mean"), ("merge_sum", "Merge: sum"), ("merge_gm", "Merge: geometric mean"),
      ("merge_hm", "Merge: harmonic mean"), ("S3_final", "Sequential (Q3, stage 3)")]
ALL = [("A_final", "GSM8K only"), ("M1_mix_final", "Mix"), ("M2_interleave_final", "Interleave"), ("M3_targeted_final", "Targeted mix"),
       ("S2_final", "Seq. stage 2"), ("S3_final", "Seq. stage 3"), ("Ds_final", "Science adapter"), ("Dg_final", "General adapter"),
       ("merge_am", "Merge AM"), ("merge_sum", "Merge sum")]
def fig_tradeoff():
    pts = [(n, chg(l, "gsm8k"), gen_avg(l)) for l, n in ALL]
    pts = [(n, c[0] * 100, g * 100) for n, c, g in pts if c and g is not None]
    if not pts: return False
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for n, x, y in pts: ax.scatter(x, y); ax.annotate(n, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax.axhline(0, color="grey", lw=.6); ax.axvline(0, color="grey", lw=.6)
    ax.set_xlabel("GSM8K change (points)"); ax.set_ylabel("average change on MMLU, ARC, HellaSwag, Winogrande (points)")
    ax.set_title("Specialization gain vs. forgetting"); fig.tight_layout(); fig.savefig("results/fig_tradeoff_all.png", dpi=150); plt.close(fig); return True

made = {"q1": fig_q1(), "q2": fig_q2(), "tradeoff": fig_tradeoff(),
        "q3": heat(Q3, "Sequential specialization: change vs base (points)", "results/fig_q3_sequential.png"),
        "q4": heat(Q4, "Adapter composition: change vs base (points)", "results/fig_q4_merging.png")}

# ---------------- tables ----------------
def table(rows):
    rows = [(l, n) for l, n in rows if has(l)]
    L = ["| Model | " + " | ".join(NAMES[k] for k in KEYS) + " |", "|---|" + "---:|" * len(KEYS)]
    for l, n in rows:
        L.append(f"| {n} | " + " | ".join(f"{val(l, k)[0]:.3f} ± {val(l, k)[1]:.3f}" if val(l, k) else "-" for k in KEYS) + " |")
    return L
def change_table(rows):
    rows = [(l, n) for l, n in rows if has(l)]
    L = ["| Model | " + " | ".join(NAMES[k] for k in KEYS) + " | Avg (4 general) |", "|---|" + "---:|" * (len(KEYS) + 1)]
    for l, n in rows:
        g = gen_avg(l)
        L.append(f"| {n} | " + " | ".join(f"{chg(l, k)[0] * 100:+.1f}{star(chg(l, k))}" if chg(l, k) else "-" for k in KEYS)
                 + f" | {g * 100:+.1f} |" if g is not None else " | - |")
    return L
def bullets(rows):
    out = []
    for l, n in rows:
        if not has(l): continue
        parts = []
        for k in ["gsm8k", "mmlu_stem", "mmlu", "sciq"]:
            c = chg(l, k)
            if c: parts.append(f"{NAMES[k]} {c[0] * 100:+.1f}{star(c)}")
        g = gen_avg(l)
        out.append(f"- **{n}:** " + ", ".join(parts) + (f"; average general change {g * 100:+.1f}" if g is not None else ""))
    return out
def img(path, alt): return [f"![{alt}]({path})", ""] if os.path.exists(path) else []

def runs_table():
    L = ["| Run | Data (examples) | Schedule | Start from | LR | Epochs | Steps | Minutes | Trainable params |",
         "|---|---|---|---|---:|---:|---:|---:|---:|"]
    for p in sorted(glob.glob("results/train_logs/*_config.json")):
        c = json.load(open(p)); name = os.path.basename(p).split("_config")[0]
        init = os.path.basename(os.path.dirname(c["init_adapter"])) if c.get("init_adapter") else "base model"
        mins = round(c["train_seconds"] / 60) if c.get("train_seconds") else "-"
        L.append(f"| {name} | {c.get('mix', 'gsm8k:all')} | {c.get('schedule', 'shuffle')} | {init} | {c.get('lr')} | {c.get('epochs')} | "
                 f"{c.get('max_steps', 468)} | {mins} | {c.get('trainable_params', 0) / 1e6:.1f}M |")
    return L

def full(m, kind, task, metric):
    f = sorted(glob.glob(f"results/{m}_{kind}_full/**/results_*.json", recursive=True))
    if not f: return None
    r = json.load(open(f[-1]))["results"][task]
    return r[metric], r[metric.replace(",", "_stderr,")]
FULLB = [("GSM8K", "gsm8k", "exact_match,flexible-extract", "gsm8k"), ("MMLU", "mmlu", "acc,none", "mmlu"),
         ("ARC-Challenge", "arc_challenge", "acc_norm,none", "fast"), ("HellaSwag", "hellaswag", "acc_norm,none", "fast"),
         ("Winogrande", "winogrande", "acc,none", "fast")]
FULLM = [("base", "Base model"), ("attn_only", "Attention only"), ("mlp_only", "MLP only"), ("both", "Both (all linear)")]

L = ["# Specialization Without Catastrophic Forgetting in Small Language Models", "",
     "We fine-tune `Qwen/Qwen2.5-1.5B` with LoRA on GSM8K (math) and measure what is gained on math and what is lost elsewhere. "
     "We ask which capabilities are most fragile, whether data strategies reduce forgetting, whether a model can be specialized "
     "for several domains in sequence, and whether separately trained adapters can be merged.", "",
     "## Benchmarks", "",
     "| Benchmark | What it tests | Questions (full set) | Metric | Role |", "|---|---|---:|---|---|",
     "| GSM8K | Grade-school math word problems; the model writes a solution | 1,319 | exact match (flexible-extract) | Target skill |",
     "| MMLU | Knowledge across 57 subjects (STEM, humanities, social sciences, other) | 14,042 | accuracy | General knowledge |",
     "| ARC-Challenge | Hard science questions | 1,172 | acc_norm | Science reasoning |",
     "| HellaSwag | Commonsense sentence completion | 10,042 | acc_norm | Commonsense |",
     "| Winogrande | Pronoun resolution | 1,267 | accuracy | Language understanding |",
     "| SciQ | Science question answering (multiple choice) | 1,000 | accuracy | Science domain / retention |", "",
     "Chance level: 25% for MMLU, ARC and HellaSwag; 50% for Winogrande. All evaluations use "
     "[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), base-model prompting without a chat template, `float16`, greedy decoding.", "",
     "## Evaluation protocols", "",
     "| Protocol | Used for | GSM8K | MMLU | ARC / HellaSwag / Winogrande | SciQ |", "|---|---|---|---|---|---|",
     "| Full test sets | reference adapters (section 1) | 5-shot, all 1,319 | 0-shot, all | 0-shot, all | - |",
     "| Limited | all stress experiments (sections 2 to 5) | 5-shot, first 300 | 0-shot, 50 per subject | 0-shot, first 1,000 | 0-shot, all 1,000 |", "",
     "GSM8K uses `max_gen_toks=256`. The base model's GSM8K in the limited protocol is its full-test score (0.622), which matches its first-300 score (0.623). "
     "Changes marked **\\*** are larger than twice the combined standard error and unlikely to be noise.", "",
     "## Training runs", "", *runs_table(), "",
     "All runs: LoRA rank 16, alpha 32, dropout 0.05, all linear layers (`q,k,v,o,gate,up,down`) unless the name says otherwise, "
     "effective batch 32 (4 x 8 accumulation), cosine schedule with 15 warmup steps, fp16, seed 42, loss on answer tokens only, "
     "max 512 tokens. `attn_only`, `mlp_only` and `both` are the reference runs (lr 2e-4, 2 epochs); the other runs are the "
     "stress setting (lr 1e-3, a single pass).", ""]

L += ["## 1. Reference setting: where to put the adapters", "",
      "Full test sets, lr 2e-4, 2 epochs on GSM8K. Values are score ± standard error.", "",
      "| Model | " + " | ".join(b[0] for b in FULLB) + " |", "|---|" + "---:|" * len(FULLB)]
for m, n in FULLM:
    L.append(f"| {n} | " + " | ".join(f"{full(m, b[3] + '' if False else b[3], b[1], b[2])[0]:.3f} ± {full(m, b[3], b[1], b[2])[1]:.3f}" if full(m, b[3], b[1], b[2]) else "-" for b in FULLB) + " |")
L += [""] + img("results/fig_scores.png", "Scores") + img("results/fig_change.png", "Change vs base") + img("results/fig_loss.png", "Training loss")

L += ["## 2. Which capabilities are most fragile?", "",
      "Run `A`: GSM8K only, lr 1e-3, 234 steps, evaluated at several checkpoints. Change relative to the base model:", ""]
L += img("results/fig_q1_fragility.png", "Fragility") + (["| Capability | Base | Final | Change (pts) | Relative |", "|---|---:|---:|---:|---:|"] +
      [f"| {NAMES[k]} | {val('base', k)[0]:.3f} | {val('A_final', k)[0]:.3f} | {chg('A_final', k)[0] * 100:+.1f}{star(chg('A_final', k))} | {rel('A_final', k)[0]:+.0f}% |"
       for k in sorted([k for k in CAPS + ['sciq'] if rel('A_final', k)], key=lambda k: rel('A_final', k)[0])] if "A_final" in df.index else []) + [""]

L += ["## 3. Data strategies", "",
      "All runs use the same GSM8K examples and the stress setting. `Mixed` adds 3,200 Alpaca examples shuffled in; `Interleaved` alternates "
      "whole batches of GSM8K and Alpaca; `Targeted` adds 3,200 SciQ examples (aimed at the most fragile area, MMLU STEM).", ""]
L += img("results/fig_q2_data_strategies.png", "Data strategies") + table(Q2) + ["", "Change vs base (points):", ""] + change_table(Q2) + [""] + bullets(Q2) + [""]

L += ["## 4. Sequential specialization", "",
      "Stage 1 is run `A` (math). Stage 2 continues from it on 1,000 GSM8K examples (replay) plus 2,400 SciQ. Stage 3 continues on 700 GSM8K, 700 SciQ and 2,000 Alpaca.", ""]
L += img("results/fig_q3_sequential.png", "Interference matrix") + change_table(Q3) + [""] + bullets(Q3) + [""]

L += ["## 5. Adapter composition", "",
      "Math (`A`), science (`Ds`, 3,200 SciQ) and general (`Dg`, 3,200 Alpaca) adapters were trained separately from the base model with the stress setting, "
      "then merged in weight-change space (the full update B·A of each adapter). Methods: arithmetic mean, sum (task arithmetic, scale 1), and "
      "geometric and harmonic means that keep an entry only where all adapters agree on its sign (a plain geometric mean is undefined for signed values).", ""]
L += img("results/fig_q4_merging.png", "Merging") + table(Q4) + ["", "Change vs base (points):", ""] + change_table(Q4) + [""] + bullets(Q4) + [""]

L += ["## Gain versus forgetting", ""] + img("results/fig_tradeoff_all.png", "Trade-off") + [
      "## Key observations", "", "*To be written from the results above.*", "",
      "## Limitations", "",
      "- Single seed per configuration; differences smaller than about two standard errors should be treated as noise.",
      "- Limited evaluations in sections 2 to 5 (first 300 GSM8K questions, 1,000 questions for ARC, HellaSwag and Winogrande, 50 per MMLU subject).",
      "- The stress setting (lr 1e-3, one pass) was chosen because the reference setting (lr 2e-4) produced no measurable forgetting; "
      "it also leaves GSM8K at or below the base model, so these runs measure forgetting more than specialization gain.",
      "- SciQ is almost saturated for this model (about 0.94), so it measures retention rather than learning.",
      "- Geometric and harmonic merges use a sign-agreement rule of our own design, not a published method.", "",
      "## Reproducing", "",
      "`train_v2.py` trains (data mix, schedule and starting adapter are flags), `ev.py` runs the evaluation suites, `merge.py` merges adapters, "
      "`run_all.py` runs the whole pipeline and resumes after interruptions, `collect.py` builds `results/summary.csv`, and `make_report2.py` "
      "regenerates the figures and this README. Fine-tuned adapters (fp16) are in `adapters_backup/`.", ""]
open("README.md", "w").write("\n".join(L))
print("figures:", made); print("README.md written")
