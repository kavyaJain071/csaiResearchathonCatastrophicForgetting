import json, glob, os, csv
KEYS = {"gsm8k": ("gsm8k", "exact_match,flexible-extract"), "mmlu": ("mmlu", "acc,none"),
        "mmlu_stem": ("mmlu_stem", "acc,none"), "mmlu_humanities": ("mmlu_humanities", "acc,none"),
        "mmlu_social": ("mmlu_social_sciences", "acc,none"), "mmlu_other": ("mmlu_other", "acc,none"),
        "arc": ("arc_challenge", "acc_norm,none"), "hellaswag": ("hellaswag", "acc_norm,none"),
        "winogrande": ("winogrande", "acc,none"), "sciq": ("sciq", "acc,none")}
rows = {}
for d in sorted(glob.glob("results/*_*")):
    name = os.path.basename(d)
    if name.endswith("_full") or not os.path.isdir(d): continue
    files = sorted(glob.glob(f"{d}/**/results_*.json", recursive=True))
    if not files: continue
    res = json.load(open(files[-1]))["results"]
    row = rows.setdefault(name.rsplit("_", 1)[0], {})
    for k, (task, metric) in KEYS.items():
        if task in res and metric in res[task]:
            row[k] = res[task][metric]
            row[k + "_se"] = res[task].get(metric.replace(",", "_stderr,"), "")
cols = ["label"] + [c for k in KEYS for c in (k, k + "_se")]
with open("results/summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, cols, restval=""); w.writeheader()
    for l, r in rows.items(): w.writerow({"label": l, **r})
for l, r in rows.items():
    print(l.ljust(20), " ".join(f"{k}={float(r[k]):.3f}" for k in KEYS if k in r and r[k] != ""))
