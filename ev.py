import argparse, os, glob, subprocess
p = argparse.ArgumentParser()
p.add_argument("--label", required=True)
p.add_argument("--peft", default=None)
p.add_argument("--merged", default=None, help="path of a full (merged) model")
p.add_argument("--suites", default="mc,mmlu,gsm8k,sciq")
p.add_argument("--gsm_bs", type=int, default=48)
a = p.parse_args()
SUITES = {"mc":    ("arc_challenge,hellaswag,winogrande", 0, "--limit 1000", 16),
          "mmlu":  ("mmlu", 0, "--limit 50", 8),
          "gsm8k": ("gsm8k", 5, "--limit 300 --gen_kwargs max_gen_toks=256", a.gsm_bs),
          "sciq":  ("sciq", 0, "", 16)}
for s in a.suites.split(","):
    tasks, shots, extra, bs = SUITES[s]
    folder = f"results/{a.label}_{s}"
    if glob.glob(f"{folder}/**/results_*.json", recursive=True):
        print("skip", folder); continue
    margs = "pretrained=" + (a.merged or "Qwen/Qwen2.5-1.5B") + ",dtype=float16"
    if a.peft: margs += f",peft={a.peft}"
    subprocess.run(f"lm_eval --model hf --model_args {margs} --tasks {tasks} --num_fewshot {shots} "
                   f"--batch_size {bs} {extra} --output_path {folder}", shell=True)
