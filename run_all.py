import os, subprocess, threading, time, shutil, glob
from safetensors.torch import load_file, save_file

R = "/kaggle/working/runs"
os.makedirs(R, exist_ok=True); os.makedirs("/kaggle/working/logs", exist_ok=True)
F1, F2 = "sciq", "alpaca"                      # targeted and general data (from the Q1 result)

def sh(cmd): return subprocess.run(cmd, shell=True, capture_output=True, text=True)
def push(msg):
    sh("git add -A"); sh(f'git commit -q -m "{msg}"')
    sh("git pull -q --rebase --autostash origin main"); sh("git push -q -u origin HEAD")

def restore():                                  # put backed-up adapters back after a restart
    for d in glob.glob("adapters_backup/*"):
        dst = f"{R}/{os.path.basename(d)}/final"
        if not os.path.exists(f"{dst}/adapter_model.safetensors"):
            os.makedirs(dst, exist_ok=True)
            for f in os.listdir(d): shutil.copy(f"{d}/{f}", dst)

def backup():                                   # save finished adapters (fp16) into the repo
    for d in glob.glob(f"{R}/*/final"):
        n = d.split("/")[-2]; dst = f"adapters_backup/{n}"; src = f"{d}/adapter_model.safetensors"
        if os.path.exists(f"{dst}/adapter_model.safetensors") or not os.path.exists(src): continue
        w = {k: v.half().contiguous() for k, v in load_file(src).items()}
        os.makedirs(dst, exist_ok=True); save_file(w, f"{dst}/adapter_model.safetensors")
        shutil.copy(f"{d}/adapter_config.json", dst)

def T(name, args):                              # train only if the adapter doesn't exist yet
    return f"test -f {R}/{name}/final/adapter_model.safetensors || python train_v2.py --name {name} {args}"
def E(label, sub, suites="mc,mmlu,gsm8k,sciq"):
    return f"python ev.py --label {label} --peft {R}/{sub} --suites {suites}"
def WAIT(*names):                               # wait (max 2 h) for another GPU's adapter
    return "; ".join(f"for i in $(seq 240); do [ -f {R}/{n}/final/adapter_model.safetensors ] && break; sleep 30; done" for n in names)

gpu0 = [T("A", "--mix gsm8k:all --save_steps 25")]
gpu0 += [E(f"A_ckpt{s}", f"A/checkpoint-{s}", "mc,mmlu") for s in [25, 50, 100, 150, 200]]
gpu0 += [E("A_final", "A/final"),
         T("Ds", f"--mix {F1}:3200"), T("Dg", f"--mix {F2}:3200"),
         WAIT("M1_mix"), E("M1_mix_final", "M1_mix/final"),
         E("Ds_final", "Ds/final"), E("Dg_final", "Dg/final"),
         T("S2", f"--init_adapter {R}/A/final --mix gsm8k:1000,{F1}:2400"), E("S2_final", "S2/final"),
         T("S3", f"--init_adapter {R}/S2/final --mix gsm8k:700,{F1}:700,{F2}:2000"), E("S3_final", "S3/final")]

paths = f"{R}/A/final,{R}/Ds/final,{R}/Dg/final"
gpu1 = ["python ev.py --label base --suites mc,mmlu,sciq",
        T("M1_mix", f"--mix gsm8k:all,{F2}:3200 --schedule shuffle"),
        T("M2_interleave", f"--mix gsm8k:all,{F2}:3200 --schedule interleave"),
        E("M2_interleave_final", "M2_interleave/final"),
        T("M3_targeted", f"--mix gsm8k:all,{F1}:3200 --schedule shuffle"),
        E("M3_targeted_final", "M3_targeted/final"),
        WAIT("A", "Ds", "Dg")]
for m, suites in [("am", "mc,mmlu,gsm8k"), ("sum", "mc,mmlu,gsm8k"), ("gm", "mc,mmlu"), ("hm", "mc,mmlu")]:
    gpu1 += [f"test -d results/merge_{m}_{suites.split(',')[-1]} || python merge.py --adapters {paths} --method {m} --out /kaggle/working/merged_{m}",
             f"python ev.py --label merge_{m} --merged /kaggle/working/merged_{m} --suites {suites}",
             f"rm -rf /kaggle/working/merged_{m}"]

def run_queue(gpu, cmds, log):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    with open(f"/kaggle/working/logs/{log}.log", "a") as f:
        for c in cmds:
            f.write(f"\n>>> {c}\n"); f.flush()
            subprocess.run(c, shell=True, env=env, stdout=f, stderr=subprocess.STDOUT)

restore()
ts = [threading.Thread(target=run_queue, args=a) for a in [(0, gpu0, "gpu0"), (1, gpu1, "gpu1")]]
for t in ts: t.start()
last = 0
while any(t.is_alive() for t in ts):
    time.sleep(30)
    if time.time() - last > 300:
        backup(); push("auto-save"); last = time.time(); print(time.strftime("%H:%M"), "saved", flush=True)
backup(); sh("python collect.py"); push("pipeline finished")
for d in glob.glob(f"{R}/*/checkpoint-*"): shutil.rmtree(d, ignore_errors=True)   # keep notebook output small
print("all done")
