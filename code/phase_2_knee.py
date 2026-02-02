import random
import struct
import csv
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm

# -----------------------------
# 0) Experiment settings
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

CKPT_PATH = "./vit_eurosat_clean.pth"

BER_MODE = "per_param"   # keep Mahdi-style for now
BERS = [0.0,
        1e-7, 1.5e-7, 2e-7,
        3e-7, 5e-7,
        7e-7, 1e-6,
        1.2e-6, 1.5e-6, 1.8e-6, 2e-6,
        3e-6, 5e-6,
        1e-5]
 # knee refinement
REPETITIONS = 12
VULN_REPS = 8   # fewer reps for vulnerability ranking (faster)


# Failure thresholds (you can report both in slides)
FAIL_THRESH = 50.0        # "mission unacceptable"
CATA_THRESH = 20.0        # "catastrophic collapse"

# -----------------------------
# 1) Data (fixed split)
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

dataset = datasets.EuroSAT(root="./data", download=False, transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
g = torch.Generator().manual_seed(SEED)
_, test_set = torch.utils.data.random_split(dataset, [train_size, test_size], generator=g)

SUBSET_N = 1000  # 1000 is a good speed/quality compromise
test_subset = torch.utils.data.Subset(test_set, list(range(SUBSET_N)))
test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)


# -----------------------------
# 2) Model + checkpoint load
# -----------------------------
model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=10).to(device)

ckpt = torch.load(CKPT_PATH, map_location=device)
if isinstance(ckpt, dict) and "state_dict" in ckpt:
    model.load_state_dict(ckpt["state_dict"])
else:
    model.load_state_dict(ckpt)

model.eval()
clean_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

# -----------------------------
# 3) Evaluation
# -----------------------------
@torch.no_grad()
def evaluate_accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        total += y.size(0)
        correct += (pred == y).sum().item()
    return 100.0 * correct / total

baseline_acc = evaluate_accuracy(model, test_loader)
print(f"Baseline (clean) accuracy: {baseline_acc:.2f}%")

# -----------------------------
# 4) Fault model: 1-bit flip in IEEE-754 float32
# -----------------------------
def bitflip_fp32_scalar(x_float32: float) -> float:
    packed = struct.pack("!f", np.float32(x_float32))
    i = struct.unpack("!I", packed)[0]
    bit = 1 << random.randint(0, 31)
    i_corrupt = i ^ bit
    packed_corrupt = struct.pack("!I", i_corrupt)
    return struct.unpack("!f", packed_corrupt)[0]

def collect_all_param_tensors(model):
    params = []
    for name, p in model.named_parameters():
        if p is not None and p.requires_grad and p.numel() > 0:
            params.append((name, p))
    return params

all_params = collect_all_param_tensors(model)
total_params = sum(p.numel() for _, p in all_params)
total_bits_fp32 = total_params * 32

print(f"Total trainable parameters (elements): {total_params}")
print(f"Total bits assuming FP32: {total_bits_fp32}")

@torch.no_grad()
def inject_n_faults(target_params, n_faults: int):
    # Inject n_faults random bit flips into the provided parameter list
    for _ in range(n_faults):
        name, p = random.choice(target_params)
        flat = p.view(-1)
        idx = random.randint(0, flat.numel() - 1)

        orig_dtype = flat[idx].dtype
        x = float(flat[idx].detach().float().cpu().item())
        x_corrupt = bitflip_fp32_scalar(x)
        flat[idx].copy_(torch.tensor(x_corrupt, device=flat.device, dtype=orig_dtype))

def ber_to_fault_count(ber_value: float, mode: str) -> int:
    if ber_value <= 0:
        return 0
    if mode == "per_param":
        return int(ber_value * total_params)
    elif mode == "per_bit":
        return int(ber_value * total_bits_fp32)
    else:
        raise ValueError("BER_MODE must be 'per_param' or 'per_bit'")

# -----------------------------
# 5) Vulnerability groups (for ViT-tiny timm naming)
# -----------------------------
def build_groups(param_list):
    groups = {
        "ALL": [],
        "PATCH_EMBED": [],
        "ATTENTION": [],
        "MLP": [],
        "LAYERNORM": [],
        "HEAD": [],
    }

    for name, p in param_list:
        groups["ALL"].append((name, p))

        # patch embedding
        if name.startswith("patch_embed"):
            groups["PATCH_EMBED"].append((name, p))

        # attention (qkv/proj)
        if ".attn." in name:
            groups["ATTENTION"].append((name, p))

        # mlp (fc1/fc2)
        if ".mlp." in name:
            groups["MLP"].append((name, p))

        # layernorms in ViT: norm, blocks.*.norm1/norm2
        if "norm" in name:
            groups["LAYERNORM"].append((name, p))

        # classifier head
        if name.startswith("head"):
            groups["HEAD"].append((name, p))

    # Print sizes (helps validate grouping)
    for k, v in groups.items():
        count = sum(p.numel() for _, p in v)
        print(f"Group {k:10s}: tensors={len(v):4d}, elements={count}")
    return groups

groups = build_groups(all_params)

# -----------------------------
# 6) Core runner: for one (ber, group) compute mean/std/worst + failure rates
# -----------------------------
def run_point(ber, group_name, target_params, repetitions):
    n_faults = ber_to_fault_count(ber, BER_MODE)

    accs = []
    for r in range(repetitions):
        # restore clean
        model.load_state_dict(clean_state, strict=True)
        model.to(device)
        model.eval()

        # new random pattern per repetition
        random.seed(SEED + r)
        np.random.seed(SEED + r)
        torch.manual_seed(SEED + r)

        if n_faults > 0 and len(target_params) > 0:
            inject_n_faults(target_params, n_faults)

        acc = evaluate_accuracy(model, test_loader)
        accs.append(acc)

    accs = np.array(accs, dtype=np.float64)

    fail_rate = float(np.mean(accs < FAIL_THRESH))
    cata_rate = float(np.mean(accs < CATA_THRESH))

    return {
        "group": group_name,
        "ber": ber,
        "mode": BER_MODE,
        "n_faults": n_faults,
        "baseline_acc": baseline_acc,
        "mean_acc": float(accs.mean()),
        "std_acc": float(accs.std()),
        "worst_acc": float(accs.min()),
        "mean_drop": float(baseline_acc - accs.mean()),
        "worst_drop": float(baseline_acc - accs.min()),
        "fail_rate(<50%)": fail_rate,
        "cata_rate(<20%)": cata_rate,
    }

# -----------------------------
# 7) STEP 1 + 2: Knee-region BER sweep on ALL parameters + failure probability
# -----------------------------
print("\n=== STEP 1+2: Knee-region BER sweep (ALL) ===")
sweep_results = []
for ber in BERS:
    row = run_point(ber, "ALL", groups["ALL"], REPETITIONS)
    sweep_results.append(row)
    print(f"BER={ber:.1e} | faults={row['n_faults']:5d} | mean={row['mean_acc']:.2f}% "
          f"std={row['std_acc']:.2f} | worst={row['worst_acc']:.2f}% "
          f"| P(acc<50)={row['fail_rate(<50%)']:.2f} P(acc<20)={row['cata_rate(<20%)']:.2f}")

# Save sweep CSV
sweep_csv = f"./phase2_knee_sweep_{BER_MODE}.csv"
with open(sweep_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=sweep_results[0].keys())
    writer.writeheader()
    writer.writerows(sweep_results)
print(f"Saved: {sweep_csv}")

# -----------------------------
# 8) STEP 3: Vulnerability ranking at ONE representative BER in the knee
# -----------------------------
# Pick a BER that shows variance (you can change later after seeing sweep)
VULN_BER = 1e-6

print(f"\n=== STEP 3: Vulnerability ranking at BER={VULN_BER} (same faults/run, different target region) ===")
vuln_groups = ["PATCH_EMBED", "ATTENTION", "MLP", "LAYERNORM", "HEAD"]
vuln_results = []
for gname in vuln_groups:
    row = run_point(VULN_BER, gname, groups[gname], VULN_REPS)
    vuln_results.append(row)
    print(f"{gname:10s} | mean={row['mean_acc']:.2f}% std={row['std_acc']:.2f} "
          f"| worst={row['worst_acc']:.2f}% | P(acc<50)={row['fail_rate(<50%)']:.2f}")

# Save vulnerability CSV
vuln_csv = f"./phase2_vulnerability_{BER_MODE}_ber{VULN_BER:.0e}.csv"
with open(vuln_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=vuln_results[0].keys())
    writer.writeheader()
    writer.writerows(vuln_results)
print(f"Saved: {vuln_csv}")
