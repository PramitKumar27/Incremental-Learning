import random
import struct
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

# Mahdi-style (per-parameter) vs bit-level (per-bit)
BER_MODE = "per_param"   # "per_param" (Mahdi direction) or "per_bit"
BER_VALUE = 1e-6         # start with one point
REPETITIONS = 10         # minimal demo; later go 50–100

# -----------------------------
# 1) Data (same preprocessing)
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

# IMPORTANT: fixed split for reproducibility
g = torch.Generator().manual_seed(SEED)
_, test_set = torch.utils.data.random_split(dataset, [train_size, test_size], generator=g)

test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

# -----------------------------
# 2) Model + checkpoint load
# -----------------------------
model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=10).to(device)

ckpt = torch.load(CKPT_PATH, map_location=device)

# If you saved dict with metadata, load state_dict accordingly:
if isinstance(ckpt, dict) and "state_dict" in ckpt:
    model.load_state_dict(ckpt["state_dict"])
else:
    # If you saved raw state_dict
    model.load_state_dict(ckpt)

model.eval()

# Make an in-memory clean copy (fast restore each repetition)
clean_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

# -----------------------------
# 3) Evaluation function
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
# 4) Fault injection fundamentals
# -----------------------------
def bitflip_fp32_scalar(x_float32: float) -> float:
    """
    Flip ONE random bit in an IEEE-754 float32 scalar.
    Steps:
    1) pack float32 -> 32-bit int
    2) XOR one random bit
    3) unpack back -> float32
    """
    packed = struct.pack("!f", np.float32(x_float32))
    i = struct.unpack("!I", packed)[0]  # uint32
    bit = 1 << random.randint(0, 31)
    i_corrupt = i ^ bit
    packed_corrupt = struct.pack("!I", i_corrupt)
    return struct.unpack("!f", packed_corrupt)[0]

def collect_all_param_tensors(model):
    """
    Mahdi requirement: do not exclude components.
    So we include ALL trainable parameters:
    - weights
    - biases
    - LayerNorm gamma/beta
    etc.
    """
    params = []
    for name, p in model.named_parameters():
        if p is not None and p.requires_grad and p.numel() > 0:
            params.append((name, p))
    return params

param_tensors = collect_all_param_tensors(model)
total_params = sum(p.numel() for _, p in param_tensors)
total_bits_fp32 = total_params * 32  # assuming float32 storage for "per_bit" mode

print(f"Total trainable parameters (elements): {total_params}")
print(f"Total bits assuming FP32: {total_bits_fp32}")

@torch.no_grad()
def inject_n_faults(model, n_faults: int):
    """
    Inject n_faults random bit flips across the entire model.
    For each fault:
    - pick a random parameter tensor
    - pick a random index in it
    - flip 1 random bit in that scalar
    """
    for _ in range(n_faults):
        name, p = random.choice(param_tensors)
        flat = p.view(-1)
        idx = random.randint(0, flat.numel() - 1)

        # Read element as float32 (even if tensor is fp16/bf16), flip bit, write back
        orig_dtype = flat[idx].dtype
        x = float(flat[idx].detach().float().cpu().item())
        x_corrupt = bitflip_fp32_scalar(x)

        flat[idx].copy_(torch.tensor(x_corrupt, device=flat.device, dtype=orig_dtype))

def ber_to_fault_count(ber_value: float, mode: str) -> int:
    """
    Converts BER knob into number of injected faults.
    - per_param (Mahdi-style): faults = BER * number_of_parameters
    - per_bit: faults = BER * number_of_bits (FP32 assumed)
    """
    if ber_value <= 0:
        return 0

    if mode == "per_param":
        return int(ber_value * total_params)
    elif mode == "per_bit":
        return int(ber_value * total_bits_fp32)
    else:
        raise ValueError("BER_MODE must be 'per_param' or 'per_bit'")

# -----------------------------
# 5) BER sweep (multiple points)
# -----------------------------
BERS = [0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
REPETITIONS = 20   # start with 20; later 50–100 for final thesis graphs

def run_ber_sweep(bers, repetitions):
    results = []

    for ber in bers:
        n_faults = ber_to_fault_count(ber, BER_MODE)
        print(f"\n=== BER={ber:.2e} ({BER_MODE}) -> n_faults={n_faults} | reps={repetitions} ===")

        accs = []
        for r in range(repetitions):
            # Restore clean model
            model.load_state_dict(clean_state, strict=True)
            model.to(device)
            model.eval()

            # new random pattern per repetition
            random.seed(SEED + r)
            np.random.seed(SEED + r)
            torch.manual_seed(SEED + r)

            if n_faults > 0:
                inject_n_faults(model, n_faults)

            acc = evaluate_accuracy(model, test_loader)
            accs.append(acc)

        accs = np.array(accs, dtype=np.float64)

        row = {
            "ber": ber,
            "mode": BER_MODE,
            "n_faults": n_faults,
            "baseline_acc": baseline_acc,
            "mean_acc": float(accs.mean()),
            "std_acc": float(accs.std()),
            "mean_drop": float(baseline_acc - accs.mean()),
            "worst_acc": float(accs.min()),
            "worst_drop": float(baseline_acc - accs.min()),
        }
        results.append(row)

        print(f"Mean acc:  {row['mean_acc']:.2f}%   Std: {row['std_acc']:.2f}")
        print(f"Mean drop: {row['mean_drop']:.2f}%   Worst acc: {row['worst_acc']:.2f}%")

    return results

results = run_ber_sweep(BERS, REPETITIONS)

# -----------------------------
# 6) Save results to CSV (for plotting later)
# -----------------------------
import csv
out_path = f"./phase2_results_{BER_MODE}.csv"
with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"\nSaved sweep results to: {out_path}")

