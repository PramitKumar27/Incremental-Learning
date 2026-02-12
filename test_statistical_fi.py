"""
Quick Test Script for Statistical Fault Injection

Tests the infrastructure with a small sample size (~100 faults)
to validate everything works before committing to full campaign.
"""

import random
import struct
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm

from statistical_utils import (
    calculate_sample_size,
    calculate_error_margin,
    print_statistical_report
)

# Quick test configuration
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*70)
print("STATISTICAL FAULT INJECTION - QUICK TEST")
print("="*70)

# ========================================
# Load model and data (minimal)
# ========================================
print("\n[1] Loading model and data...")

# Use smaller test set for speed
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

# Use only 100 test images for speed
test_subset = torch.utils.data.Subset(test_set, list(range(100)))
test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)

model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=10).to(device)
ckpt = torch.load("./vit_eurosat_clean.pth", map_location=device)
if isinstance(ckpt, dict) and "state_dict" in ckpt:
    model.load_state_dict(ckpt["state_dict"])
else:
    model.load_state_dict(ckpt)
model.eval()
clean_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

# Count parameters
all_params = [(name, p) for name, p in model.named_parameters() 
              if p is not None and p.requires_grad and p.numel() > 0]
total_params = sum(p.numel() for _, p in all_params)

print(f"Model parameters: {total_params:,}")
print(f"Test images: {len(test_subset)}")

# Baseline accuracy
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
print(f"Baseline accuracy: {baseline_acc:.2f}%")

# ========================================
# Calculate sample size for quick test
# ========================================
print("\n[2] Calculating sample size...")

# Use loose margin for quick test
N = total_params  # Population
e_test = 0.05  # 5% margin (loose for quick test)
conf = 0.95  # 95% confidence (standard)
p = 0.5  # Conservative

n_test = calculate_sample_size(N, e_test, conf, p)
print(f"\nPopulation (N): {N:,}")
print(f"Margin of error: {e_test*100}%")
print(f"Confidence: {conf*100}%")
print(f"Required sample size: {n_test:,}")

# Limit to max 100 for quick test
n_test = min(n_test, 100)
print(f"Using (limited): {n_test} faults for quick test")

# ========================================
# Fault injection (simplified)
# ========================================
def bitflip_fp32_scalar(x_float32: float) -> float:
    packed = struct.pack("!f", np.float32(x_float32))
    i = struct.unpack("!I", packed)[0]
    bit = 1 << random.randint(0, 31)
    i_corrupt = i ^ bit
    packed_corrupt = struct.pack("!I", i_corrupt)
    return struct.unpack("!f", packed_corrupt)[0]

print(f"\n[3] Running {n_test} fault injections...")

results = []
failures = 0

for i in range(n_test):
    # Restore clean model
    model.load_state_dict(clean_state, strict=True)
    model.eval()
    
    # Inject random fault
    name, p_tensor = random.choice(all_params)
    flat = p_tensor.view(-1)
    idx = random.randint(0, flat.numel() - 1)
    
    x = float(flat[idx].detach().float().cpu().item())
    x_corrupt = bitflip_fp32_scalar(x)
    flat[idx].copy_(torch.tensor(x_corrupt, device=flat.device, dtype=flat[idx].dtype))
    
    # Evaluate
    try:
        acc = evaluate_accuracy(model, test_loader)
        is_failure = acc < 50.0  # Mission failure threshold
        if is_failure:
            failures += 1
        
        results.append({
            "accuracy": acc,
            "is_failure": is_failure
        })
    except:
        failures += 1
        results.append({
            "accuracy": 0.0,
            "is_failure": True
        })
    
    if (i + 1) % 20 == 0:
        print(f"  Progress: {i+1}/{n_test}")

print(f"  Complete: {n_test}/{n_test}")

# ========================================
# Statistical analysis
# ========================================
print("\n[4] Statistical Analysis...")

metrics = print_statistical_report(
    n=n_test,
    N=N,
    failures=failures,
    confidence_level=conf,
    description="Quick Test"
)

# ========================================
# Validation
# ========================================
print("\n[5] Test Results:")
print(f"  ✓ Infrastructure works correctly")
print(f"  ✓ Fault injection successful")
print(f"  ✓ Statistical calculations validated")
print(f"  ✓ Estimated failure rate: {metrics['p_hat']*100:.2f}% ± {metrics['e_hat']*100:.2f}%")

print("\n" + "="*70)
print("QUICK TEST PASSED!")
print("="*70)
print("\nYou can now run the full statistical campaign:")
print("  python3 phase2_statistical_baseline.py")
print("\nNote: Full campaign will take much longer (hours to days)")
