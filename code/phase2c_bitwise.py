"""
Phase 2C: Bit-Wise Statistical Fault Injection

Complete 32-bit analysis to validate bit-30 hypothesis across all parameters.
Uses optimized parameters for reasonable runtime.

Based on baseline: bit-30 causes 96% of failures (86% failure rate)
"""

import random
import struct
import csv
import json
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import timm

from statistical_utils import (
    calculate_sample_size,
    calculate_error_margin,
)

# ========================================
# Configuration
# ========================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

CKPT_PATH = "./vit_eurosat_clean.pth"
OUTPUT_DIR = Path("./statistical_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Statistical Parameters - Optimized for bit analysis
MARGIN_OF_ERROR = 0.03  # 3% margin (reasonable for bit ranking)
CONFIDENCE_LEVEL = 0.95
# Use different p for different bit types
P_SAFE_BITS = 0.001   # Bits 0-22 expected safe
P_RISKY_BITS = 0.05   # Bits 23-29 somewhat risky
P_CRITICAL_BIT = 0.86  # Bit 30 from baseline

FAIL_THRESH = 50.0
CATA_THRESH = 20.0
SUBSET_N = 1000

print("="*70)
print("PHASE 2C: BIT-WISE STATISTICAL FAULT INJECTION")
print("="*70)
print(f"Analyzing all 32 bits in FP32 representation")
print(f"Margin of Error: {MARGIN_OF_ERROR*100}%")
print(f"Confidence Level: {CONFIDENCE_LEVEL*100}%")
print(f"Expected total time: ~6-8 hours")
print("="*70)

# ========================================
# Data & Model Loading
# ========================================
print("\n[1] Loading dataset and model...")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = datasets.EuroSAT(root="./data", download=False, transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
g = torch.Generator().manual_seed(SEED)
_, test_set = torch.utils.data.random_split(dataset, [train_size, test_size], generator=g)
test_subset = torch.utils.data.Subset(test_set, list(range(SUBSET_N)))
test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)

model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=10).to(device)
ckpt = torch.load(CKPT_PATH, map_location=device)
if isinstance(ckpt, dict) and "state_dict" in ckpt:
    model.load_state_dict(ckpt["state_dict"])
else:
    model.load_state_dict(ckpt)
model.eval()
clean_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

# Get all parameters
all_params = [(name, p) for name, p in model.named_parameters() 
              if p is not None and p.requires_grad and p.numel() > 0]
total_params = sum(p.numel() for _, p in all_params)

print(f"Total parameters: {total_params:,}")

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
# Bit-Specific Fault Injector
# ========================================
class BitFaultInjector:
    """Fault injector that only flips a specific bit position."""
    
    def __init__(self, bit_idx, param_list, test_loader, device, baseline_acc):
        self.bit_idx = bit_idx
        self.param_list = param_list
        self.test_loader = test_loader
        self.device = device
        self.baseline_acc = baseline_acc
        self.injected_faults = set()
        
        # Population is all parameters (each can have this bit flipped)
        self.N = sum(p.numel() for _, p in param_list)
    
    def generate_fault_id(self):
        """Generate fault in random parameter, but always the same bit."""
        param_idx = random.randint(0, len(self.param_list) - 1)
        name, p = self.param_list[param_idx]
        elem_idx = random.randint(0, p.numel() - 1)
        fault_id = f"{param_idx}_{elem_idx}_{self.bit_idx}"
        return fault_id, param_idx, elem_idx
    
    def inject_fault(self, param_idx, elem_idx):
        """Inject fault at specific bit position."""
        name, p = self.param_list[param_idx]
        flat = p.view(-1)
        x = float(flat[elem_idx].item())
        
        # Flip the specific bit
        packed = struct.pack("!f", np.float32(x))
        i = struct.unpack("!I", packed)[0]
        bit = 1 << self.bit_idx
        i_corrupt = i ^ bit
        packed_corrupt = struct.pack("!I", i_corrupt)
        x_corrupt = struct.unpack("!f", packed_corrupt)[0]
        
        with torch.no_grad():
            flat.data[elem_idx] = x_corrupt
    
    @torch.no_grad()
    def run_campaign(self, model, clean_state, n_faults):
        results = []
        
        for i in range(n_faults):
            attempts = 0
            while attempts < 100:
                fault_id, param_idx, elem_idx = self.generate_fault_id()
                if fault_id not in self.injected_faults:
                    break
                attempts += 1
            
            if attempts >= 100:
                continue
            
            self.injected_faults.add(fault_id)
            
            # Restore clean model
            model.load_state_dict(clean_state, strict=True)
            model.to(self.device)
            model.eval()
            
            # Inject fault
            self.inject_fault(param_idx, elem_idx)
            
            # Evaluate
            try:
                acc = evaluate_accuracy(model, self.test_loader)
                
                has_nan = any(torch.isnan(p).any() for _, p in model.named_parameters())
                has_inf = any(torch.isinf(p).any() for _, p in model.named_parameters())
                
                if has_nan or has_inf:
                    failure_type = "catastrophic_nan_inf"
                elif acc < CATA_THRESH:
                    failure_type = "catastrophic_low_acc"
                elif acc < FAIL_THRESH:
                    failure_type = "mission_failure"
                elif acc < self.baseline_acc - 5.0:
                    failure_type = "sdc_significant"
                elif acc < self.baseline_acc:
                    failure_type = "sdc_minor"
                else:
                    failure_type = "no_effect"
                
                results.append({
                    "fault_id": fault_id,
                    "param_idx": param_idx,
                    "elem_idx": elem_idx,
                    "bit_idx": self.bit_idx,
                    "accuracy": acc,
                    "has_nan": has_nan,
                    "has_inf": has_inf,
                    "failure_type": failure_type,
                })
            except Exception:
                results.append({
                    "fault_id": fault_id,
                    "param_idx": param_idx,
                    "elem_idx": elem_idx,
                    "bit_idx": self.bit_idx,
                    "accuracy": 0.0,
                    "has_nan": True,
                    "has_inf": False,
                    "failure_type": "error",
                })
        
        return results

# ========================================
# Run Bit-Wise Campaigns
# ========================================
print("\n" + "="*70)
print("RUNNING BIT-WISE CAMPAIGNS (32 bits)")
print("="*70)

campaign_start = datetime.now()
all_bit_results = {}
bit_summary = []

for bit_idx in range(32):
    print(f"\n{'='*70}")
    print(f"BIT: {bit_idx}")
    
    # Determine bit type for appropriate p value
    if bit_idx < 23:
        p_est = P_SAFE_BITS
        bit_type = "mantissa"
    elif bit_idx < 30:
        p_est = P_RISKY_BITS
        bit_type = "exponent"
    elif bit_idx == 30:
        p_est = P_CRITICAL_BIT
        bit_type = "exponent_MSB"
    else:
        p_est = P_SAFE_BITS  # bit 31 showed 0% in baseline
        bit_type = "sign"
    
    print(f"Type: {bit_type}")
    print(f"{'='*70}")
    
    injector = BitFaultInjector(bit_idx, all_params, test_loader, device, baseline_acc)
    
    # Calculate sample size with appropriate p
    n_required = calculate_sample_size(
        N=injector.N,
        e=MARGIN_OF_ERROR,
        confidence_level=CONFIDENCE_LEVEL,
        p=p_est
    )
    
    # Cap at reasonable maximum for very high p values
    n_required = min(n_required, 2000)
    
    print(f"Population (N): {injector.N:,}")
    print(f"Required sample size: {n_required:,}")
    print(f"Estimated time: {n_required * 7 / 60:.1f} minutes")
    
    bit_start = datetime.now()
    results = injector.run_campaign(model, clean_state, n_required)
    bit_duration = (datetime.now() - bit_start).total_seconds()
    
    # Analyze
    failure_counts = {
        "catastrophic_nan_inf": sum(1 for r in results if r["failure_type"] == "catastrophic_nan_inf"),
        "catastrophic_low_acc": sum(1 for r in results if r["failure_type"] == "catastrophic_low_acc"),
        "mission_failure": sum(1 for r in results if r["failure_type"] == "mission_failure"),
        "sdc_significant": sum(1 for r in results if r["failure_type"] == "sdc_significant"),
        "sdc_minor": sum(1 for r in results if r["failure_type"] == "sdc_minor"),
        "no_effect": sum(1 for r in results if r["failure_type"] == "no_effect"),
    }
    
    critical_failures = (
        failure_counts["catastrophic_nan_inf"] +
        failure_counts["catastrophic_low_acc"] +
        failure_counts["mission_failure"] +
        failure_counts["sdc_significant"]
    )
    
    p_hat = critical_failures / len(results) if results else 0
    e_hat = calculate_error_margin(len(results), injector.N, p_hat, CONFIDENCE_LEVEL)
    ci_lower = max(0, p_hat - e_hat)
    ci_upper = min(1, p_hat + e_hat)
    
    print(f"\nResults for bit {bit_idx}:")
    print(f"  Tests: {len(results)}")
    print(f"  Failures: {critical_failures}")
    print(f"  Failure rate: {p_hat*100:.2f}% ± {e_hat*100:.2f}%")
    print(f"  CI: [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]")
    print(f"  Duration: {bit_duration:.1f} sec ({bit_duration/60:.1f} min)")
    
    all_bit_results[bit_idx] = results
    bit_summary.append({
        "bit_idx": bit_idx,
        "bit_type": bit_type,
        "population_size": injector.N,
        "sample_size": len(results),
        "failures": critical_failures,
        "failure_rate": p_hat,
        "margin_error": e_hat,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "duration_seconds": bit_duration,
        "failure_counts": failure_counts
    })

campaign_duration = (datetime.now() - campaign_start).total_seconds()

# ========================================
# Generate Bit Ranking
# ========================================
print("\n" + "="*70)
print("BIT VULNERABILITY RANKING")
print("="*70)

ranked_bits = sorted(bit_summary, key=lambda x: x["failure_rate"], reverse=True)

print(f"\n{'Rank':<6} {'Bit':>4} {'Type':<15} {'Rate':>15} {'95% CI':>25} {'Tests':>8}")
print("-" * 80)
for rank, bit in enumerate(ranked_bits, 1):
    print(f"{rank:<6} {bit['bit_idx']:>4} {bit['bit_type']:<15} "
          f"{bit['failure_rate']*100:>6.2f}% ± {bit['margin_error']*100:>4.2f}% "
          f"[{bit['ci_lower']*100:>5.2f}%, {bit['ci_upper']*100:>5.2f}%] "
          f"{bit['sample_size']:>8,}")

# ========================================
# Bit Range Summary
# ========================================
print("\n" + "="*70)
print("BIT RANGE SUMMARY")
print("="*70)

bit_ranges = {
    "Mantissa (0-19)": list(range(0, 20)),
    "Mantissa (20-22)": list(range(20, 23)),
    "Exponent (23-29)": list(range(23, 30)),
    "Bit 30 (Exp MSB)": [30],
    "Bit 31 (Sign)": [31],
}

for range_name, bits in bit_ranges.items():
    range_bits = [b for b in bit_summary if b["bit_idx"] in bits]
    if range_bits:
        total_tests = sum(b["sample_size"] for b in range_bits)
        total_failures = sum(b["failures"] for b in range_bits)
        avg_rate = (total_failures / total_tests * 100) if total_tests > 0 else 0
        print(f"\n{range_name}:")
        print(f"  Total tests: {total_tests:,}")
        print(f"  Total failures: {total_failures}")
        print(f"  Aggregate failure rate: {avg_rate:.2f}%")

# ========================================
# Save Results
# ========================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

summary_file = OUTPUT_DIR / f"bitwise_summary_{timestamp}.json"
with open(summary_file, "w") as f:
    json.dump({
        "experiment": "bitwise_analysis",
        "timestamp": timestamp,
        "configuration": {
            "margin_of_error": MARGIN_OF_ERROR,
            "confidence_level": CONFIDENCE_LEVEL,
            "baseline_accuracy": baseline_acc,
        },
        "total_duration_seconds": campaign_duration,
        "bit_summary": bit_summary,
        "ranked_bits": [
            {"rank": i+1, "bit": b["bit_idx"], "failure_rate": b["failure_rate"]}
            for i, b in enumerate(ranked_bits)
        ],
        "bit_range_summary": {
            name: {
                "bits": bits,
                "total_tests": sum(b["sample_size"] for b in bit_summary if b["bit_idx"] in bits),
                "total_failures": sum(b["failures"] for b in bit_summary if b["bit_idx"] in bits),
            }
            for name, bits in bit_ranges.items()
        }
    }, f, indent=2)
print(f"\nSummary saved: {summary_file}")

# Save per-bit details
for bit_idx, results in all_bit_results.items():
    if results:
        results_file = OUTPUT_DIR / f"bitwise_bit{bit_idx:02d}_{timestamp}.csv"
        with open(results_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

print("\n" + "="*70)
print("BIT-WISE ANALYSIS COMPLETE")
print("="*70)
print(f"\nTotal duration: {campaign_duration:.1f} sec ({campaign_duration/3600:.2f} hours)")
print(f"\nTop 5 Most Vulnerable Bits:")
for i, bit in enumerate(ranked_bits[:5], 1):
    print(f"  {i}. Bit {bit['bit_idx']:2d} ({bit['bit_type']:<15}): {bit['failure_rate']*100:>6.2f}% ± {bit['margin_error']*100:>4.2f}%")

print(f"\nBottom 5 Most Robust Bits:")
for i, bit in enumerate(ranked_bits[-5:], 1):
    print(f"  {i}. Bit {bit['bit_idx']:2d} ({bit['bit_type']:<15}): {bit['failure_rate']*100:>6.2f}% ± {bit['margin_error']*100:>4.2f}%")
