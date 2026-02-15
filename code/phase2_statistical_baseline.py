"""
Phase 2 Statistical Fault Injection - Conservative Baseline

Conservative one-step approach using p=0.5 (no prior knowledge).
This serves as our baseline to validate the statistical formulas.

Based on:
- Leveugle et al. (DATE 2009): Statistical FI framework
- Your previous phase_2_knee.py for fault model and evaluation
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

# Import our statistical utilities
from statistical_utils import (
    calculate_sample_size,
    calculate_error_margin,
    print_statistical_report,
    validate_statistical_assumptions
)

# ========================================
# 0) Experiment Configuration
# ========================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
CKPT_PATH = "./vit_eurosat_clean.pth"
OUTPUT_DIR = Path("./statistical_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Statistical Parameters
MARGIN_OF_ERROR = 0.001  # 0.1% margin
CONFIDENCE_LEVEL = 0.99  # 99% confidence
FAILURE_PROB = 0.5  # Conservative: assume 50% (worst case)

# Failure Thresholds
FAIL_THRESH = 50.0  # Mission failure
CATA_THRESH = 20.0  # Catastrophic failure

# Test Configuration
SUBSET_N = 1000  # Test set size (kept same as your previous work)

print("="*70)
print("PHASE 2: STATISTICAL FAULT INJECTION - CONSERVATIVE BASELINE")
print("="*70)
print(f"Margin of Error: {MARGIN_OF_ERROR*100}%")
print(f"Confidence Level: {CONFIDENCE_LEVEL*100}%")
print(f"Failure Probability (p): {FAILURE_PROB} (conservative)")
print("="*70)

# ========================================
# 1) Data Loading
# ========================================
print("\n[1] Loading EuroSAT dataset...")
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

test_subset = torch.utils.data.Subset(test_set, list(range(SUBSET_N)))
test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)
print(f"Test subset size: {SUBSET_N} images")

# ========================================
# 2) Model Loading
# ========================================
print("\n[2] Loading ViT-tiny model...")
model = timm.create_model("vit_tiny_patch16_224", pretrained=False, num_classes=10).to(device)

ckpt = torch.load(CKPT_PATH, map_location=device)
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
total_bits = total_params * 32

print(f"Total trainable parameters: {total_params:,}")
print(f"Total bits (FP32): {total_bits:,}")

# ========================================
# 3) Baseline Evaluation
# ========================================
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

print("\n[3] Evaluating baseline accuracy...")
baseline_acc = evaluate_accuracy(model, test_loader)
print(f"Baseline (clean) accuracy: {baseline_acc:.2f}%")

# ========================================
# 4) Fault Injection Functions
# ========================================
def bitflip_fp32_scalar(x_float32: float) -> float:
    """Flip one random bit in FP32 representation."""
    packed = struct.pack("!f", np.float32(x_float32))
    i = struct.unpack("!I", packed)[0]
    bit = 1 << random.randint(0, 31)
    i_corrupt = i ^ bit
    packed_corrupt = struct.pack("!I", i_corrupt)
    return struct.unpack("!f", packed_corrupt)[0]

class FaultInjector:
    """Manages fault injection with sampling without replacement."""
    
    def __init__(self, param_list, test_loader, device):
        self.param_list = param_list
        self.test_loader = test_loader
        self.device = device
        self.injected_faults = set()  # Track injected faults to avoid duplicates
        
        # Calculate population size
        self.total_params = sum(p.numel() for _, p in param_list)
        self.N = self.total_params  # For now, assume single time point
        
        print(f"\nFault Injector initialized:")
        print(f"  Parameters: {self.total_params:,}")
        print(f"  Population size (N): {self.N:,}")
    
    def generate_fault_id(self) -> str:
        """Generate unique fault identifier."""
        # Choose random parameter
        param_idx = random.randint(0, len(self.param_list) - 1)
        name, p = self.param_list[param_idx]
        
        # Choose random element in that parameter
        elem_idx = random.randint(0, p.numel() - 1)
        
        # Choose random bit
        bit_idx = random.randint(0, 31)
        
        # Create unique ID
        fault_id = f"{param_idx}_{elem_idx}_{bit_idx}"
        return fault_id, param_idx, elem_idx, bit_idx
    
    def inject_fault(self, param_idx: int, elem_idx: int, bit_idx: int):
        """Inject specific fault into model."""
        name, p = self.param_list[param_idx]
        flat = p.view(-1)
        
        # Get current value
        x = float(flat[elem_idx].item())
        
        # Flip specific bit
        packed = struct.pack("!f", np.float32(x))
        i = struct.unpack("!I", packed)[0]
        bit = 1 << bit_idx
        i_corrupt = i ^ bit
        packed_corrupt = struct.pack("!I", i_corrupt)
        x_corrupt = struct.unpack("!f", packed_corrupt)[0]
        
        # Update model (use .data to avoid gradient tracking)
        with torch.no_grad():
            flat.data[elem_idx] = x_corrupt
    
    @torch.no_grad()
    def run_fault_injection(self, model, clean_state, n_faults: int):
        """
        Run fault injection campaign with n faults.
        Sampling without replacement.
        
        Returns:
            List of results for each fault
        """
        results = []
        
        print(f"\nRunning fault injection campaign...")
        print(f"  Target: {n_faults:,} faults")
        print(f"  Sampling: without replacement")
        
        for i in range(n_faults):
            # Generate unique fault
            attempts = 0
            while attempts < 100:  # Avoid infinite loop
                fault_id, param_idx, elem_idx, bit_idx = self.generate_fault_id()
                if fault_id not in self.injected_faults:
                    break
                attempts += 1
            
            if attempts >= 100:
                print(f"Warning: Could not find unique fault after 100 attempts at iteration {i}")
                continue
            
            self.injected_faults.add(fault_id)
            
            # Restore clean model
            model.load_state_dict(clean_state, strict=True)
            model.to(self.device)
            model.eval()
            
            # Inject fault
            self.inject_fault(param_idx, elem_idx, bit_idx)
            
            # Evaluate
            try:
                acc = evaluate_accuracy(model, self.test_loader)
                
                # Check for NaN/Inf
                has_nan = False
                has_inf = False
                for name, p in model.named_parameters():
                    if torch.isnan(p).any():
                        has_nan = True
                    if torch.isinf(p).any():
                        has_inf = True
                
                # Classify failure type
                if has_nan or has_inf:
                    failure_type = "catastrophic_nan_inf"
                elif acc < CATA_THRESH:
                    failure_type = "catastrophic_low_acc"
                elif acc < FAIL_THRESH:
                    failure_type = "mission_failure"
                elif acc < baseline_acc - 5.0:  # Significant degradation
                    failure_type = "sdc_significant"
                elif acc < baseline_acc:
                    failure_type = "sdc_minor"
                else:
                    failure_type = "no_effect"
                
                results.append({
                    "fault_id": fault_id,
                    "param_idx": param_idx,
                    "elem_idx": elem_idx,
                    "bit_idx": bit_idx,
                    "accuracy": acc,
                    "has_nan": has_nan,
                    "has_inf": has_inf,
                    "failure_type": failure_type,
                })
                
            except Exception as e:
                print(f"  Error at fault {i}: {e}")
                results.append({
                    "fault_id": fault_id,
                    "param_idx": param_idx,
                    "elem_idx": elem_idx,
                    "bit_idx": bit_idx,
                    "accuracy": 0.0,
                    "has_nan": True,
                    "has_inf": False,
                    "failure_type": "error",
                })
            
            # Progress reporting
            if (i + 1) % 1000 == 0 or (i + 1) == n_faults:
                print(f"  Progress: {i+1:,}/{n_faults:,} ({100*(i+1)/n_faults:.1f}%)")
        
        return results

# ========================================
# 5) Statistical Campaign
# ========================================
print("\n" + "="*70)
print("CALCULATING REQUIRED SAMPLE SIZE")
print("="*70)

injector = FaultInjector(all_params, test_loader, device)

# Calculate required sample size using conservative p=0.5
n_required = calculate_sample_size(
    N=injector.N,
    e=MARGIN_OF_ERROR,
    confidence_level=CONFIDENCE_LEVEL,
    p=FAILURE_PROB
)

print(f"\nRequired sample size: {n_required:,}")
print(f"Percentage of population: {100*n_required/injector.N:.4f}%")
print(f"Expected time (1 sec/test): {n_required/3600:.2f} hours = {n_required/(3600*24):.2f} days")

# Ask for confirmation if very large
if n_required > 100000:
    print(f"\n{'!'*70}")
    print(f"WARNING: This will require {n_required:,} fault injections!")
    print(f"{'!'*70}")
    response = input("Continue? (yes/no): ")
    if response.lower() != "yes":
        print("Aborting.")
        exit()

# ========================================
# 6) Run Campaign
# ========================================
print("\n" + "="*70)
print("RUNNING CONSERVATIVE STATISTICAL FAULT INJECTION")
print("="*70)

start_time = datetime.now()
print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

# Run fault injection
results = injector.run_fault_injection(model, clean_state, n_required)

end_time = datetime.now()
duration = (end_time - start_time).total_seconds()

print(f"\nEnd time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total duration: {duration:.2f} seconds = {duration/3600:.2f} hours")

# ========================================
# 7) Analyze Results
# ========================================
print("\n" + "="*70)
print("STATISTICAL ANALYSIS")
print("="*70)

# Count different failure types
failure_counts = {
    "catastrophic_nan_inf": 0,
    "catastrophic_low_acc": 0,
    "mission_failure": 0,
    "sdc_significant": 0,
    "sdc_minor": 0,
    "no_effect": 0,
}

for r in results:
    failure_counts[r["failure_type"]] += 1

# Define "failure" as anything worse than minor SDC
critical_failures = (
    failure_counts["catastrophic_nan_inf"] +
    failure_counts["catastrophic_low_acc"] +
    failure_counts["mission_failure"] +
    failure_counts["sdc_significant"]
)

# Generate statistical report
metrics = print_statistical_report(
    n=len(results),
    N=injector.N,
    failures=critical_failures,
    confidence_level=CONFIDENCE_LEVEL,
    description="Conservative Baseline (p=0.5)"
)

# Validate assumptions
validate_statistical_assumptions(len(results), critical_failures)

# Print failure breakdown
print(f"\nFailure Type Breakdown:")
print(f"  Catastrophic (NaN/Inf):  {failure_counts['catastrophic_nan_inf']:6d} ({100*failure_counts['catastrophic_nan_inf']/len(results):.2f}%)")
print(f"  Catastrophic (Low Acc):  {failure_counts['catastrophic_low_acc']:6d} ({100*failure_counts['catastrophic_low_acc']/len(results):.2f}%)")
print(f"  Mission Failure:         {failure_counts['mission_failure']:6d} ({100*failure_counts['mission_failure']/len(results):.2f}%)")
print(f"  SDC (Significant):       {failure_counts['sdc_significant']:6d} ({100*failure_counts['sdc_significant']/len(results):.2f}%)")
print(f"  SDC (Minor):             {failure_counts['sdc_minor']:6d} ({100*failure_counts['sdc_minor']/len(results):.2f}%)")
print(f"  No Effect:               {failure_counts['no_effect']:6d} ({100*failure_counts['no_effect']/len(results):.2f}%)")

# ========================================
# 8) Save Results
# ========================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save detailed results
results_file = OUTPUT_DIR / f"baseline_results_{timestamp}.csv"
with open(results_file, "w", newline="") as f:
    if results:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
print(f"\nDetailed results saved: {results_file}")

# Save summary
summary = {
    "experiment": "conservative_baseline",
    "timestamp": timestamp,
    "configuration": {
        "margin_of_error": MARGIN_OF_ERROR,
        "confidence_level": CONFIDENCE_LEVEL,
        "failure_probability": FAILURE_PROB,
        "population_size": injector.N,
        "sample_size": len(results),
        "baseline_accuracy": baseline_acc,
    },
    "metrics": metrics,
    "failure_counts": failure_counts,
    "duration_seconds": duration,
}

summary_file = OUTPUT_DIR / f"baseline_summary_{timestamp}.json"
with open(summary_file, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Summary saved: {summary_file}")

print("\n" + "="*70)
print("BASELINE STATISTICAL FAULT INJECTION COMPLETE")
print("="*70)
print(f"\nKey Result:")
print(f"  Estimated failure rate: {metrics['p_hat']*100:.4f}% ± {metrics['e_hat']*100:.4f}%")
print(f"  With {CONFIDENCE_LEVEL*100:.0f}% confidence")
print(f"  After testing {len(results):,} faults ({100*len(results)/injector.N:.4f}% of population)")