"""
Phase 2C: Component-Wise Statistical Fault Injection

Analyzes vulnerability at component granularity within critical layers.
Focuses on: attention, MLP, layer_norm components in vulnerable layers.

Based on layer-wise results:
- norm layer: 11.11% (CRITICAL)
- blocks 2-5: 3.57-4.29% (HIGH)
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

# Statistical Parameters - Balanced for component analysis
MARGIN_OF_ERROR = 0.03  # 3% margin (reasonable for component ranking)
CONFIDENCE_LEVEL = 0.95
FAILURE_PROB = 0.04  # Use slightly higher than baseline due to vulnerable layers

FAIL_THRESH = 50.0
CATA_THRESH = 20.0
SUBSET_N = 1000

# Focus on vulnerable layers identified from layer-wise analysis
FOCUS_LAYERS = ["norm", "blocks.2", "blocks.3", "blocks.4", "blocks.5"]

print("="*70)
print("PHASE 2C: COMPONENT-WISE STATISTICAL FAULT INJECTION")
print("="*70)
print(f"Margin of Error: {MARGIN_OF_ERROR*100}%")
print(f"Confidence Level: {CONFIDENCE_LEVEL*100}%")
print(f"Focus layers: {', '.join(FOCUS_LAYERS)}")
print(f"Estimated time: ~4-5 hours")
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
# Component Grouping
# ========================================
print("\n[2] Organizing parameters by component...")

def group_by_component(model):
    """
    Group parameters by functional component.
    
    For transformer blocks:
    - attn_qkv: Query, Key, Value projections
    - attn_proj: Attention output projection  
    - mlp_fc1: First MLP layer
    - mlp_fc2: Second MLP layer
    - norm1: Pre-attention layer norm
    - norm2: Pre-MLP layer norm
    """
    components = {}
    
    for name, p in model.named_parameters():
        if p is None or not p.requires_grad or p.numel() == 0:
            continue
        
        # Norm layer (final)
        if name == "norm.weight" or name == "norm.bias":
            comp = "final_norm"
        
        # Transformer blocks
        elif name.startswith("blocks."):
            parts = name.split(".")
            block_num = int(parts[1])
            
            # Only process focus blocks
            focus_block = f"blocks.{block_num}"
            if focus_block not in FOCUS_LAYERS:
                continue
            
            comp_key = f"block_{block_num}_"
            
            if "attn.qkv" in name:
                comp_key += "attn_qkv"
            elif "attn.proj" in name:
                comp_key += "attn_proj"
            elif "mlp.fc1" in name:
                comp_key += "mlp_fc1"
            elif "mlp.fc2" in name:
                comp_key += "mlp_fc2"
            elif "norm1" in name:
                comp_key += "norm1"
            elif "norm2" in name:
                comp_key += "norm2"
            else:
                continue
            
            comp = comp_key
        else:
            continue
        
        if comp not in components:
            components[comp] = []
        components[comp].append((name, p))
    
    return components

components = group_by_component(model)

print(f"\nComponent structure:")
print(f"{'Component':<30} {'Parameters':>12} {'Percentage':>12}")
print("-" * 56)
total_params = sum(sum(p.numel() for _, p in params) for params in components.values())
for comp_name in sorted(components.keys()):
    params = components[comp_name]
    param_count = sum(p.numel() for _, p in params)
    pct = 100 * param_count / total_params if total_params > 0 else 0
    print(f"{comp_name:<30} {param_count:>12,} {pct:>11.2f}%")

# ========================================
# Component Fault Injector
# ========================================
class ComponentFaultInjector:
    def __init__(self, component_params, test_loader, device, baseline_acc):
        self.component_params = component_params
        self.test_loader = test_loader
        self.device = device
        self.baseline_acc = baseline_acc
        self.injected_faults = set()
        self.N = sum(p.numel() for _, p in component_params)
    
    def generate_fault_id(self):
        param_idx = random.randint(0, len(self.component_params) - 1)
        name, p = self.component_params[param_idx]
        elem_idx = random.randint(0, p.numel() - 1)
        bit_idx = random.randint(0, 31)
        fault_id = f"{param_idx}_{elem_idx}_{bit_idx}"
        return fault_id, param_idx, elem_idx, bit_idx
    
    def inject_fault(self, param_idx, elem_idx, bit_idx):
        name, p = self.component_params[param_idx]
        flat = p.view(-1)
        x = float(flat[elem_idx].item())
        packed = struct.pack("!f", np.float32(x))
        i = struct.unpack("!I", packed)[0]
        bit = 1 << bit_idx
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
                fault_id, param_idx, elem_idx, bit_idx = self.generate_fault_id()
                if fault_id not in self.injected_faults:
                    break
                attempts += 1
            if attempts >= 100:
                continue
            
            self.injected_faults.add(fault_id)
            model.load_state_dict(clean_state, strict=True)
            model.to(self.device)
            model.eval()
            self.inject_fault(param_idx, elem_idx, bit_idx)
            
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
                    "bit_idx": bit_idx,
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
                    "bit_idx": bit_idx,
                    "accuracy": 0.0,
                    "has_nan": True,
                    "has_inf": False,
                    "failure_type": "error",
                })
        
        return results

# ========================================
# Run Component-Wise Campaigns
# ========================================
print("\n" + "="*70)
print("RUNNING COMPONENT-WISE CAMPAIGNS")
print("="*70)

campaign_start = datetime.now()
all_component_results = {}
component_summary = []

for comp_name in sorted(components.keys()):
    comp_params = components[comp_name]
    
    print(f"\n{'='*70}")
    print(f"COMPONENT: {comp_name}")
    print(f"{'='*70}")
    
    injector = ComponentFaultInjector(comp_params, test_loader, device, baseline_acc)
    
    n_required = calculate_sample_size(
        N=injector.N,
        e=MARGIN_OF_ERROR,
        confidence_level=CONFIDENCE_LEVEL,
        p=FAILURE_PROB
    )
    
    print(f"Population (N): {injector.N:,}")
    print(f"Required sample size: {n_required:,}")
    print(f"Estimated time: {n_required * 7 / 60:.1f} minutes")
    
    comp_start = datetime.now()
    results = injector.run_campaign(model, clean_state, n_required)
    comp_duration = (datetime.now() - comp_start).total_seconds()
    
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
    
    print(f"\nResults for {comp_name}:")
    print(f"  Tests: {len(results)}")
    print(f"  Failures: {critical_failures}")
    print(f"  Failure rate: {p_hat*100:.2f}% ± {e_hat*100:.2f}%")
    print(f"  CI: [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]")
    print(f"  Duration: {comp_duration:.1f} sec ({comp_duration/60:.1f} min)")
    
    all_component_results[comp_name] = results
    component_summary.append({
        "component_name": comp_name,
        "population_size": injector.N,
        "sample_size": len(results),
        "failures": critical_failures,
        "failure_rate": p_hat,
        "margin_error": e_hat,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "duration_seconds": comp_duration,
        "failure_counts": failure_counts
    })

campaign_duration = (datetime.now() - campaign_start).total_seconds()

# ========================================
# Generate Component Ranking
# ========================================
print("\n" + "="*70)
print("COMPONENT VULNERABILITY RANKING")
print("="*70)

ranked_components = sorted(component_summary, key=lambda x: x["failure_rate"], reverse=True)

print(f"\n{'Rank':<6} {'Component':<30} {'Rate':>15} {'95% CI':>25} {'Params':>12}")
print("-" * 92)
for rank, comp in enumerate(ranked_components, 1):
    print(f"{rank:<6} {comp['component_name']:<30} "
          f"{comp['failure_rate']*100:>6.2f}% ± {comp['margin_error']*100:>4.2f}% "
          f"[{comp['ci_lower']*100:>5.2f}%, {comp['ci_upper']*100:>5.2f}%] "
          f"{comp['population_size']:>12,}")

# ========================================
# Save Results
# ========================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

summary_file = OUTPUT_DIR / f"componentwise_summary_{timestamp}.json"
with open(summary_file, "w") as f:
    json.dump({
        "experiment": "componentwise_analysis",
        "timestamp": timestamp,
        "configuration": {
            "margin_of_error": MARGIN_OF_ERROR,
            "confidence_level": CONFIDENCE_LEVEL,
            "failure_probability": FAILURE_PROB,
            "baseline_accuracy": baseline_acc,
            "focus_layers": FOCUS_LAYERS,
        },
        "total_duration_seconds": campaign_duration,
        "component_summary": component_summary,
        "ranked_components": [
            {"rank": i+1, "component": c["component_name"], "failure_rate": c["failure_rate"]}
            for i, c in enumerate(ranked_components)
        ]
    }, f, indent=2)
print(f"\nSummary saved: {summary_file}")

for comp_name, results in all_component_results.items():
    if results:
        results_file = OUTPUT_DIR / f"componentwise_{comp_name}_{timestamp}.csv"
        with open(results_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

print("\n" + "="*70)
print("COMPONENT-WISE ANALYSIS COMPLETE")
print("="*70)
print(f"\nTotal duration: {campaign_duration:.1f} sec ({campaign_duration/3600:.2f} hours)")
print(f"\nTop 3 Most Vulnerable Components:")
for i, comp in enumerate(ranked_components[:3], 1):
    print(f"  {i}. {comp['component_name']}: {comp['failure_rate']*100:.2f}% ± {comp['margin_error']*100:.2f}%")
