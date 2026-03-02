"""
Phase 2C: Layer-Wise Statistical Fault Injection

Analyzes vulnerability at layer granularity to identify which ViT layers are most critical.
Uses optimized parameters for speed while maintaining statistical rigor.

Based on baseline results: p̂=0.0293, so we use p=0.03 instead of conservative p=0.5
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

# Statistical Parameters - OPTIMIZED FOR SPEED
MARGIN_OF_ERROR = 0.05  # 5% margin (vs 1% in baseline) - 25x faster
CONFIDENCE_LEVEL = 0.95  # 95% confidence (vs 99%) - slightly faster
FAILURE_PROB = 0.03  # Use measured p from baseline (vs 0.5 conservative) - HUGE speedup!

# Failure Thresholds
FAIL_THRESH = 50.0
CATA_THRESH = 20.0

# Test Configuration
SUBSET_N = 1000

print("="*70)
print("PHASE 2C: LAYER-WISE STATISTICAL FAULT INJECTION")
print("="*70)
print(f"Margin of Error: {MARGIN_OF_ERROR*100}% (optimized for speed)")
print(f"Confidence Level: {CONFIDENCE_LEVEL*100}%")
print(f"Failure Probability (p): {FAILURE_PROB} (from baseline measurement)")
print("="*70)
print("\nSpeed optimizations:")
print("  ✓ Using measured p̂=0.0293 (not conservative 0.5)")
print("  ✓ Looser margin: 5% (acceptable for layer ranking)")
print("  ✓ 95% confidence (still rigorous)")
print("  → Expected: ~50 faults/layer vs ~16,000 for baseline")
print("  → Estimated time: ~1.5-2 hours for all 12 layers")
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

# ========================================
# 3) Layer Grouping
# ========================================
print("\n[3] Organizing parameters by layer...")

def group_by_layer(model):
    """
    Group ViT parameters by layer.
    
    ViT-tiny structure:
    - patch_embed: Initial patch embedding
    - blocks.0 through blocks.11: 12 transformer layers
    - norm: Final layer norm
    - head: Classification head
    """
    layers = {
        "patch_embed": [],
        "norm": [],
        "head": [],
    }
    
    # Add transformer blocks
    for i in range(12):
        layers[f"block_{i}"] = []
    
    # Categorize parameters
    for name, p in model.named_parameters():
        if p is None or not p.requires_grad or p.numel() == 0:
            continue
            
        if name.startswith("patch_embed"):
            layers["patch_embed"].append((name, p))
        elif name.startswith("blocks."):
            # Extract block number
            block_num = int(name.split(".")[1])
            layers[f"block_{block_num}"].append((name, p))
        elif name.startswith("norm"):
            layers["norm"].append((name, p))
        elif name.startswith("head"):
            layers["head"].append((name, p))
        else:
            print(f"Warning: Unclassified parameter: {name}")
    
    return layers

layers = group_by_layer(model)

# Print layer summary
print("\nLayer structure:")
print(f"{'Layer Name':<20} {'Parameters':>12} {'Percentage':>12}")
print("-" * 46)
total_params = sum(sum(p.numel() for _, p in params) for params in layers.values())
for layer_name, params in sorted(layers.items()):
    param_count = sum(p.numel() for _, p in params)
    pct = 100 * param_count / total_params
    print(f"{layer_name:<20} {param_count:>12,} {pct:>11.2f}%")
print(f"{'TOTAL':<20} {total_params:>12,} {100.0:>11.2f}%")

# ========================================
# 4) Baseline Evaluation
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

baseline_acc = evaluate_accuracy(model, test_loader)
print(f"\nBaseline accuracy: {baseline_acc:.2f}%")

# ========================================
# 5) Fault Injection Class
# ========================================
class LayerFaultInjector:
    """Fault injector for specific layer."""
    
    def __init__(self, layer_params, test_loader, device, baseline_acc):
        self.layer_params = layer_params
        self.test_loader = test_loader
        self.device = device
        self.baseline_acc = baseline_acc
        self.injected_faults = set()
        
        # Calculate population size for this layer
        self.N = sum(p.numel() for _, p in layer_params)
        
    def generate_fault_id(self):
        """Generate unique fault identifier."""
        # Choose random parameter from this layer
        param_idx = random.randint(0, len(self.layer_params) - 1)
        name, p = self.layer_params[param_idx]
        
        # Choose random element
        elem_idx = random.randint(0, p.numel() - 1)
        
        # Choose random bit
        bit_idx = random.randint(0, 31)
        
        fault_id = f"{param_idx}_{elem_idx}_{bit_idx}"
        return fault_id, param_idx, elem_idx, bit_idx
    
    def inject_fault(self, param_idx, elem_idx, bit_idx):
        """Inject fault into model."""
        name, p = self.layer_params[param_idx]
        flat = p.view(-1)
        
        # Get value
        x = float(flat[elem_idx].item())
        
        # Flip bit
        packed = struct.pack("!f", np.float32(x))
        i = struct.unpack("!I", packed)[0]
        bit = 1 << bit_idx
        i_corrupt = i ^ bit
        packed_corrupt = struct.pack("!I", i_corrupt)
        x_corrupt = struct.unpack("!f", packed_corrupt)[0]
        
        # Update
        with torch.no_grad():
            flat.data[elem_idx] = x_corrupt
    
    @torch.no_grad()
    def run_campaign(self, model, clean_state, n_faults):
        """Run fault injection campaign on this layer."""
        results = []
        
        for i in range(n_faults):
            # Generate unique fault
            attempts = 0
            while attempts < 100:
                fault_id, param_idx, elem_idx, bit_idx = self.generate_fault_id()
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
            self.inject_fault(param_idx, elem_idx, bit_idx)
            
            # Evaluate
            try:
                acc = evaluate_accuracy(model, self.test_loader)
                
                # Check NaN/Inf
                has_nan = any(torch.isnan(p).any() for _, p in model.named_parameters())
                has_inf = any(torch.isinf(p).any() for _, p in model.named_parameters())
                
                # Classify
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
                
            except Exception as e:
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
# 6) Run Layer-Wise Campaigns
# ========================================
print("\n" + "="*70)
print("RUNNING LAYER-WISE STATISTICAL CAMPAIGNS")
print("="*70)

campaign_start = datetime.now()
all_layer_results = {}
layer_summary = []

# Process each layer
for layer_name in sorted(layers.keys()):
    layer_params = layers[layer_name]
    
    if not layer_params:
        print(f"\n[SKIP] {layer_name}: No parameters")
        continue
    
    print(f"\n{'='*70}")
    print(f"LAYER: {layer_name}")
    print(f"{'='*70}")
    
    # Create injector for this layer
    injector = LayerFaultInjector(layer_params, test_loader, device, baseline_acc)
    
    # Calculate sample size
    n_required = calculate_sample_size(
        N=injector.N,
        e=MARGIN_OF_ERROR,
        confidence_level=CONFIDENCE_LEVEL,
        p=FAILURE_PROB
    )
    
    print(f"Population (N): {injector.N:,}")
    print(f"Required sample size: {n_required:,}")
    print(f"Estimated time: {n_required * 10 / 60:.1f} minutes")
    
    # Run campaign
    layer_start = datetime.now()
    results = injector.run_campaign(model, clean_state, n_required)
    layer_duration = (datetime.now() - layer_start).total_seconds()
    
    # Analyze results
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
    
    # Calculate statistics
    p_hat = critical_failures / len(results) if results else 0
    e_hat = calculate_error_margin(len(results), injector.N, p_hat, CONFIDENCE_LEVEL)
    ci_lower = max(0, p_hat - e_hat)
    ci_upper = min(1, p_hat + e_hat)
    
    print(f"\nResults for {layer_name}:")
    print(f"  Tests: {len(results)}")
    print(f"  Failures: {critical_failures}")
    print(f"  Failure rate: {p_hat*100:.2f}% ± {e_hat*100:.2f}%")
    print(f"  Confidence interval: [{ci_lower*100:.2f}%, {ci_upper*100:.2f}%]")
    print(f"  Duration: {layer_duration:.1f} seconds ({layer_duration/60:.1f} minutes)")
    
    # Store results
    all_layer_results[layer_name] = results
    layer_summary.append({
        "layer_name": layer_name,
        "population_size": injector.N,
        "sample_size": len(results),
        "failures": critical_failures,
        "failure_rate": p_hat,
        "margin_error": e_hat,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "duration_seconds": layer_duration,
        "failure_counts": failure_counts
    })

campaign_duration = (datetime.now() - campaign_start).total_seconds()

# ========================================
# 7) Generate Layer Ranking
# ========================================
print("\n" + "="*70)
print("LAYER VULNERABILITY RANKING")
print("="*70)

# Sort by failure rate
ranked_layers = sorted(layer_summary, key=lambda x: x["failure_rate"], reverse=True)

print(f"\n{'Rank':<6} {'Layer':<20} {'Failure Rate':>15} {'95% CI':>25} {'Parameters':>12}")
print("-" * 82)
for rank, layer in enumerate(ranked_layers, 1):
    print(f"{rank:<6} {layer['layer_name']:<20} "
          f"{layer['failure_rate']*100:>6.2f}% ± {layer['margin_error']*100:>4.2f}% "
          f"[{layer['ci_lower']*100:>5.2f}%, {layer['ci_upper']*100:>5.2f}%] "
          f"{layer['population_size']:>12,}")

# ========================================
# 8) Save Results
# ========================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save summary
summary_file = OUTPUT_DIR / f"layerwise_summary_{timestamp}.json"
with open(summary_file, "w") as f:
    json.dump({
        "experiment": "layerwise_analysis",
        "timestamp": timestamp,
        "configuration": {
            "margin_of_error": MARGIN_OF_ERROR,
            "confidence_level": CONFIDENCE_LEVEL,
            "failure_probability": FAILURE_PROB,
            "baseline_accuracy": baseline_acc,
        },
        "total_duration_seconds": campaign_duration,
        "layer_summary": layer_summary,
        "ranked_layers": [
            {"rank": i+1, "layer": l["layer_name"], "failure_rate": l["failure_rate"]}
            for i, l in enumerate(ranked_layers)
        ]
    }, f, indent=2)
print(f"\nSummary saved: {summary_file}")

# Save detailed results for each layer
for layer_name, results in all_layer_results.items():
    if results:
        results_file = OUTPUT_DIR / f"layerwise_{layer_name}_{timestamp}.csv"
        with open(results_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

print("\n" + "="*70)
print("LAYER-WISE ANALYSIS COMPLETE")
print("="*70)
print(f"\nTotal duration: {campaign_duration:.1f} seconds ({campaign_duration/3600:.2f} hours)")
print(f"\nTop 3 Most Vulnerable Layers:")
for i, layer in enumerate(ranked_layers[:3], 1):
    print(f"  {i}. {layer['layer_name']}: {layer['failure_rate']*100:.2f}% ± {layer['margin_error']*100:.2f}%")

print(f"\nNext steps:")
print(f"  1. Review layer ranking in {summary_file}")
print(f"  2. Consider tighter analysis on most vulnerable layers")
print(f"  3. Proceed to component-wise or bit-wise analysis")
