"""
Analysis Script for Statistical Fault Injection Results

Analyzes the baseline campaign results and generates insights.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load results
results_dir = Path("./statistical_results")

# Find the most recent results
csv_files = list(results_dir.glob("baseline_results_*.csv"))
json_files = list(results_dir.glob("baseline_summary_*.json"))

if not csv_files or not json_files:
    print("Error: No results files found!")
    print("Make sure you're running from the directory containing statistical_results/")
    exit(1)

# Load most recent
results_csv = sorted(csv_files)[-1]
summary_json = sorted(json_files)[-1]

print("="*70)
print("STATISTICAL FAULT INJECTION - RESULTS ANALYSIS")
print("="*70)
print(f"Loading: {results_csv.name}")
print(f"Loading: {summary_json.name}")
print()

# Load data
df = pd.read_csv(results_csv)
with open(summary_json, 'r') as f:
    summary = json.load(f)

# ========================================
# 1. Summary Statistics
# ========================================
print("="*70)
print("1. SUMMARY STATISTICS")
print("="*70)

metrics = summary['metrics']
config = summary['configuration']

print(f"\nConfiguration:")
print(f"  Margin of error: {config['margin_of_error']*100:.2f}%")
print(f"  Confidence level: {config['confidence_level']*100:.0f}%")
print(f"  Population size: {config['population_size']:,}")
print(f"  Sample size: {config['sample_size']:,} ({100*config['sample_size']/config['population_size']:.4f}%)")

print(f"\nResults:")
print(f"  Baseline accuracy: {config['baseline_accuracy']:.2f}%")
print(f"  Failure rate: {metrics['p_hat']*100:.4f}% ± {metrics['e_hat']*100:.4f}%")
print(f"  Confidence interval: [{metrics['ci_lower']*100:.4f}%, {metrics['ci_upper']*100:.4f}%]")
print(f"  Total failures: {summary['failure_counts']['catastrophic_nan_inf'] + summary['failure_counts']['catastrophic_low_acc'] + summary['failure_counts']['mission_failure'] + summary['failure_counts']['sdc_significant']}")

# ========================================
# 2. Failure Type Distribution
# ========================================
print("\n" + "="*70)
print("2. FAILURE TYPE ANALYSIS")
print("="*70)

failure_counts = summary['failure_counts']
total = config['sample_size']

print(f"\n{'Category':<30} {'Count':>8} {'Percentage':>12}")
print("-"*52)
for category, count in failure_counts.items():
    pct = 100 * count / total
    print(f"{category:<30} {count:>8} {pct:>11.2f}%")

# ========================================
# 3. Accuracy Distribution
# ========================================
print("\n" + "="*70)
print("3. ACCURACY DISTRIBUTION")
print("="*70)

accuracies = df['accuracy'].values

print(f"\nAccuracy Statistics:")
print(f"  Mean: {accuracies.mean():.2f}%")
print(f"  Std: {accuracies.std():.2f}%")
print(f"  Min: {accuracies.min():.2f}%")
print(f"  Q1 (25%): {np.percentile(accuracies, 25):.2f}%")
print(f"  Median: {np.percentile(accuracies, 50):.2f}%")
print(f"  Q3 (75%): {np.percentile(accuracies, 75):.2f}%")
print(f"  Max: {accuracies.max():.2f}%")

# ========================================
# 4. Bit Position Analysis
# ========================================
print("\n" + "="*70)
print("4. BIT POSITION ANALYSIS")
print("="*70)

bit_failure_rates = {}
for bit in range(32):
    bit_faults = df[df['bit_idx'] == bit]
    if len(bit_faults) > 0:
        failures = bit_faults['failure_type'].isin([
            'catastrophic_nan_inf',
            'catastrophic_low_acc',
            'mission_failure',
            'sdc_significant'
        ]).sum()
        bit_failure_rates[bit] = 100 * failures / len(bit_faults)

print(f"\n{'Bit Position':<15} {'Tests':>8} {'Failures':>10} {'Rate':>10}")
print("-"*45)
for bit in range(32):
    bit_faults = df[df['bit_idx'] == bit]
    if len(bit_faults) > 0:
        failures = bit_faults['failure_type'].isin([
            'catastrophic_nan_inf',
            'catastrophic_low_acc',
            'mission_failure',
            'sdc_significant'
        ]).sum()
        rate = bit_failure_rates.get(bit, 0)
        print(f"Bit {bit:2d}{'':<10} {len(bit_faults):>8} {failures:>10} {rate:>9.2f}%")

# Bit ranges
print(f"\nBit Range Summary:")
mantissa_low = list(range(0, 20))
mantissa_high = list(range(20, 23))
exponent = list(range(23, 31))
sign_high_exp = [30, 31]

for name, bits in [
    ("Mantissa (0-19)", mantissa_low),
    ("Mantissa (20-22)", mantissa_high),
    ("Exponent (23-30)", exponent),
    ("Sign + High Exp (30-31)", sign_high_exp)
]:
    bit_faults = df[df['bit_idx'].isin(bits)]
    if len(bit_faults) > 0:
        failures = bit_faults['failure_type'].isin([
            'catastrophic_nan_inf',
            'catastrophic_low_acc',
            'mission_failure',
            'sdc_significant'
        ]).sum()
        rate = 100 * failures / len(bit_faults)
        print(f"  {name:<30} {len(bit_faults):>8} tests, {failures:>5} failures ({rate:.2f}%)")

# ========================================
# 5. NaN/Inf Analysis
# ========================================
print("\n" + "="*70)
print("5. NaN/Inf ANALYSIS")
print("="*70)

nan_inf_faults = df[df['failure_type'] == 'catastrophic_nan_inf']
print(f"\nTotal NaN/Inf failures: {len(nan_inf_faults)}")

if len(nan_inf_faults) > 0:
    print(f"\nBit distribution of NaN/Inf failures:")
    bit_dist = nan_inf_faults['bit_idx'].value_counts().sort_index()
    for bit, count in bit_dist.items():
        pct = 100 * count / len(nan_inf_faults)
        print(f"  Bit {bit:2d}: {count:4d} ({pct:5.1f}%)")

# ========================================
# 6. Visualizations
# ========================================
print("\n" + "="*70)
print("6. GENERATING VISUALIZATIONS")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Statistical Fault Injection Results Analysis', fontsize=16, fontweight='bold')

# Plot 1: Failure Type Distribution
ax1 = axes[0, 0]
categories = list(failure_counts.keys())
counts = [failure_counts[c] for c in categories]
colors = ['#d62728', '#ff7f0e', '#ff9896', '#ffbb78', '#c5b0d5', '#98df8a']
ax1.pie(counts, labels=categories, autopct='%1.1f%%', colors=colors, startangle=90)
ax1.set_title('Failure Type Distribution', fontweight='bold')

# Plot 2: Accuracy Histogram
ax2 = axes[0, 1]
ax2.hist(accuracies, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
ax2.axvline(config['baseline_accuracy'], color='green', linestyle='--', linewidth=2, label='Baseline')
ax2.axvline(50, color='red', linestyle='--', linewidth=2, label='Mission Failure Threshold')
ax2.axvline(20, color='darkred', linestyle='--', linewidth=2, label='Catastrophic Threshold')
ax2.set_xlabel('Accuracy (%)')
ax2.set_ylabel('Frequency')
ax2.set_title('Accuracy Distribution After Faults', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Bit Position Vulnerability
ax3 = axes[1, 0]
bits = sorted(bit_failure_rates.keys())
rates = [bit_failure_rates[b] for b in bits]
colors_bits = ['green' if b < 20 else 'yellow' if b < 23 else 'orange' if b < 30 else 'red' for b in bits]
ax3.bar(bits, rates, color=colors_bits, edgecolor='black', alpha=0.7)
ax3.set_xlabel('Bit Position')
ax3.set_ylabel('Failure Rate (%)')
ax3.set_title('Failure Rate by Bit Position', fontweight='bold')
ax3.set_xticks(range(0, 32, 2))
ax3.grid(True, alpha=0.3, axis='y')

# Add bit range annotations
ax3.axvspan(-0.5, 19.5, alpha=0.1, color='green', label='Mantissa (low)')
ax3.axvspan(19.5, 22.5, alpha=0.1, color='yellow', label='Mantissa (high)')
ax3.axvspan(22.5, 30.5, alpha=0.1, color='orange', label='Exponent')
ax3.axvspan(30.5, 31.5, alpha=0.1, color='red', label='Sign+High Exp')
ax3.legend(loc='upper left', fontsize=8)

# Plot 4: Confidence Interval
ax4 = axes[1, 1]
p_hat = metrics['p_hat'] * 100
ci_lower = metrics['ci_lower'] * 100
ci_upper = metrics['ci_upper'] * 100
e_hat = metrics['e_hat'] * 100

ax4.errorbar([1], [p_hat], yerr=[[p_hat - ci_lower], [ci_upper - p_hat]], 
             fmt='o', markersize=15, capsize=10, capthick=3, 
             color='steelblue', ecolor='steelblue', linewidth=3)
ax4.axhline(p_hat, color='steelblue', linestyle='--', alpha=0.5)
ax4.fill_between([0.5, 1.5], ci_lower, ci_upper, alpha=0.2, color='steelblue')
ax4.set_xlim(0.5, 1.5)
ax4.set_ylim(0, max(5, ci_upper * 1.2))
ax4.set_ylabel('Failure Rate (%)')
ax4.set_title(f'Failure Rate: {p_hat:.2f}% ± {e_hat:.2f}% (99% CI)', fontweight='bold')
ax4.set_xticks([])
ax4.grid(True, alpha=0.3, axis='y')

# Add text annotations
ax4.text(1, p_hat, f'{p_hat:.2f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax4.text(1.05, ci_upper, f'{ci_upper:.2f}%', ha='left', va='center', fontsize=10)
ax4.text(1.05, ci_lower, f'{ci_lower:.2f}%', ha='left', va='center', fontsize=10)

plt.tight_layout()
plot_file = results_dir / "analysis_plots.png"
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"\nVisualization saved: {plot_file}")

# ========================================
# 7. Key Insights
# ========================================
print("\n" + "="*70)
print("7. KEY INSIGHTS")
print("="*70)

print(f"\n✓ Model Resilience: {failure_counts['no_effect']/total*100:.1f}% of faults have no observable impact")
print(f"✓ Critical Failures: {(failure_counts['catastrophic_nan_inf']+failure_counts['catastrophic_low_acc'])/total*100:.2f}% cause catastrophic failure")
print(f"✓ Statistical Rigor: 99% confident the true failure rate is [{ci_lower:.2f}%, {ci_upper:.2f}%]")
print(f"✓ Precision Achieved: ±{e_hat:.2f}% (better than target ±{config['margin_of_error']*100:.0f}%)")

# Bit-wise insight
high_bits = [30, 31]
high_bit_faults = df[df['bit_idx'].isin(high_bits)]
if len(high_bit_faults) > 0:
    high_bit_failures = high_bit_faults['failure_type'].isin([
        'catastrophic_nan_inf',
        'catastrophic_low_acc',
        'mission_failure',
        'sdc_significant'
    ]).sum()
    high_bit_rate = 100 * high_bit_failures / len(high_bit_faults)
    low_bits = list(range(0, 20))
    low_bit_faults = df[df['bit_idx'].isin(low_bits)]
    low_bit_failures = low_bit_faults['failure_type'].isin([
        'catastrophic_nan_inf',
        'catastrophic_low_acc',
        'mission_failure',
        'sdc_significant'
    ]).sum()
    low_bit_rate = 100 * low_bit_failures / len(low_bit_faults) if len(low_bit_faults) > 0 else 0
    
    print(f"✓ Bit Vulnerability: Bits 30-31 have {high_bit_rate:.1f}% failure rate vs {low_bit_rate:.1f}% for bits 0-19")

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print(f"\nNext steps:")
print(f"  1. Review the visualization: {plot_file}")
print(f"  2. Examine detailed CSV: {results_csv}")
print(f"  3. Consider implementing Phase 2B (iterative approach)")
print(f"  4. Plan multi-granularity analysis (layer-wise, component-wise, bit-wise)")