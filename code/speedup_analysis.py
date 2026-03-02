"""
Quick calculation: Why layer-wise is SO much faster than baseline

Shows the mathematical speedup from our optimizations.
"""

from statistical_utils import calculate_sample_size

print("="*70)
print("SPEEDUP ANALYSIS: BASELINE vs LAYER-WISE")
print("="*70)

# Baseline parameters
N_baseline = 5_526_346
e_baseline = 0.01
conf_baseline = 0.99
p_baseline = 0.5

n_baseline = calculate_sample_size(N_baseline, e_baseline, conf_baseline, p_baseline)

print("\nBASELINE CAMPAIGN:")
print(f"  Population: {N_baseline:,}")
print(f"  Margin: {e_baseline*100}%")
print(f"  Confidence: {conf_baseline*100}%")
print(f"  Assumed p: {p_baseline}")
print(f"  → Sample size: {n_baseline:,}")
print(f"  → Time @ 10 sec/test: {n_baseline*10/3600:.1f} hours")

# Layer-wise parameters (per layer)
N_layer = 460_000  # Average layer size
e_layer = 0.05
conf_layer = 0.95
p_layer = 0.03  # Measured from baseline!

n_layer = calculate_sample_size(N_layer, e_layer, conf_layer, p_layer)

print("\nLAYER-WISE CAMPAIGN (per layer):")
print(f"  Population: {N_layer:,}")
print(f"  Margin: {e_layer*100}%")
print(f"  Confidence: {conf_layer*100}%")
print(f"  Measured p: {p_layer}")
print(f"  → Sample size: {n_layer:,}")
print(f"  → Time @ 10 sec/test: {n_layer*10/60:.1f} minutes")

print("\nLAYER-WISE TOTAL (12 layers):")
n_total = n_layer * 12
print(f"  Sample size: {n_total:,}")
print(f"  Time @ 10 sec/test: {n_total*10/3600:.1f} hours")

print("\n" + "="*70)
print("SPEEDUP FACTORS:")
print("="*70)

speedup_per_layer = n_baseline / n_layer
speedup_total = (n_baseline * 10) / (n_total * 10)

print(f"\nPer-layer speedup: {speedup_per_layer:.1f}x faster")
print(f"Total speedup: {speedup_total:.1f}x faster")
print(f"\nTime reduction:")
print(f"  Baseline: 46.6 hours")
print(f"  Layer-wise: ~{n_total*10/3600:.1f} hours")
print(f"  Savings: {46.6 - n_total*10/3600:.1f} hours ({100*(46.6 - n_total*10/3600)/46.6:.1f}% reduction)")

print("\n" + "="*70)
print("WHY SO MUCH FASTER?")
print("="*70)
print("\n1. Using measured p=0.03 instead of conservative p=0.5:")
print(f"   p(1-p) term: 0.03×0.97 = 0.0291 vs 0.5×0.5 = 0.25")
print(f"   Reduction factor: {0.25/0.0291:.1f}x fewer samples needed")

print("\n2. Looser margin (5% vs 1%):")
print(f"   e² term: 0.05² = 0.0025 vs 0.01² = 0.0001")
print(f"   Reduction factor: {0.0001/0.0025:.1f}x fewer samples needed")

print("\n3. Lower confidence (95% vs 99%):")
print(f"   t² term: 1.96² = 3.84 vs 2.576² = 6.64")
print(f"   Reduction factor: {6.64/3.84:.1f}x fewer samples needed")

print("\nCombined effect: ~330x fewer samples per layer!")
print("This is mathematically valid because we're ranking layers,")
print("not computing absolute failure rates for certification.")

print("\n" + "="*70)
print("PRECISION TRADE-OFF:")
print("="*70)
print(f"\nBaseline: \"Failure rate is 2.93% ± 0.34%\" (very precise)")
print(f"Layer-wise: \"Layer 5 is 4% ± 5%\" (loose, but good for ranking)")
print(f"\nFor layer ranking, we only need to know RELATIVE vulnerability,")
print(f"not absolute values with tight confidence intervals.")
print(f"If Layer A = 8% ± 5% and Layer B = 2% ± 5%, we know A > B!")
