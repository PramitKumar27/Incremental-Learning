"""
Statistical Fault Injection Utilities

Implements core formulas from:
- Leveugle et al. (DATE 2009): Statistical Fault Injection - Quantified Error and Confidence
- Ruospo et al. (IEEE TC 2025): Iterative Statistical Fault Injection for Deep Neural Networks
"""

import numpy as np
from scipy import stats
from typing import Dict, Tuple


def get_confidence_constant(confidence_level: float) -> float:
    """
    Get the t-value (z-score) for a given confidence level.
    
    Args:
        confidence_level: Confidence level (e.g., 0.95, 0.99, 0.999)
    
    Returns:
        t-value from standard normal distribution
        
    Common values:
        0.90 → 1.645
        0.95 → 1.960
        0.99 → 2.576
        0.999 → 3.291
    """
    # For two-tailed test, use (1 + confidence_level) / 2
    alpha = 1 - confidence_level
    t = stats.norm.ppf(1 - alpha / 2)
    return t


def calculate_sample_size(
    N: int,
    e: float,
    confidence_level: float,
    p: float = 0.5
) -> int:
    """
    Calculate required sample size for statistical fault injection.
    
    Based on Leveugle et al. (2009), Equation 4:
    n = N / (1 + e²(N-1)/(t²p(1-p)))
    
    Args:
        N: Population size (total possible faults)
        e: Margin of error (decimal, e.g., 0.01 for 1%)
        confidence_level: Confidence level (e.g., 0.99 for 99%)
        p: Estimated failure probability (default 0.5 for conservative)
        
    Returns:
        Required sample size n
        
    Example:
        >>> calculate_sample_size(N=10_000_000, e=0.001, confidence_level=0.99, p=0.5)
        1658772
    """
    if N <= 0:
        raise ValueError("Population size N must be positive")
    if not 0 < e < 1:
        raise ValueError("Margin of error e must be between 0 and 1")
    if not 0 < confidence_level < 1:
        raise ValueError("Confidence level must be between 0 and 1")
    if not 0 < p < 1:
        raise ValueError("Failure probability p must be between 0 and 1")
    
    t = get_confidence_constant(confidence_level)
    
    # Calculate sample size using Leveugle formula
    numerator = t**2 * p * (1 - p) * N
    denominator = e**2 * (N - 1) + t**2 * p * (1 - p)
    
    n = numerator / denominator
    
    # Round up to ensure we meet the target precision
    n_rounded = int(np.ceil(n))
    
    # Sanity check: n cannot exceed N
    n_final = min(n_rounded, N)
    
    return n_final


def calculate_error_margin(
    n: int,
    N: int,
    p_hat: float,
    confidence_level: float
) -> float:
    """
    Calculate achieved margin of error for a given sample.
    
    Based on Leveugle et al. (2009), error margin formula:
    e = t * sqrt(p(1-p)/n) * sqrt((N-n)/(N-1))
    
    Args:
        n: Sample size (faults injected)
        N: Population size
        p_hat: Observed failure rate
        confidence_level: Confidence level
        
    Returns:
        Achieved margin of error e
        
    Example:
        >>> calculate_error_margin(n=1000, N=10_000_000, p_hat=0.01, confidence_level=0.99)
        0.00806
    """
    if n <= 0 or n > N:
        raise ValueError(f"Sample size n={n} must be 0 < n <= N={N}")
    if not 0 <= p_hat <= 1:
        raise ValueError("Observed failure rate p_hat must be between 0 and 1")
    
    t = get_confidence_constant(confidence_level)
    
    # Handle edge cases for p_hat
    if p_hat == 0 or p_hat == 1:
        # When all successes or all failures, use continuity correction
        # Add 0.5/n to avoid division issues
        p_hat_adjusted = max(0.5/n, min(p_hat, 1 - 0.5/n))
    else:
        p_hat_adjusted = p_hat
    
    # Standard error
    se = np.sqrt(p_hat_adjusted * (1 - p_hat_adjusted) / n)
    
    # Finite population correction factor
    if N > n:
        fpc = np.sqrt((N - n) / (N - 1))
    else:
        fpc = 0  # If n == N, we've tested everything
    
    # Margin of error
    e = t * se * fpc
    
    return e


def calculate_ep_curve_next_error(
    p_hat: float,
    e_hat_current: float,
    e_goal: float
) -> float:
    """
    Calculate next error margin using E-P curve from Ruospo et al. (2025).
    
    Based on Equation 5 from the paper:
    e_(i+1) = -k*P_hat_i^2 + k*P_hat_i + e_goal
    where k = 4*(E_hat_i/3 - e_goal)
    
    The curve is a parabola passing through:
    - (0, e_goal): When failure rate is 0%, jump to target
    - (1, e_goal): When failure rate is 100%, jump to target
    - (0.5, E_hat/3): When failure rate is 50%, reduce gradually
    
    Args:
        p_hat: Current failure rate estimate
        e_hat_current: Current achieved error margin
        e_goal: Target error margin
        
    Returns:
        Next error margin to use in next iteration
    """
    # Check if we should use E-P curve or just go to goal
    if e_hat_current / 3 <= e_goal:
        # Already close enough, just use goal
        return e_goal
    
    # Calculate k parameter
    k = 4 * (e_hat_current / 3 - e_goal)
    
    # Calculate next error using parabola formula
    e_next = -k * (p_hat ** 2) + k * p_hat + e_goal
    
    # Ensure e_next is within reasonable bounds
    e_next = max(e_goal, min(e_next, e_hat_current))
    
    return e_next


def print_statistical_report(
    n: int,
    N: int,
    failures: int,
    confidence_level: float,
    description: str = ""
) -> Dict[str, float]:
    """
    Print a formatted statistical report and return metrics.
    
    Args:
        n: Sample size
        N: Population size
        failures: Number of failures observed
        confidence_level: Confidence level used
        description: Optional description for the report
        
    Returns:
        Dictionary with statistical metrics
    """
    p_hat = failures / n if n > 0 else 0
    e_hat = calculate_error_margin(n, N, p_hat, confidence_level)
    
    t = get_confidence_constant(confidence_level)
    
    ci_lower = max(0, p_hat - e_hat)
    ci_upper = min(1, p_hat + e_hat)
    
    print("\n" + "="*70)
    if description:
        print(f"Statistical Report: {description}")
    else:
        print("Statistical Report")
    print("="*70)
    print(f"Population size (N):        {N:,}")
    print(f"Sample size (n):            {n:,} ({100*n/N:.4f}% of population)")
    print(f"Failures observed:          {failures}")
    print(f"Estimated failure rate:     {100*p_hat:.4f}%")
    print(f"Margin of error:            ±{100*e_hat:.4f}%")
    print(f"Confidence level:           {100*confidence_level:.1f}% (t={t:.3f})")
    print(f"Confidence interval:        [{100*ci_lower:.4f}%, {100*ci_upper:.4f}%]")
    print("="*70)
    
    return {
        "n": n,
        "N": N,
        "failures": failures,
        "p_hat": p_hat,
        "e_hat": e_hat,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "confidence_level": confidence_level
    }


def validate_statistical_assumptions(
    n: int,
    failures: int,
    verbose: bool = True
) -> Tuple[bool, str]:
    """
    Validate that statistical assumptions are met for normal approximation.
    
    For the normal approximation to binomial distribution to be valid:
    - n*p >= 5
    - n*(1-p) >= 5
    
    Args:
        n: Sample size
        failures: Number of failures
        verbose: Whether to print validation results
        
    Returns:
        Tuple of (is_valid, message)
    """
    if n == 0:
        return False, "Sample size is 0"
    
    p_hat = failures / n
    
    np_val = n * p_hat
    nq_val = n * (1 - p_hat)
    
    is_valid = (np_val >= 5) and (nq_val >= 5)
    
    if verbose:
        print(f"\nStatistical Assumption Validation:")
        print(f"  n*p = {n} * {p_hat:.4f} = {np_val:.2f} {'✓' if np_val >= 5 else '✗ (need >= 5)'}")
        print(f"  n*(1-p) = {n} * {1-p_hat:.4f} = {nq_val:.2f} {'✓' if nq_val >= 5 else '✗ (need >= 5)'}")
        print(f"  Normal approximation: {'Valid ✓' if is_valid else 'Invalid ✗'}")
    
    if not is_valid:
        if np_val < 5:
            msg = f"n*p = {np_val:.2f} < 5. Need more failures or larger sample."
        else:
            msg = f"n*(1-p) = {nq_val:.2f} < 5. Too many failures for normal approximation."
        return False, msg
    
    return True, "Statistical assumptions satisfied"


# Example usage and testing
if __name__ == "__main__":
    print("="*70)
    print("Statistical Fault Injection Utilities - Demo")
    print("="*70)
    
    # Example parameters (similar to your ViT case)
    N = 8_589_072_000  # 268K params * 32 bits * 1000 time points
    e_goal = 0.001  # 0.1% margin of error
    confidence = 0.99  # 99% confidence
    
    print(f"\nScenario: ViT-tiny Fault Injection")
    print(f"  Population (N): {N:,} possible faults")
    print(f"  Target margin (e): {e_goal} ({e_goal*100}%)")
    print(f"  Confidence level: {confidence} ({confidence*100}%)")
    
    # Conservative approach (p=0.5)
    print(f"\n{'='*70}")
    print("CONSERVATIVE APPROACH (p=0.5)")
    print('='*70)
    n_conservative = calculate_sample_size(N, e_goal, confidence, p=0.5)
    print(f"Required sample size: {n_conservative:,}")
    print(f"Percentage of population: {100*n_conservative/N:.4f}%")
    print(f"If each test takes 1 sec: {n_conservative/(3600*24):.2f} days")
    
    # Optimized approach (p=0.01, if we knew failure rate was ~1%)
    print(f"\n{'='*70}")
    print("OPTIMIZED APPROACH (p=0.01, if we knew p≈1%)")
    print('='*70)
    n_optimized = calculate_sample_size(N, e_goal, confidence, p=0.01)
    print(f"Required sample size: {n_optimized:,}")
    print(f"Percentage of population: {100*n_optimized/N:.4f}%")
    print(f"Reduction: {100*(n_conservative-n_optimized)/n_conservative:.2f}%")
    print(f"If each test takes 1 sec: {n_optimized/(3600*24):.2f} days")
    
    # Simulate measuring results
    print(f"\n{'='*70}")
    print("EXAMPLE: After injecting n_conservative faults")
    print('='*70)
    simulated_failures = int(0.012 * n_conservative)  # Simulate 1.2% failure rate
    metrics = print_statistical_report(
        n=n_conservative,
        N=N,
        failures=simulated_failures,
        confidence_level=confidence,
        description="Simulated Result"
    )
    
    # Validate assumptions
    validate_statistical_assumptions(n_conservative, simulated_failures)
    
    # Show E-P curve behavior
    print(f"\n{'='*70}")
    print("E-P CURVE DEMONSTRATION")
    print('='*70)
    e_start = 0.05  # Start with 5% margin
    p_hat_measured = 0.012  # Measured 1.2% failure rate
    e_hat_measured = calculate_error_margin(
        n=calculate_sample_size(N, e_start, confidence, p=0.5),
        N=N,
        p_hat=p_hat_measured,
        confidence_level=confidence
    )
    
    e_next = calculate_ep_curve_next_error(p_hat_measured, e_hat_measured, e_goal)
    
    print(f"Starting margin: {e_start*100}%")
    print(f"Measured P_hat: {p_hat_measured*100}%")
    print(f"Achieved margin: {e_hat_measured*100:.4f}%")
    print(f"Next margin (E-P curve): {e_next*100:.4f}%")
    print(f"Goal margin: {e_goal*100}%")
    
    print("\n" + "="*70)
    print("Utilities module ready for use!")
    print("="*70)
