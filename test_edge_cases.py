import numpy as np

def normalize_fracs(fracs):
    s = sum(fracs)
    if s <= 0:
        return [0.0 for _ in fracs]
    return [f / s for f in fracs]

def test_edge_cases():
    print("🧪 Testing Edge Cases for Milestone Calculations")
    print("=" * 50)
    
    test_cases = [
        # (phi1, phi2, phi3, CC_paid, description)
        (0.33, 0.33, 0.34, 1000.0, "Uneven splits"),
        (0.5, 0.3, 0.2, 1234.56, "Decimal amounts"),
        (1.0, 0.0, 0.0, 500.0, "All to first milestone"),
        (0.1, 0.1, 0.8, 999.99, "Most to final milestone"),
        (0.333333, 0.333333, 0.333334, 3000.0, "Equal thirds"),
        (0.0, 0.0, 0.0, 1000.0, "Zero inputs"),
    ]
    
    for i, (phi1, phi2, phi3, CC_paid, desc) in enumerate(test_cases, 1):
        print(f"\n🔬 Test Case {i}: {desc}")
        print(f"Input: φ₁={phi1}, φ₂={phi2}, φ₃={phi3}, CC_paid={CC_paid}")
        
        # Normalize
        phi1_norm, phi2_norm, phi3_norm = normalize_fracs([phi1, phi2, phi3])
        
        # Calculate milestones
        phi = np.array([phi1_norm, phi2_norm, phi3_norm], dtype=float)
        milestone_amounts = (CC_paid * phi).round(2)
        
        total = milestone_amounts.sum()
        difference = abs(CC_paid - total)
        
        print(f"Milestones: {milestone_amounts[0]:.2f}, {milestone_amounts[1]:.2f}, {milestone_amounts[2]:.2f}")
        print(f"Total: {total:.2f}, Difference: {difference:.2f}")
        
        if difference < 0.01:
            print("✅ PASS")
        else:
            print("❌ FAIL - Rounding error detected!")

if __name__ == "__main__":
    test_edge_cases()
