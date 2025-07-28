import numpy as np

# Test milestone calculations with your scenario values
def test_milestone_calculation():
    print("🧪 Testing Milestone Calculations")
    print("=" * 40)
    
    # Your scenario values
    phi1, phi2, phi3 = 0.4, 0.4, 0.2
    CC_paid = 2221.6650600000003
    
    print(f"Input milestone shares: φ₁={phi1}, φ₂={phi2}, φ₃={phi3}")
    print(f"CC_paid: {CC_paid:,.2f}")
    print()
    
    # Normalize fractions (same as app)
    def normalize_fracs(fracs):
        s = sum(fracs)
        if s <= 0:
            return [0.0 for _ in fracs]
        return [f / s for f in fracs]
    
    phi1_norm, phi2_norm, phi3_norm = normalize_fracs([phi1, phi2, phi3])
    print(f"Normalized shares: φ₁={phi1_norm:.3f}, φ₂={phi2_norm:.3f}, φ₃={phi3_norm:.3f}")
    print(f"Sum of normalized shares: {sum([phi1_norm, phi2_norm, phi3_norm]):.3f}")
    print()
    
    # Calculate milestones (same as app)
    phi = np.array([phi1_norm, phi2_norm, phi3_norm], dtype=float)
    milestone_amounts = (CC_paid * phi).round(2)
    
    print("Milestone calculations:")
    print(f"Deposit (40%):      {milestone_amounts[0]:,.2f}")
    print(f"Design (40%):       {milestone_amounts[1]:,.2f}")
    print(f"Final (20%):        {milestone_amounts[2]:,.2f}")
    print(f"Total milestones:   {milestone_amounts.sum():,.2f}")
    print()
    
    difference = abs(CC_paid - milestone_amounts.sum())
    print(f"Difference from CC_paid: {difference:,.2f}")
    
    if difference < 0.01:
        print("✅ PASS: Milestones add up correctly")
    else:
        print("❌ FAIL: Milestones don't add up correctly")
    
    return milestone_amounts, difference

if __name__ == "__main__":
    test_milestone_calculation()
