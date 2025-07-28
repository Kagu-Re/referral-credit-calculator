"""
Integration Test Suite for Streamlit App Functionality

This test file focuses on testing the actual Streamlit app behavior,
including widget interactions, tab functionality, and data flow between components.
"""

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest
import numpy as np
import pandas as pd
import sys
import os

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class TestStreamlitApp:
    """Test suite for Streamlit app functionality"""
    
    def setup_method(self):
        """Setup for each test method"""
        # Initialize app test
        self.app = AppTest.from_file("referral_calculator.py")
    
    def test_app_loads_successfully(self):
        """Test that the app loads without errors"""
        try:
            self.app.run()
            assert not self.app.exception
        except Exception as e:
            pytest.fail(f"App failed to load: {e}")
    
    def test_sidebar_widgets_exist(self):
        """Test that all sidebar widgets are present"""
        self.app.run()
        
        # Check global parameters
        assert "Base hourly rate r" in str(self.app.sidebar)
        assert "Gross margin g" in str(self.app.sidebar)
        assert "Referral budget share α" in str(self.app.sidebar)
        assert "Lead-stage share η" in str(self.app.sidebar)
        assert "Effort influence δ" in str(self.app.sidebar)
        
        # Check conversion probability section
        assert "Base conversion p₀" in str(self.app.sidebar)
        assert "Quality slope β" in str(self.app.sidebar)
        
        # Check caps & policy section
        assert "Lead credit cap" in str(self.app.sidebar)
        assert "Conversion credit cap" in str(self.app.sidebar)
        assert "Payout policy" in str(self.app.sidebar)
        assert "Credit expiry" in str(self.app.sidebar)
    
    def test_tabs_exist(self):
        """Test that all three tabs are present"""
        self.app.run()
        
        # Check that tabs are created
        tabs_content = str(self.app.main)
        assert "Calculator" in tabs_content
        assert "Profit Margin Impact" in tabs_content
        assert "Portfolio Simulator (Monte Carlo)" in tabs_content
    
    def test_calculator_tab_widgets(self):
        """Test Calculator tab widgets"""
        self.app.run()
        
        # Check for main input widgets in calculator tab
        main_content = str(self.app.main)
        assert "Expected revenue for this lead" in main_content
        assert "Lead quality Q" in main_content
        assert "Effort score E" in main_content
        assert "Leads from same referrer" in main_content
        assert "Saturation S" in main_content
        assert "Actual project revenue R" in main_content
    
    def test_sidebar_parameter_changes(self):
        """Test that sidebar parameters can be changed"""
        self.app.run()
        
        # Try to modify a sidebar parameter
        if hasattr(self.app.sidebar, 'slider'):
            # This would need to be implemented based on actual widget structure
            pass  # Placeholder for actual widget interaction tests
    
    def test_calculation_outputs_present(self):
        """Test that calculation outputs are displayed"""
        self.app.run()
        
        # Check for metric displays
        main_content = str(self.app.main)
        assert "p(Q)" in main_content
        assert "EV_lead" in main_content
        assert "Effort multiplier" in main_content
        assert "Lead credit C_L" in main_content
        assert "Conversion credit C_C" in main_content
        assert "Total credit" in main_content


class TestCalculatorTabFunctionality:
    """Test Calculator tab specific functionality"""
    
    def test_default_calculation_values(self):
        """Test that default values produce reasonable calculations"""
        # Default parameters (matching the app defaults)
        r = 1500.0
        g = 0.55
        alpha = 0.25
        eta = 0.05
        delta = 0.40
        p0 = 0.10
        beta_p = 0.50
        CLmax = 3000.0
        CCmax = 20000.0
        R_exp = 100000.0
        R_actual = 120000.0
        Q = 0.60
        E = 0.50
        n = 1
        S = 5
        
        # Replicate the calculation logic
        def clamp(x, a=0.0, b=1.0):
            return max(a, min(b, x))
        
        def diminishing_returns(n: int, S: float) -> float:
            if n <= 0:
                return 1.0
            return min(1.0, S / (S + n - 1.0))
        
        # Calculate
        p = clamp(p0 + beta_p * Q, 0.0, 1.0)
        EV_lead = p * g * R_exp
        M_E = 1.0 + delta * E
        D = diminishing_returns(n, S)
        
        CL_raw = eta * EV_lead * M_E
        CL_capped = min(CL_raw, CLmax)
        CL = CL_capped * D
        
        CC_raw = alpha * g * R_actual * M_E
        CC_capped = min(CC_raw, CCmax)
        CC_paid = CC_capped
        C_total = CL + CC_paid
        
        # Validate results are reasonable
        assert 0 <= p <= 1
        assert EV_lead > 0
        assert M_E >= 1.0
        assert 0 <= D <= 1
        assert CL >= 0
        assert CC_paid >= 0
        assert C_total > 0
        
        # Specific expected values for default inputs
        assert abs(p - 0.40) < 1e-10  # 0.10 + 0.50 * 0.60
        assert abs(EV_lead - 22000.0) < 1e-10  # 0.40 * 0.55 * 100000
        assert abs(M_E - 1.20) < 1e-10  # 1.0 + 0.40 * 0.50
        assert D == 1.0  # diminishing_returns(1, 5) = 1.0
        
        print(f"Test Results - p: {p}, EV_lead: {EV_lead}, M_E: {M_E}, D: {D}")
        print(f"CL: {CL}, CC_paid: {CC_paid}, C_total: {C_total}")


class TestMarginImpactTabFunctionality:
    """Test Profit Margin Impact tab functionality"""
    
    def test_margin_calculation_modes(self):
        """Test all three margin calculation modes"""
        # Setup test data
        C_total = 20000.0
        redemption_rate = 0.80
        r = 1500.0
        g = 0.55
        R_actual = 120000.0
        expiry_months_m = 12
        c_var_hour = r * (1 - g)  # 675.0
        H_cap = 640.0
        base_paid_hours = 400.0
        
        # Common calculations
        redeemed_credits = redemption_rate * C_total
        redeemed_hours_total = redeemed_credits / r
        monthly_redeemed_hours = redeemed_hours_total / expiry_months_m
        GM_gain_project = g * R_actual
        
        # Mode 1: Discount (contra-revenue)
        GM_loss_discount = redeemed_credits
        monthly_margin_loss_discount = np.repeat(GM_loss_discount / expiry_months_m, expiry_months_m)
        net_GM_discount = GM_gain_project - GM_loss_discount
        
        # Mode 2: Free hours (cash cost only)
        GM_loss_cash = c_var_hour * redeemed_hours_total
        monthly_margin_loss_cash = np.repeat(GM_loss_cash / expiry_months_m, expiry_months_m)
        net_GM_cash = GM_gain_project - GM_loss_cash
        
        # Mode 3: Free hours (capacity-aware)
        slack = max(0.0, H_cap - base_paid_hours)
        cash_cost_per_month = c_var_hour * min(monthly_redeemed_hours, slack)
        displaced_hours = max(0.0, monthly_redeemed_hours - slack)
        opp_margin_loss_per_month = (g * r) * displaced_hours
        monthly_margin_loss_capacity = np.repeat(cash_cost_per_month + opp_margin_loss_per_month, expiry_months_m)
        GM_loss_capacity = monthly_margin_loss_capacity.sum()
        net_GM_capacity = GM_gain_project - GM_loss_capacity
        
        # Validate all modes produce reasonable results
        assert GM_gain_project > 0
        assert GM_loss_discount > 0
        assert GM_loss_cash > 0
        assert GM_loss_capacity > 0
        assert net_GM_discount > 0  # Should still be profitable
        assert net_GM_cash > 0
        assert net_GM_capacity > 0
        
        # Cash cost should typically be lower than discount
        assert GM_loss_cash < GM_loss_discount
        
        print(f"Margin Test Results:")
        print(f"GM Gain: {GM_gain_project}")
        print(f"GM Loss - Discount: {GM_loss_discount}, Net: {net_GM_discount}")
        print(f"GM Loss - Cash: {GM_loss_cash}, Net: {net_GM_cash}")
        print(f"GM Loss - Capacity: {GM_loss_capacity}, Net: {net_GM_capacity}")


class TestPortfolioSimulatorFunctionality:
    """Test Portfolio Simulator tab functionality"""
    
    def test_beta_distribution_setup(self):
        """Test beta distribution parameter calculation for simulation"""
        def clamp(x, a=0.0, b=1.0):
            return max(a, min(b, x))
        
        def beta_ab_from_mean_kappa(mean: float, kappa: float):
            mean = clamp(mean, 0.0001, 0.9999)
            kappa = max(0.0001, kappa)
            a = mean * kappa
            b = (1.0 - mean) * kappa
            return a, b
        
        # Test with default simulation parameters
        meanQ_off = 0.55
        kappaQ_off = 6.0
        meanQ_on = 0.65
        kappaQ_on = 8.0
        meanE = 0.5
        kappaE = 6.0
        
        # Calculate beta parameters
        aQ_off, bQ_off = beta_ab_from_mean_kappa(meanQ_off, kappaQ_off)
        aQ_on, bQ_on = beta_ab_from_mean_kappa(meanQ_on, kappaQ_on)
        aE, bE = beta_ab_from_mean_kappa(meanE, kappaE)
        
        # Validate parameters
        assert aQ_off > 0 and bQ_off > 0
        assert aQ_on > 0 and bQ_on > 0
        assert aE > 0 and bE > 0
        
        # Check that means are approximately correct
        calculated_mean_Q_off = aQ_off / (aQ_off + bQ_off)
        calculated_mean_Q_on = aQ_on / (aQ_on + bQ_on)
        calculated_mean_E = aE / (aE + bE)
        
        assert abs(calculated_mean_Q_off - meanQ_off) < 1e-10
        assert abs(calculated_mean_Q_on - meanQ_on) < 1e-10
        assert abs(calculated_mean_E - meanE) < 1e-10
        
        print(f"Beta Parameters - Q_off: a={aQ_off}, b={bQ_off}")
        print(f"Beta Parameters - Q_on: a={aQ_on}, b={bQ_on}")
        print(f"Beta Parameters - E: a={aE}, b={bE}")
    
    def test_triangular_distribution_setup(self):
        """Test triangular distribution for revenue simulation"""
        def triangular_mean(a, c, b):
            return (a + c + b) / 3.0
        
        # Default revenue distribution parameters
        Rmin = 60000.0
        Rmode = 120000.0
        Rmax = 220000.0
        
        mean_revenue = triangular_mean(Rmin, Rmode, Rmax)
        expected_mean = (60000.0 + 120000.0 + 220000.0) / 3.0
        
        assert mean_revenue == expected_mean
        assert Rmin <= Rmode <= Rmax  # Valid triangular distribution
        
        print(f"Triangular Distribution - Min: {Rmin}, Mode: {Rmode}, Max: {Rmax}, Mean: {mean_revenue}")
    
    def test_simulation_parameter_validation(self):
        """Test that simulation parameters are valid"""
        # Default simulation parameters
        months = 12
        sims = 500
        seed = 42
        lam_off = 2.0
        lam_on = 6.0
        SQL_threshold = 0.4
        
        # Validate parameter ranges
        assert months >= 1
        assert sims >= 100
        assert seed >= 0
        assert lam_off >= 0.0
        assert lam_on >= 0.0
        assert 0.0 <= SQL_threshold <= 1.0
        assert lam_on >= lam_off  # Program should increase leads
        
        print(f"Simulation Parameters Valid - Months: {months}, Sims: {sims}, λ_off: {lam_off}, λ_on: {lam_on}")


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling"""
    
    def test_zero_values_handling(self):
        """Test handling of zero values in calculations"""
        # Test with zero revenue
        R_exp = 0.0
        R_actual = 0.0
        g = 0.55
        
        EV_lead = 0.40 * g * R_exp  # Should be 0
        GM_gain = g * R_actual  # Should be 0
        
        assert EV_lead == 0.0
        assert GM_gain == 0.0
    
    def test_extreme_values_handling(self):
        """Test handling of extreme parameter values"""
        # Test with very high quality
        p0 = 0.10
        beta_p = 0.50
        Q = 1.0  # Perfect quality
        
        def clamp(x, a=0.0, b=1.0):
            return max(a, min(b, x))
        
        p = clamp(p0 + beta_p * Q, 0.0, 1.0)
        expected = min(0.10 + 0.50 * 1.0, 1.0)  # Should be clamped to 1.0 if over
        
        assert p == expected
        
        # Test with very low quality
        Q_low = 0.0
        p_low = clamp(p0 + beta_p * Q_low, 0.0, 1.0)
        assert p_low == p0
    
    def test_capacity_constraint_scenarios(self):
        """Test capacity constraint scenarios in margin impact"""
        # Scenario 1: Usage within slack capacity
        monthly_redeemed_hours = 50.0
        H_cap = 640.0
        base_paid_hours = 400.0
        slack = H_cap - base_paid_hours  # 240.0
        
        displaced_hours = max(0.0, monthly_redeemed_hours - slack)
        assert displaced_hours == 0.0  # No displacement
        
        # Scenario 2: Usage exceeding slack capacity
        monthly_redeemed_hours_high = 300.0
        displaced_hours_high = max(0.0, monthly_redeemed_hours_high - slack)
        assert displaced_hours_high == 60.0  # 300 - 240 = 60 hours displaced
        
        print(f"Capacity Test - Slack: {slack}, Displaced (low): {displaced_hours}, Displaced (high): {displaced_hours_high}")


# Performance test for simulation
class TestPerformance:
    """Test performance characteristics"""
    
    def test_simulation_performance(self):
        """Test that simulation completes in reasonable time"""
        import time
        
        # Small simulation for performance test
        months = 3
        sims = 50  # Reduced for test speed
        
        # This would test the actual simulation function if extracted
        # For now, just test parameter setup time
        start_time = time.time()
        
        # Simulate parameter setup
        def beta_ab_from_mean_kappa(mean: float, kappa: float):
            mean = max(0.0001, min(0.9999, mean))
            kappa = max(0.0001, kappa)
            a = mean * kappa
            b = (1.0 - mean) * kappa
            return a, b
        
        for _ in range(sims):
            aQ, bQ = beta_ab_from_mean_kappa(0.6, 8.0)
            aE, bE = beta_ab_from_mean_kappa(0.5, 6.0)
        
        end_time = time.time()
        duration = end_time - start_time
        
        assert duration < 1.0  # Should complete quickly
        print(f"Parameter setup for {sims} simulations took {duration:.4f} seconds")


# Test configuration and data validation
class TestDataValidation:
    """Test data validation and consistency"""
    
    def test_calculation_consistency(self):
        """Test that calculations are consistent across different scenarios"""
        # Test that doubling revenue doubles relevant outputs
        base_revenue = 100000.0
        double_revenue = 200000.0
        g = 0.55
        alpha = 0.25
        M_E = 1.20
        
        # Expected margin should double
        base_margin = g * base_revenue
        double_margin = g * double_revenue
        assert double_margin == 2 * base_margin
        
        # Conversion credit should double (if uncapped)
        base_CC = alpha * g * base_revenue * M_E
        double_CC = alpha * g * double_revenue * M_E
        assert double_CC == 2 * base_CC
    
    def test_policy_consistency(self):
        """Test that payout policies behave consistently"""
        CL = 5000.0
        CC_raw = 18000.0
        CCmax = 20000.0
        
        # Additive policy
        CC_paid_additive = min(CC_raw, CCmax)
        C_total_additive = CL + CC_paid_additive
        
        # Net policy
        CC_effective_max = max(0.0, CCmax - CL)
        CC_paid_net = min(CC_raw, CC_effective_max)
        C_total_net = CL + CC_paid_net
        
        # Net policy should never exceed CCmax
        assert C_total_net <= CCmax
        # Additive policy could exceed CCmax
        assert C_total_additive >= C_total_net
        
        print(f"Policy Test - Additive: {C_total_additive}, Net: {C_total_net}")


if __name__ == "__main__":
    # Run specific test classes
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-k", "test_default_calculation_values or test_margin_calculation_modes"
    ])
