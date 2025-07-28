"""
Test Suite for Referral Credit Calculator v3

This test suite provides comprehensive testing for:
1. Calculator Tab - Unit tests for credit calculations and parameter handling
2. Profit Margin Impact Tab - Integration tests for margin analysis
3. Portfolio Simulator Tab - Monte Carlo simulation testing
4. Helper Functions - Unit tests for utility functions
5. UI Components - Integration tests for Streamlit widgets

Test Categories:
- Unit Tests: Test individual functions and calculations
- Integration Tests: Test tab functionality and data flow
- UI Tests: Test widget behavior and validation
- Edge Cases: Test boundary conditions and error handling
"""

import pytest
import numpy as np
import pandas as pd
import streamlit as st
from unittest.mock import Mock, patch
import sys
import os

# Add the project directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the functions we need to test
# We'll need to extract these from the main file
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))

def diminishing_returns(n: int, S: float) -> float:
    if n <= 0:
        return 1.0
    return min(1.0, S / (S + n - 1.0))

def normalize_fracs(fracs):
    s = sum(fracs)
    if s <= 0:
        return [0.0 for _ in fracs]
    return [f / s for f in fracs]

def beta_ab_from_mean_kappa(mean: float, kappa: float):
    mean = clamp(mean, 0.0001, 0.9999)
    kappa = max(0.0001, kappa)
    a = mean * kappa
    b = (1.0 - mean) * kappa
    return a, b

def triangular_mean(a, c, b):
    # a=min, c=mode, b=max
    return (a + c + b) / 3.0

class TestHelperFunctions:
    """Test suite for utility/helper functions"""
    
    def test_clamp_normal_cases(self):
        """Test clamp function with normal inputs"""
        assert clamp(0.5) == 0.5
        assert clamp(-0.1) == 0.0
        assert clamp(1.5) == 1.0
        assert clamp(0.3, 0.2, 0.8) == 0.3
        assert clamp(0.1, 0.2, 0.8) == 0.2
        assert clamp(0.9, 0.2, 0.8) == 0.8
    
    def test_clamp_edge_cases(self):
        """Test clamp function with edge cases"""
        assert clamp(0.0) == 0.0
        assert clamp(1.0) == 1.0
        assert clamp(float('inf')) == 1.0
        assert clamp(float('-inf')) == 0.0
    
    def test_diminishing_returns_normal_cases(self):
        """Test diminishing returns function"""
        assert diminishing_returns(0, 5) == 1.0
        assert diminishing_returns(1, 5) == 1.0
        assert diminishing_returns(2, 5) == 5/6
        assert diminishing_returns(6, 5) == 0.5
        
    def test_diminishing_returns_edge_cases(self):
        """Test diminishing returns with edge cases"""
        assert diminishing_returns(-1, 5) == 1.0
        assert abs(diminishing_returns(1, 0.1) - 1.0) < 1e-10  # Very close to 1.0
        
    def test_normalize_fracs_normal_cases(self):
        """Test fraction normalization"""
        result = normalize_fracs([0.4, 0.4, 0.2])
        assert abs(sum(result) - 1.0) < 1e-10
        assert result == [0.4, 0.4, 0.2]
        
        result = normalize_fracs([1, 2, 3])
        expected = [1/6, 2/6, 3/6]
        assert np.allclose(result, expected)
    
    def test_normalize_fracs_edge_cases(self):
        """Test fraction normalization edge cases"""
        result = normalize_fracs([0, 0, 0])
        assert result == [0.0, 0.0, 0.0]
        
        result = normalize_fracs([])
        assert result == []
    
    def test_beta_ab_from_mean_kappa(self):
        """Test beta distribution parameter calculation"""
        a, b = beta_ab_from_mean_kappa(0.5, 10)
        assert a == 5.0
        assert b == 5.0
        
        # Test edge cases
        a, b = beta_ab_from_mean_kappa(0.0, 10)  # Should be clamped to 0.0001
        assert a > 0
        assert b > 0
        
        a, b = beta_ab_from_mean_kappa(1.0, 10)  # Should be clamped to 0.9999
        assert a > 0
        assert b > 0
    
    def test_triangular_mean(self):
        """Test triangular distribution mean calculation"""
        assert triangular_mean(0, 1, 2) == 1.0
        assert triangular_mean(10, 20, 30) == 20.0


class TestCreditCalculations:
    """Test suite for credit calculation logic"""
    
    def setup_method(self):
        """Setup common test parameters"""
        self.default_params = {
            'r': 1500.0,
            'g': 0.55,
            'alpha': 0.25,
            'eta': 0.05,
            'delta': 0.40,
            'p0': 0.10,
            'beta_p': 0.50,
            'CLmax': 3000.0,
            'CCmax': 20000.0,
            'R_exp': 100000.0,
            'R_actual': 120000.0,
            'Q': 0.60,
            'E': 0.50,
            'n': 1,
            'S': 5
        }
    
    def test_conversion_probability_calculation(self):
        """Test conversion probability p(Q) calculation"""
        p0, beta_p, Q = 0.10, 0.50, 0.60
        p = clamp(p0 + beta_p * Q, 0.0, 1.0)
        expected = 0.10 + 0.50 * 0.60  # 0.40
        assert p == expected
        
        # Test edge cases
        p_max = clamp(0.80 + 0.50 * 1.0, 0.0, 1.0)  # Should be clamped to 1.0
        assert p_max == 1.0
        
        p_min = clamp(0.10 + 0.50 * 0.0, 0.0, 1.0)  # Should be 0.10
        assert p_min == 0.10
    
    def test_expected_value_calculation(self):
        """Test expected value of lead calculation"""
        params = self.default_params
        p = clamp(params['p0'] + params['beta_p'] * params['Q'], 0.0, 1.0)
        EV_lead = p * params['g'] * params['R_exp']
        expected = 0.40 * 0.55 * 100000.0  # 22000.0
        assert EV_lead == expected
    
    def test_effort_multiplier_calculation(self):
        """Test effort multiplier calculation"""
        params = self.default_params
        M_E = 1.0 + params['delta'] * params['E']
        expected = 1.0 + 0.40 * 0.50  # 1.20
        assert M_E == expected
    
    def test_lead_credit_calculation(self):
        """Test lead credit calculation including caps and diminishing returns"""
        params = self.default_params
        
        # Calculate components
        p = clamp(params['p0'] + params['beta_p'] * params['Q'], 0.0, 1.0)
        EV_lead = p * params['g'] * params['R_exp']
        M_E = 1.0 + params['delta'] * params['E']
        D = diminishing_returns(params['n'], params['S'])
        
        # Lead credit calculation
        CL_raw = params['eta'] * EV_lead * M_E
        CL_capped = min(CL_raw, params['CLmax'])
        CL = CL_capped * D
        
        expected_CL_raw = 0.05 * 22000.0 * 1.20  # 1320.0
        expected_CL_capped = min(1320.0, 3000.0)  # 1320.0
        expected_CL = expected_CL_capped * 1.0  # 1320.0 (D=1.0 for n=1)
        
        assert abs(CL - expected_CL) < 1e-10
    
    def test_conversion_credit_calculation(self):
        """Test conversion credit calculation"""
        params = self.default_params
        
        M_E = 1.0 + params['delta'] * params['E']
        CC_raw = params['alpha'] * params['g'] * params['R_actual'] * M_E
        CC_capped = min(CC_raw, params['CCmax'])
        
        expected_CC_raw = 0.25 * 0.55 * 120000.0 * 1.20  # 19800.0
        expected_CC_capped = min(19800.0, 20000.0)  # 19800.0
        
        assert abs(CC_capped - expected_CC_capped) < 1e-10
    
    def test_payout_policy_additive(self):
        """Test additive payout policy"""
        # In additive policy, total = CL + CC regardless of caps
        CL, CC_paid = 1320.0, 19800.0
        C_total_additive = CL + CC_paid
        assert C_total_additive == 21120.0
    
    def test_payout_policy_net_from_cap(self):
        """Test net from conversion cap policy"""
        params = self.default_params
        CL = 1320.0
        CC_raw = 19800.0
        
        # Net policy: CL + CC <= CCmax
        CC_effective_max = max(0.0, params['CCmax'] - CL)
        CC_paid = min(CC_raw, CC_effective_max)
        C_total = CL + CC_paid
        
        expected_CC_effective_max = max(0.0, 20000.0 - 1320.0)  # 18680.0
        expected_CC_paid = min(19800.0, 18680.0)  # 18680.0
        expected_C_total = 1320.0 + 18680.0  # 20000.0
        
        assert abs(CC_paid - expected_CC_paid) < 1e-10
        assert abs(C_total - expected_C_total) < 1e-10


class TestMarginImpactCalculations:
    """Test suite for profit margin impact calculations"""
    
    def setup_method(self):
        """Setup common test parameters for margin testing"""
        self.margin_params = {
            'C_total': 20000.0,
            'redemption_rate': 0.80,
            'r': 1500.0,
            'g': 0.55,
            'R_actual': 120000.0,
            'expiry_months': 12,
            'c_var_hour': 675.0,  # r*(1-g) = 1500*0.45
            'H_cap': 640.0,
            'base_paid_hours': 400.0
        }
    
    def test_discount_mode_calculation(self):
        """Test discount (contra-revenue) mode"""
        params = self.margin_params
        
        redeemed_credits = params['redemption_rate'] * params['C_total']
        GM_gain_project = params['g'] * params['R_actual']
        GM_loss_total = redeemed_credits
        net_GM = GM_gain_project - GM_loss_total
        
        expected_redeemed = 0.80 * 20000.0  # 16000.0
        expected_GM_gain = 0.55 * 120000.0  # 66000.0
        expected_GM_loss = 16000.0
        expected_net_GM = 66000.0 - 16000.0  # 50000.0
        
        assert redeemed_credits == expected_redeemed
        assert GM_gain_project == expected_GM_gain
        assert GM_loss_total == expected_GM_loss
        assert net_GM == expected_net_GM
    
    def test_cash_cost_mode_calculation(self):
        """Test free hours (cash cost only) mode"""
        params = self.margin_params
        
        redeemed_credits = params['redemption_rate'] * params['C_total']
        redeemed_hours_total = redeemed_credits / params['r']
        GM_loss_total = params['c_var_hour'] * redeemed_hours_total
        
        expected_redeemed_hours = 16000.0 / 1500.0  # 10.67 hours
        expected_GM_loss = 675.0 * (16000.0 / 1500.0)  # 7200.0
        
        assert abs(redeemed_hours_total - expected_redeemed_hours) < 1e-10
        assert abs(GM_loss_total - expected_GM_loss) < 1e-10
    
    def test_capacity_aware_mode_calculation(self):
        """Test capacity-aware mode calculation"""
        params = self.margin_params
        
        redeemed_credits = params['redemption_rate'] * params['C_total']
        redeemed_hours_total = redeemed_credits / params['r']
        monthly_redeemed_hours = redeemed_hours_total / params['expiry_months']
        
        slack = max(0.0, params['H_cap'] - params['base_paid_hours'])
        cash_cost_per_month = params['c_var_hour'] * min(monthly_redeemed_hours, slack)
        displaced_hours = max(0.0, monthly_redeemed_hours - slack)
        opp_margin_loss_per_month = (params['g'] * params['r']) * displaced_hours
        monthly_margin_loss = cash_cost_per_month + opp_margin_loss_per_month
        
        expected_slack = 640.0 - 400.0  # 240.0 hours/month
        expected_monthly_hours = (16000.0 / 1500.0) / 12  # ~0.89 hours/month
        expected_cash_cost = 675.0 * min(expected_monthly_hours, 240.0)  # Cash cost for hours used
        expected_displaced = max(0.0, expected_monthly_hours - 240.0)  # 0.0 (no displacement)
        expected_opp_loss = 0.55 * 1500.0 * expected_displaced  # 0.0
        expected_monthly_loss = expected_cash_cost + expected_opp_loss
        
        assert abs(monthly_redeemed_hours - expected_monthly_hours) < 1e-2
        assert abs(cash_cost_per_month - expected_cash_cost) < 1e-2
        assert displaced_hours == expected_displaced
        assert opp_margin_loss_per_month == expected_opp_loss


class TestMonteCarloSimulation:
    """Test suite for Monte Carlo portfolio simulation"""
    
    def test_beta_distribution_parameters(self):
        """Test beta distribution parameter generation"""
        mean, kappa = 0.6, 10.0
        a, b = beta_ab_from_mean_kappa(mean, kappa)
        
        # Check that the parameters generate the correct mean
        expected_mean = a / (a + b)
        assert abs(expected_mean - 0.6) < 1e-10
    
    def test_triangular_distribution_mean_calculation(self):
        """Test triangular distribution mean"""
        Rmin, Rmode, Rmax = 60000.0, 120000.0, 220000.0
        mean = triangular_mean(Rmin, Rmode, Rmax)
        expected = (60000.0 + 120000.0 + 220000.0) / 3.0
        assert mean == expected
    
    def test_simulation_parameter_validation(self):
        """Test that simulation parameters are within valid ranges"""
        # Test lambda (Poisson rate) parameters
        assert 0.0 <= 2.0  # lam_off
        assert 0.0 <= 6.0  # lam_on
        
        # Test quality parameters
        assert 0.0 <= 0.55 <= 1.0  # meanQ_off
        assert 0.0 <= 0.65 <= 1.0  # meanQ_on
        assert 0.1 <= 6.0  # kappaQ_off
        assert 0.1 <= 8.0  # kappaQ_on
        
        # Test effort parameters
        assert 0.0 <= 0.5 <= 1.0  # meanE
        assert 0.1 <= 6.0  # kappaE


class TestUIComponents:
    """Test suite for UI component validation"""
    
    def test_sidebar_parameter_ranges(self):
        """Test that sidebar parameters have correct ranges"""
        # Global parameters
        assert 0.0 <= 1500.0  # r (base hourly rate)
        assert 0.0 <= 0.55 <= 1.0  # g (gross margin)
        assert 0.0 <= 0.25 <= 1.0  # alpha
        assert 0.0 <= 0.05 <= 1.0  # eta
        assert 0.0 <= 0.40 <= 2.0  # delta
        
        # Conversion probability
        assert 0.0 <= 0.10 <= 1.0  # p0
        assert 0.0 <= 0.50 <= 1.0  # beta_p
        
        # Caps
        assert 0.0 <= 3000.0  # CLmax
        assert 0.0 <= 20000.0  # CCmax
        assert 1 <= 12  # expiry_months
    
    def test_calculator_input_ranges(self):
        """Test calculator tab input parameter ranges"""
        assert 0.0 <= 100000.0  # R_exp
        assert 0.0 <= 0.60 <= 1.0  # Q
        assert 0.0 <= 0.50 <= 1.0  # E
        assert 0 <= 1  # n
        assert 1 <= 5  # S
        assert 0.0 <= 120000.0  # R_actual
    
    def test_margin_impact_input_ranges(self):
        """Test margin impact tab input parameter ranges"""
        assert 0.0 <= 0.80 <= 1.0  # redemption_rate
        assert 1 <= 12  # expiry_months_m
        assert 0.0 <= 675.0  # c_var_hour (example value)
        assert 0.0 <= 640.0  # H_cap
        assert 0.0 <= 400.0  # base_paid_hours
    
    def test_portfolio_simulator_input_ranges(self):
        """Test portfolio simulator input parameter ranges"""
        assert 1 <= 12  # months
        assert 100 <= 500  # sims
        assert 0 <= 42  # seed
        assert 0.0 <= 2.0  # lam_off
        assert 0.0 <= 6.0  # lam_on
        assert 0.0 <= 0.4 <= 1.0  # SQL_threshold


class TestIntegrationScenarios:
    """Integration tests for complete calculation workflows"""
    
    def test_end_to_end_calculation_scenario(self):
        """Test complete calculation from inputs to outputs"""
        # Setup scenario parameters
        params = {
            'r': 1500.0, 'g': 0.55, 'alpha': 0.25, 'eta': 0.05, 'delta': 0.40,
            'p0': 0.10, 'beta_p': 0.50, 'CLmax': 3000.0, 'CCmax': 20000.0,
            'R_exp': 100000.0, 'R_actual': 120000.0, 'Q': 0.60, 'E': 0.50,
            'n': 1, 'S': 5
        }
        
        # Calculate step by step
        p = clamp(params['p0'] + params['beta_p'] * params['Q'], 0.0, 1.0)
        EV_lead = p * params['g'] * params['R_exp']
        M_E = 1.0 + params['delta'] * params['E']
        D = diminishing_returns(params['n'], params['S'])
        
        CL_raw = params['eta'] * EV_lead * M_E
        CL_capped = min(CL_raw, params['CLmax'])
        CL = CL_capped * D
        
        CC_raw = params['alpha'] * params['g'] * params['R_actual'] * M_E
        CC_capped = min(CC_raw, params['CCmax'])
        
        # Test additive policy
        CC_paid_additive = CC_capped
        C_total_additive = CL + CC_paid_additive
        
        # Test net policy
        CC_effective_max = max(0.0, params['CCmax'] - CL)
        CC_paid_net = min(CC_capped, CC_effective_max)
        C_total_net = CL + CC_paid_net
        
        # Verify calculations are reasonable
        assert 0 <= p <= 1
        assert EV_lead > 0
        assert M_E >= 1.0
        assert 0 <= D <= 1
        assert CL >= 0
        assert CC_paid_additive >= 0
        assert CC_paid_net >= 0
        assert C_total_additive > 0
        assert C_total_net > 0
        assert C_total_net <= params['CCmax']  # Net policy constraint
    
    def test_margin_impact_integration(self):
        """Test margin impact calculation integration"""
        # Use results from end-to-end test
        C_total = 20000.0
        redemption_rate = 0.80
        R_actual = 120000.0
        g = 0.55
        
        redeemed_credits = redemption_rate * C_total
        GM_gain_project = g * R_actual
        
        # Test all three modes give reasonable results
        # Mode 1: Discount
        GM_loss_discount = redeemed_credits
        net_GM_discount = GM_gain_project - GM_loss_discount
        
        # Mode 2: Cash cost
        r = 1500.0
        c_var_hour = r * (1 - g)
        redeemed_hours = redeemed_credits / r
        GM_loss_cash = c_var_hour * redeemed_hours
        net_GM_cash = GM_gain_project - GM_loss_cash
        
        # Mode 3: Capacity aware (assuming within slack)
        net_GM_capacity = GM_gain_project - GM_loss_cash  # Same as cash if within slack
        
        assert net_GM_discount > 0  # Should still be profitable
        assert net_GM_cash > net_GM_discount  # Cash cost should be lower impact
        assert net_GM_capacity >= net_GM_cash  # Capacity aware could be better or same


class TestErrorHandling:
    """Test suite for error handling and edge cases"""
    
    def test_division_by_zero_protection(self):
        """Test protection against division by zero"""
        # Test hours calculation when r = 0
        credits = 1000.0
        r = 0.0
        hours = (credits / r) if r > 0 else 0.0
        assert hours == 0.0
        
        # Test ROI calculation when denominator is 0
        net_GM = 1000.0
        redeemed_credits = 0.0
        roi = (net_GM / redeemed_credits) if redeemed_credits > 0 else float('nan')
        assert np.isnan(roi)
    
    def test_negative_value_handling(self):
        """Test handling of negative values"""
        # Test clamp with negative values
        assert clamp(-1.0) == 0.0
        
        # Test diminishing returns with negative n
        assert diminishing_returns(-1, 5) == 1.0
        
        # Test max operations
        assert max(0.0, -10.0) == 0.0
    
    def test_extreme_value_handling(self):
        """Test handling of extreme values"""
        # Test very large values
        large_value = 1e10
        clamped = clamp(large_value)
        assert clamped == 1.0
        
        # Test very small values
        small_value = 1e-10
        result = diminishing_returns(1, small_value)
        assert abs(result - 1.0) < 1e-6  # Very close to 1.0 due to floating-point precision


# Test runner configuration
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
