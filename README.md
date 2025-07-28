# Referral Credit Calculator

A Streamlit application for calculating referral credits in a loyalty program based on lead quality, conversion probability, and realized revenue.

## Features

- **Lead Credits**: Calculated based on expected margin and conversion probability
- **Conversion Credits**: Based on realized revenue with milestone-based payouts
- **Configurable Parameters**: Adjustable rates, margins, caps, and policies
- **What-if Analysis**: Visualize credit variations based on lead quality
- **Export Functionality**: Download scenarios as JSON for record-keeping

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:
```bash
streamlit run referral_calculator.py
```

The application will open in your default web browser at `http://localhost:8501`

## How It Works

### Global Parameters (Sidebar)
- **Base hourly rate (r)**: Conversion rate from credits to hours
- **Gross margin (g)**: Fraction of revenue that is margin after delivery costs
- **Referral budget share (α)**: Share of margin allocated to referral credits per conversion
- **Lead-stage share (η)**: Share of expected margin for lead-stage credits
- **Effort influence (δ)**: Multiplier effect of referrer effort

### Conversion Probability Model
- **Base conversion (p₀)**: Baseline conversion probability
- **Quality slope (β)**: How much lead quality affects conversion probability
- Formula: `p(Q) = clamp(p₀ + β·Q, 0, 1)`

### Credit Calculations

#### Lead Credits (C_L)
- Based on expected value: `η × (conversion_probability × margin × expected_revenue) × effort_multiplier`
- Subject to caps and diminishing returns for multiple leads from same referrer

#### Conversion Credits (C_C)
- Based on actual revenue: `α × margin × actual_revenue × effort_multiplier`
- Paid out across milestones (deposit, design sign-off, final payment)
- Subject to caps and payout policy constraints

### Payout Policies
1. **Additive**: Lead credits + conversion credits (separate caps)
2. **Net from conversion cap**: Total credits capped at conversion credit maximum

### Optional Features
- **Upsell Kicker**: Additional credits for repeat/upsell business
- **Milestone Split**: Configurable payout timing across project phases

## Model Components

### Diminishing Returns
Multiple leads from the same referrer receive diminishing credit amounts:
```
D(n) = min(1, S / (S + n - 1))
```
Where `S` is the saturation parameter and `n` is the number of leads.

### Effort Multiplier
Referrer actions (warm intro, budget sharing, etc.) increase credit amounts:
```
M_E = 1 + δE
```
Where `δ` is effort influence and `E` is the effort score (0-1).

## Export Format

Scenarios can be exported as JSON containing:
- All global parameters
- Input values for the specific lead/project
- Calculated results including credits, hours, and milestone breakdown
- Upsell credits (if applicable)

## Notes

This is a planning and modeling tool. Ensure your published referral policy includes:
- Eligibility requirements
- Attribution rules
- Clawback conditions
- Credit redemption process
- Expiry terms
