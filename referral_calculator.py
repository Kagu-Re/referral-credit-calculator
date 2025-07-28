import math
import numpy as np
import pandas as pd
import streamlit as st
import json

st.set_page_config(
    page_title="Referral Credit Calculator v2", 
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/Kagu-Re/referral-credit-calculator',
        'Report a bug': "https://github.com/Kagu-Re/referral-credit-calculator/issues",
        'About': "# Referral Credit Calculator v2\nA tool for modeling loyalty program credits with profit margin impact analysis."
    }
)

st.title("💰 Referral Credit Calculator v2")
st.caption("🎯 Model: lead credits based on expected margin and conversion probability; conversion credits based on realized revenue. Now with **Profit Margin Impact** analysis.")

# Add a public info banner
st.info("🌟 **Enhanced Demo** - This calculator helps you design fair and profitable referral programs with margin impact analysis. All calculations happen in your browser - no data is stored or shared.")

# -----------------------------
# Helpers
# -----------------------------
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))

def diminishing_returns(n: int, S: float) -> float:
    # D(n) = min(1, S / (S + n - 1)) for n>=1; if n==0, return 1
    if n <= 0:
        return 1.0
    return min(1.0, S / (S + n - 1.0))

def normalize_fracs(fracs):
    s = sum(fracs)
    if s <= 0:
        return [0.0 for _ in fracs]
    return [f / s for f in fracs]

# -----------------------------
# Sidebar: Global params
# -----------------------------
with st.sidebar:
    st.header("Global Parameters")

    r = st.number_input("Base hourly rate r", min_value=0.0, value=1500.0, step=50.0, 
                       help="💰 How much one credit is worth in your currency. Example: If r=1500, then 3000 credits = 2 hours of work value.")
    g = st.slider("Gross margin g", 0.0, 1.0, 0.55, 0.01, 
                 help="📊 What percentage of each sale is actual profit after costs. Example: 55% means if you sell $100k, $55k is profit.")
    alpha = st.slider("Referral budget share α (of margin, per conversion)", 0.0, 1.0, 0.25, 0.01,
                     help="🎯 How much of your profit you'll share when someone converts. Example: 25% means you give away 1/4 of profit as referral rewards.")
    eta = st.slider("Lead-stage share η (of expected margin)", 0.0, 1.0, 0.05, 0.005,
                   help="🌱 Reward for just bringing a lead (before they buy). Example: 5% means you pay 5% of expected profit just for the introduction.")
    delta = st.slider("Effort influence δ", 0.0, 2.0, 0.40, 0.05, 
                     help="⚡ Bonus multiplier for extra effort. Example: 40% means someone who does warm intros gets 40% more credits than cold referrals.")

    st.markdown("---")
    st.subheader("Conversion probability")
    p0 = st.slider("Base conversion p₀", 0.0, 1.0, 0.10, 0.01,
                  help="🎲 Your typical conversion rate. Example: 10% means normally 1 out of 10 leads becomes a customer.")
    beta = st.slider("Quality slope β", 0.0, 1.0, 0.50, 0.01,
                    help="📈 How much lead quality matters. Example: 50% means a perfect quality lead (Q=1) has 50% higher conversion chance than base rate.")
    st.caption("🧮 Formula: p(Q) = clamp(p₀ + β·Q, 0, 1) — Better quality leads = higher conversion chance")

    st.markdown("---")
    st.subheader("Caps & policy")
    CLmax = st.number_input("Lead credit cap (C_L,max)", min_value=0.0, value=3000.0, step=100.0,
                           help="🚫 Maximum credits you'll pay for just bringing a lead. Example: 3000 credits max, even for a $1M opportunity.")
    CCmax = st.number_input("Conversion credit cap (C_C,max)", min_value=0.0, value=20000.0, step=100.0,
                           help="🏆 Maximum credits you'll pay when someone actually converts. Example: 20k credits max, even on huge deals.")
    payout_policy = st.selectbox("Payout policy", ["Additive (C_L + C_C)", "Net from conversion cap (C_L + C_C ≤ C_C,max)"],
                                help="💡 Additive: Pay both lead + conversion credits separately. Net: Total credits can't exceed conversion cap.")
    expiry_months = st.number_input("Credit expiry (months)", min_value=1, value=12, step=1,
                                   help="⏰ How long credits last before they expire. Example: 12 months = use it or lose it after 1 year.")

    st.markdown("---")
    st.subheader("Milestone split for conversion credit")
    st.caption("🏗️ When to pay conversion credits during the project timeline")
    colA, colB, colC = st.columns(3)
    with colA:
        phi1 = st.number_input("Deposit φ₁", min_value=0.0, value=0.40, step=0.05,
                              help="💳 % of conversion credit paid when customer pays deposit")
    with colB:
        phi2 = st.number_input("Design sign-off φ₂", min_value=0.0, value=0.40, step=0.05,
                              help="✅ % of conversion credit paid when design is approved")
    with colC:
        phi3 = st.number_input("Final payment φ₃", min_value=0.0, value=0.20, step=0.05,
                              help="🎉 % of conversion credit paid when project is completed")
    phi1, phi2, phi3 = normalize_fracs([phi1, phi2, phi3])

    st.markdown("---")
    st.subheader("Upsell kicker (optional)")
    st.caption("🚀 Extra rewards for bringing repeat customers or additional projects")
    enable_kicker = st.checkbox("Enable upsell kicker", value=False,
                               help="✨ Turn on bonus credits for when referred customers buy again or upgrade")
    alpha_repeat = st.slider("Repeat share α_repeat (of margin)", 0.0, 1.0, 0.15, 0.01, disabled=not enable_kicker,
                            help="🔄 % of profit you'll share on repeat/upsell business. Usually lower than initial conversion rate.")

# -----------------------------
# Tabs
# -----------------------------
tab_calc, tab_margin = st.tabs(["📊 Calculator", "📈 Profit Margin Impact"])

with tab_calc:
    # -----------------------------
    # Lead/project inputs
    # -----------------------------
    st.header("Lead / Project Inputs")
    st.caption("📝 Details about this specific referral opportunity")
    c1, c2, c3 = st.columns(3)
    with c1:
        R_exp = st.number_input("Expected revenue for this lead (R̄)", min_value=0.0, value=100_000.0, step=5_000.0,
                               help="💵 How much money you expect this project to be worth if it converts")
        Q = st.slider("Lead quality Q (0–1)", 0.0, 1.0, 0.60, 0.01,
                     help="⭐ How good is this lead? 0=terrible, 1=perfect. Consider: budget confirmed, decision maker identified, urgent need, good fit")
    with c2:
        E = st.slider("Effort score E (0–1)", 0.0, 1.0, 0.50, 0.01, 
                     help="💪 How much work did the referrer put in? 0=just mentioned your name, 1=warm intro + shared budget + multiple touchpoints")
        n = st.number_input("Leads from same referrer in window n", min_value=0, value=1, step=1,
                           help="🔢 How many leads has this person given you recently? (Used to reduce credits for volume referrers)")
    with c3:
        S = st.number_input("Saturation S (diminishing returns)", min_value=1, value=5, step=1, 
                           help="🔄 How quickly to reduce credits for multiple leads. Higher number = slower reduction. Example: 5 means 5th lead gets decent credits, 2 means drops off quickly")
        R_actual = st.number_input("Actual project revenue R (if converted)", min_value=0.0, value=120_000.0, step=5_000.0,
                                  help="💰 The real project value (if they bought). Often different from initial estimate")

    if enable_kicker:
        R_upsell = st.number_input("Upsell revenue R_upsell", min_value=0.0, value=0.0, step=5_000.0,
                                  help="🎁 Additional revenue from repeat business, upgrades, or follow-on projects from this referral")

    # -----------------------------
    # Calculations
    # -----------------------------
    p = clamp(p0 + beta * Q, 0.0, 1.0)                  # Conversion probability from quality
    EV_lead = p * g * R_exp                              # Expected margin value of a lead
    M_E = 1.0 + delta * E                                # Effort multiplier
    D = diminishing_returns(n, S)                        # Diminishing returns factor

    CL_raw = eta * EV_lead * M_E
    CL_capped = min(CL_raw, CLmax)
    CL = CL_capped * D

    CC_raw = alpha * g * R_actual * M_E
    CC_capped = min(CC_raw, CCmax)

    if payout_policy.startswith("Net"):
        CC_effective_max = max(0.0, CCmax - CL)  # remaining headroom after lead credit
        CC_paid = min(CC_capped, CC_effective_max)
        C_total = CL + CC_paid
    else:
        CC_paid = CC_capped
        C_total = CL + CC_paid

    # Hours equivalents
    hours_CL = (CL / r) if r > 0 else 0.0
    hours_CC = (CC_paid / r) if r > 0 else 0.0
    hours_total = (C_total / r) if r > 0 else 0.0

    # Milestones
    phi = np.array([phi1, phi2, phi3], dtype=float)
    phi = phi / phi.sum() if phi.sum() > 0 else np.array([0.0, 0.0, 0.0])
    milestone_amounts = (CC_paid * phi).round(2)

    milestone_df = pd.DataFrame({
        "Milestone": ["Deposit (φ₁)", "Design sign-off (φ₂)", "Final payment (φ₃)"],
        "Share": phi.round(3),
        "Credit payout": milestone_amounts
    })

    # Upsell kicker
    kicker_amount = 0.0
    if enable_kicker and R_upsell > 0:
        kicker_amount = alpha_repeat * g * R_upsell * M_E
        # Apply cap logic? Often separate cap. We'll show as separate amount.
        kicker_hours = kicker_amount / r if r > 0 else 0.0
    else:
        kicker_hours = 0.0

    # -----------------------------
    # Display
    # -----------------------------
    st.subheader("Results")
    st.caption("🧮 Calculated values based on your inputs")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("p(Q)", f"{p:.2f}", help="Conversion probability for this lead quality")
    m2.metric("EV_lead (expected margin value)", f"{EV_lead:,.0f}", help="Expected profit from this lead")
    m3.metric("Effort multiplier M_E", f"{M_E:.2f}", help="Bonus multiplier for referrer effort")
    m4.metric("Diminishing factor D(n)", f"{D:.2f}", help="Reduction factor for multiple leads from same person")

    st.markdown("### Credits (Currency Units) and Hours")
    st.caption("💳 How much you'll pay the referrer and what it's worth in work time")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Lead credit C_L", f"{CL:,.0f}", help="💡 Credits paid just for bringing the lead (before they buy anything)")
        st.caption(f"≈ {hours_CL:.2f} hours @ r={r:,.0f}")
    with col2:
        st.metric("Conversion credit C_C (paid)", f"{CC_paid:,.0f}", help="🎯 Credits paid when the lead actually becomes a customer")
        st.caption(f"≈ {hours_CC:.2f} hours @ r={r:,.0f}")
    with col3:
        st.metric("Total credit (lead + conversion)", f"{C_total:,.0f}", help="💰 Total credits you'll pay for this successful referral")
        st.caption(f"≈ {hours_total:.2f} hours @ r={r:,.0f}")

    st.markdown("#### Conversion Credit Milestones")
    st.caption("📅 When credits get paid out during the project")
    st.dataframe(milestone_df, use_container_width=True)

    if enable_kicker and R_upsell > 0:
        st.markdown("#### Upsell Kicker (Separate)")
        st.info(f"🚀 Upsell credit (separate from caps): {kicker_amount:,.0f} (≈ {kicker_hours:.2f} hours)")

    # What-if analysis: vary Q from 0..1
    with st.expander("📊 What-if: credits vs. quality (Q)"):
        st.caption("See how lead quality affects credit amounts")
        QQ = np.linspace(0, 1, 21)
        pQ = np.clip(p0 + beta * QQ, 0, 1)
        EVQ = pQ * g * R_exp
        CLQ_raw = eta * EVQ * M_E
        CLQ_capped = np.minimum(CLQ_raw, CLmax) * D
        # Expected conversion credit (EV) if using expected revenue instead of realized R
        exp_CCQ_raw = alpha * g * (R_exp) * M_E * pQ  # Expected value view (not actual payout rule)
        exp_CCQ = np.minimum(exp_CCQ_raw, CCmax if payout_policy.startswith("Additive") else np.maximum(0.0, CCmax - CLQ_capped))
        df_whatif = pd.DataFrame({
            "Q": QQ,
            "Lead Credit CL (expected)": CLQ_capped,
            "Expected Conversion Credit (EV view)": exp_CCQ
        })
        st.line_chart(df_whatif.set_index("Q"))

    # Export scenario
    st.markdown("---")
    st.subheader("Export Scenario")
    st.caption("💾 Save this calculation setup and results for your records")
    scenario = {
        "globals": dict(r=r, g=g, alpha=alpha, eta=eta, delta=delta, p0=p0, beta=beta, CLmax=CLmax, CCmax=CCmax, payout_policy=payout_policy, expiry_months=expiry_months,
                        milestone_shares={"phi1": float(phi1), "phi2": float(phi2), "phi3": float(phi3)},
                        upsell=dict(enabled=enable_kicker, alpha_repeat=alpha_repeat if enable_kicker else 0.0)),
        "inputs": dict(R_exp=R_exp, Q=Q, E=E, n=n, S=S, R_actual=R_actual, R_upsell=(R_upsell if enable_kicker else 0.0)),
        "results": dict(p=p, EV_lead=EV_lead, M_E=M_E, D=D, CL=CL, CC_paid=CC_paid, C_total=C_total,
                        hours=dict(CL=hours_CL, CC=hours_CC, total=hours_total),
                        milestones=[{"name": "Deposit", "share": float(phi[0]), "credit": float(milestone_amounts[0])},
                                    {"name": "Design sign-off", "share": float(phi[1]), "credit": float(milestone_amounts[1])},
                                    {"name": "Final payment", "share": float(phi[2]), "credit": float(milestone_amounts[2])}],
                        upsell_credit=(kicker_amount if enable_kicker else 0.0))
    }
    scenario_json = json.dumps(scenario, indent=2)
    st.download_button("📁 Download scenario as JSON", data=scenario_json, file_name="referral_scenario.json", mime="application/json")

# -----------------------------
# Profit Margin Impact Tab
# -----------------------------
with tab_margin:
    st.header("📈 Profit Margin Impact Analysis")
    st.caption("🔍 Estimate how credits affect gross margin under different redemption/accounting modes.")
    
    # Need to ensure variables are available from first tab
    if 'C_total' not in locals():
        st.warning("⚠️ Please configure the calculator in the first tab before viewing margin impact.")
    else:
        colm1, colm2, colm3 = st.columns(3)
        with colm1:
            redemption_rate = st.slider("Redemption rate ρ", 0.0, 1.0, 0.80, 0.05, 
                                       help="💳 Fraction of issued credits that are actually redeemed (1 - breakage). Example: 80% means 20% of credits expire unused.")
            include_kicker_margin = st.checkbox("Include upsell kicker in credit pool", value=False, 
                                               help="✨ If enabled, adds upsell credits to redemption pool for margin calculations.")
        with colm2:
            expiry_months_m = st.number_input("Redemption window (months)", min_value=1, value=12, step=1, 
                                             help="📅 Assume redemptions spread evenly over this time period.")
            redemption_mode = st.selectbox("Redemption/accounting mode", 
                                         ["Discount (contra-revenue)", "Free hours (cash cost only)", "Free hours (capacity-aware)"],
                                         help="💡 How credits are redeemed: as discounts, free work hours, or capacity-constrained hours.")
        with colm3:
            c_var_hour = st.number_input("Variable delivery cost per hour (c_var)", min_value=0.0, 
                                        value=float(r*(1-g)), step=50.0,
                                        help="💰 Cash cost per delivery hour. Default uses r×(1-g).")
            H_cap = st.number_input("Monthly capacity H_cap (hours)", min_value=0.0, value=640.0, step=10.0, 
                                   help="⏰ Team capacity per month, used for capacity-aware mode.")
            base_paid_hours = st.number_input("Base paid hours/month (non-referral)", min_value=0.0, value=400.0, step=10.0,
                                            help="📊 Current monthly billable hours baseline (before referral program).")

        # Calculate margin impact
        total_credits_pool = C_total + (kicker_amount if (include_kicker_margin and enable_kicker and 'kicker_amount' in locals()) else 0.0)
        redeemed_credits = redemption_rate * total_credits_pool
        redeemed_hours_total = redeemed_credits / r if r > 0 else 0.0
        monthly_redeemed_hours = redeemed_hours_total / expiry_months_m

        # Project margin gained (from referred project)
        GM_gain_project = g * R_actual
        
        # Margin loss from credits by mode
        if redemption_mode.startswith("Discount"):
            # Each 1 unit of discount reduces margin by 1 unit (costs assumed unchanged)
            GM_loss_total = redeemed_credits
            monthly_margin_loss = np.repeat(GM_loss_total / expiry_months_m, expiry_months_m)
        elif redemption_mode.startswith("Free hours (cash cost"):
            # Cash cost only; no opportunity displacement modeled
            GM_loss_total = c_var_hour * redeemed_hours_total
            monthly_margin_loss = np.repeat(GM_loss_total / expiry_months_m, expiry_months_m)
        else:
            # Capacity-aware approximation
            slack = max(0.0, H_cap - base_paid_hours)
            # For each month, some portion of free hours consumes slack (cash cost), rest displaces paid hours (lost margin)
            cash_cost_per_month = c_var_hour * min(monthly_redeemed_hours, slack)
            displaced_hours = max(0.0, monthly_redeemed_hours - slack)
            opp_margin_loss_per_month = (g * r) * displaced_hours
            monthly_margin_loss_amount = cash_cost_per_month + opp_margin_loss_per_month
            monthly_margin_loss = np.repeat(monthly_margin_loss_amount, expiry_months_m)
            GM_loss_total = monthly_margin_loss.sum()

        net_GM = GM_gain_project - GM_loss_total
        net_margin_pct_on_project = (net_GM / R_actual) if R_actual > 0 else 0.0
        roi_on_credits = (net_GM / redeemed_credits) if redeemed_credits > 0 else float('nan')

        # Display key metrics
        st.subheader("💰 Margin Impact Summary")
        mA, mB, mC, mD = st.columns(4)
        mA.metric("Project GM gained", f"${GM_gain_project:,.0f}", 
                 help="Gross margin earned from the referred project")
        mB.metric("GM lost to credits", f"${GM_loss_total:,.0f}", 
                 help="Gross margin lost due to credit redemptions")
        mC.metric("Net GM impact", f"${net_GM:,.0f}", 
                 help="Net gross margin after accounting for credit costs",
                 delta=f"{net_GM:,.0f}")
        mD.metric("Net margin % on project", f"{100*net_margin_pct_on_project:.1f}%", 
                 help="Final margin percentage on this project after credit costs")

        # Additional insights
        st.subheader("📊 Credit Pool Breakdown")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Total credits issued", f"{total_credits_pool:,.0f}")
        col_b.metric("Credits redeemed", f"{redeemed_credits:,.0f}", f"({redemption_rate:.0%} rate)")
        col_c.metric("Hours equivalent", f"{redeemed_hours_total:.1f} hrs", f"({monthly_redeemed_hours:.1f}/month)")

        # Monthly breakdown chart
        if expiry_months_m > 1:
            st.subheader("📈 Monthly Margin Loss from Redemptions")
            df_month = pd.DataFrame({
                "Month": np.arange(1, expiry_months_m + 1), 
                "GM Loss ($)": monthly_margin_loss
            })
            st.line_chart(df_month.set_index("Month"))

        # Mode explanations
        st.subheader("🧮 Calculation Methods")
        if redemption_mode.startswith("Discount"):
            st.info("**Discount Mode**: Each redeemed credit = $1 reduction in revenue, assuming costs stay constant. Simple 1:1 margin impact.")
        elif redemption_mode.startswith("Free hours (cash cost"):
            st.info(f"**Free Hours (Cash Cost)**: Each redeemed hour costs ${c_var_hour:.0f} in variable delivery costs. No opportunity cost modeled.")
        else:
            slack_hours = max(0.0, H_cap - base_paid_hours)
            st.info(f"""**Capacity-Aware Mode**: 
            - Available slack: {slack_hours:.0f} hours/month
            - Slack hours cost: ${c_var_hour:.0f}/hour (cash cost only)
            - Hours above slack displace paid work: ${g*r:.0f}/hour opportunity cost
            - Monthly breakdown: {min(monthly_redeemed_hours, slack_hours):.1f} slack hours + {max(0, monthly_redeemed_hours - slack_hours):.1f} displaced hours""")

        # ROI insight
        if not np.isnan(roi_on_credits):
            if roi_on_credits > 1:
                st.success(f"✅ **Positive ROI**: Each redeemed credit generates ${roi_on_credits:.2f} in net margin")
            elif roi_on_credits > 0:
                st.warning(f"⚠️ **Marginal ROI**: Each redeemed credit generates ${roi_on_credits:.2f} in net margin")
            else:
                st.error(f"❌ **Negative ROI**: Each redeemed credit loses ${abs(roi_on_credits):.2f} in net margin")

        # Advanced insights
        with st.expander("🔬 Advanced Analysis"):
            st.markdown("### Sensitivity Analysis")
            st.caption("How margin impact changes with different redemption rates")
            
            redemption_scenarios = np.arange(0.5, 1.05, 0.1)
            scenario_results = []
            
            for scenario_rate in redemption_scenarios:
                scenario_redeemed = scenario_rate * total_credits_pool
                if redemption_mode.startswith("Discount"):
                    scenario_loss = scenario_redeemed
                elif redemption_mode.startswith("Free hours (cash cost"):
                    scenario_loss = c_var_hour * (scenario_redeemed / r)
                else:
                    scenario_hours_monthly = (scenario_redeemed / r) / expiry_months_m
                    scenario_slack = max(0.0, H_cap - base_paid_hours)
                    scenario_cash = c_var_hour * min(scenario_hours_monthly, scenario_slack) * expiry_months_m
                    scenario_displaced = max(0.0, scenario_hours_monthly - scenario_slack) * expiry_months_m
                    scenario_loss = scenario_cash + (g * r) * scenario_displaced
                
                scenario_net = GM_gain_project - scenario_loss
                scenario_results.append({
                    "Redemption Rate": f"{scenario_rate:.0%}",
                    "Net GM": scenario_net,
                    "Margin %": (scenario_net / R_actual * 100) if R_actual > 0 else 0
                })
            
            df_sensitivity = pd.DataFrame(scenario_results)
            st.dataframe(df_sensitivity, use_container_width=True)

st.caption("⚠️ **Note:** This is a planning tool. Ensure your published referral policy includes eligibility, attribution, clawback, and redemption rules.")

# Footer for public version
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em; margin-top: 2rem;'>
    <p>🛠️ Built with Streamlit • 💡 Open source calculator for referral program modeling</p>
    <p>📊 All calculations are performed locally in your browser - no data is collected or stored</p>
</div>
""", unsafe_allow_html=True)
