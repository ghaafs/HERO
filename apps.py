# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import datetime

# ==============
# CONFIG
# ==============
st.set_page_config(page_title="HE Model: BIA + Markov CEA (Portfolio)", layout="wide")

# Replace this with your real name before publishing
AUTHOR_NAME = "Ghany Fitriamara S"
COPYRIGHT_YEAR = 2025

# ==============
# CSS / Watermark
# ==============
watermark_text = f"Created by {AUTHOR_NAME} — Health Economics Modelling Portfolio (BIA + CEA) ©{COPYRIGHT_YEAR}"


# ==============
# Helper functions
# ==============
def run_bia(pop_adult, prevalence, years, screen_cost, treat_early, treat_comp, coverage_start, coverage_ramp, reduction_comp_prob):
    years = int(years)
    bia = pd.DataFrame({'year': np.arange(1, years+1)})
    bia['coverage'] = np.clip(coverage_start + (bia['year']-1)*coverage_ramp, 0, 1)
    bia['people_screened'] = (pop_adult * bia['coverage']).astype(int)
    bia['screen_cost_total'] = bia['people_screened'] * screen_cost
    bia['cases_detected'] = (bia['people_screened'] * prevalence).astype(int)
    bia['treatment_early_cost_total'] = bia['cases_detected'] * treat_early
    bia['averted_comp_cost_total'] = bia['cases_detected'] * reduction_comp_prob * (treat_comp - treat_early)
    bia['net_budget_impact'] = bia['screen_cost_total'] + bia['treatment_early_cost_total'] - bia['averted_comp_cost_total']
    bia['cumulative_net_impact'] = bia['net_budget_impact'].cumsum()
    return bia

def run_markov(P, cohort, years, costs_state, utilities_state, discount_rate):
    years = int(years)
    n_states = len(costs_state)
    pop = np.zeros((years+1, n_states))
    pop[0,0] = cohort
    yearly_costs = np.zeros(years)
    yearly_qalys = np.zeros(years)
    for t in range(years):
        pop[t+1,:] = pop[t,:].dot(P)
        occupancy = (pop[t] + pop[t+1]) / 2.0  # half-cycle approximation
        cost = sum(occupancy[i] * costs_state[i] for i in range(n_states))
        qaly = sum(occupancy[i] * utilities_state[i] for i in range(n_states))
        df = 1 / ((1+discount_rate)**(t+1))
        yearly_costs[t] = cost * df
        yearly_qalys[t] = qaly * df
    total_cost = yearly_costs.sum()
    total_qalys = yearly_qalys.sum()
    return {'total_cost': total_cost, 'total_qalys': total_qalys, 'yearly_costs': yearly_costs, 'yearly_qalys': yearly_qalys}

def to_csv_bytes(df):
    return df.to_csv(index=False).encode('utf-8')

# ==============
# SIDEBAR - Project info + Inputs + Watermark text (human readable)
# ==============
st.sidebar.title("Project — Diabetes Screening (Portfolio)")
st.sidebar.write(
    """
**Project purpose:** Demo portfolio to showcase Budget Impact Analysis (BIA) and Cost-Effectiveness Analysis (CEA)
for a simulated national diabetes screening program in Indonesia.

**Note:** This is a *demonstration model* using illustrative assumptions. Replace parameters with local data for policy work.
"""
)

st.sidebar.markdown("---")
st.sidebar.header("Your ownership / copyright")
st.sidebar.write(f"Creator: **{AUTHOR_NAME}**  \nCopyright © {COPYRIGHT_YEAR} — All rights reserved.")
st.sidebar.markdown("**Disclaimer:** This model uses simulated/illustrative inputs. Not for clinical use.")

st.sidebar.markdown("---")
st.sidebar.header("Input - General")
pop_adult = st.sidebar.number_input("Adult population (20-79)", value=185_217_400, step=1000)
prevalence = st.sidebar.number_input("Diabetes prevalence (proportion)", value=0.113, format="%.3f", step=0.001)
years_bia = st.sidebar.number_input("BIA horizon (years)", value=5, min_value=1, max_value=20)
years_cea = st.sidebar.number_input("CEA horizon (years)", value=20, min_value=5, max_value=100)
cohort = st.sidebar.number_input("CEA cohort size (persons)", value=1000, step=100)
discount_rate = st.sidebar.number_input("Discount rate (decimal)", value=0.03, format="%.3f", step=0.01)

st.sidebar.markdown("---")
st.sidebar.header("Costs (IDR)")
screen_cost_per_person = st.sidebar.number_input("Screening cost per person", value=50000, step=1000)
treat_early_cost_per_patient = st.sidebar.number_input("Annual early-treatment cost per patient", value=2_500_000, step=10000)
treat_comp_cost_per_patient = st.sidebar.number_input("Annual complication cost per patient", value=8_000_000, step=10000)

st.sidebar.markdown("---")
st.sidebar.header("BIA coverage & effect")
coverage_start = st.sidebar.number_input("Coverage year1 (proportion)", value=0.20, format="%.2f", step=0.05)
coverage_ramp = st.sidebar.number_input("Coverage ramp per year", value=0.05, format="%.2f", step=0.01)
reduction_comp_prob = st.sidebar.number_input("Reduction comp. prob. due to early detection (prop)", value=0.30, format="%.2f", step=0.05)

st.sidebar.markdown("---")
st.sidebar.header("Markov: states & utilities (QALY weights)")
st.sidebar.write("Default states order: [Healthy/AtRisk, Diagnosed_Treated, Complication, Dead]")
u_healthy = st.sidebar.number_input("Healthy/AtRisk utility", value=0.85, format="%.2f", step=0.01)
u_diag = st.sidebar.number_input("Diagnosed & treated utility", value=0.75, format="%.2f", step=0.01)
u_comp = st.sidebar.number_input("Complication utility", value=0.50, format="%.2f", step=0.01)
u_dead = 0.0

st.sidebar.markdown("---")
st.sidebar.caption("Tip: Replace AUTHOR_NAME at top of file with your real name before sharing the app link publicly.")

# ==============
# Main area - Run models
# ==============
st.title("Health Economic Portfolio — BIA + Markov CEA (Demo)")
st.write("Interactive demo for a portfolio: change assumptions in the sidebar, run models, and download outputs. Watermark & copyright identify ownership.")

if st.button("▶️ Run analysis"):
    # Build and run BIA
    bia_df = run_bia(
        pop_adult=int(pop_adult),
        prevalence=float(prevalence),
        years=int(years_bia),
        screen_cost=int(screen_cost_per_person),
        treat_early=int(treat_early_cost_per_patient),
        treat_comp=int(treat_comp_cost_per_patient),
        coverage_start=float(coverage_start),
        coverage_ramp=float(coverage_ramp),
        reduction_comp_prob=float(reduction_comp_prob)
    )

    st.subheader("Budget Impact Analysis")
    st.dataframe(bia_df.style.format({
        'coverage': '{:.2%}',
        'people_screened': '{:,.0f}',
        'screen_cost_total': '{:,.0f}',
        'cases_detected': '{:,.0f}',
        'treatment_early_cost_total': '{:,.0f}',
        'averted_comp_cost_total': '{:,.0f}',
        'net_budget_impact': '{:,.0f}',
        'cumulative_net_impact': '{:,.0f}'
    }))

    # Plot net budget impact
    fig1, ax1 = plt.subplots(figsize=(8,3.5))
    ax1.plot(bia_df['year'], bia_df['net_budget_impact']/1e12, marker='o')
    ax1.set_title('Net Budget Impact per Year (Trillion IDR)')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Net Budget Impact (trillion IDR)')
    ax1.grid(True)
    st.pyplot(fig1)

    # Prepare Markov inputs
    costs_state = [
        0,
        int(treat_early_cost_per_patient),
        int(treat_comp_cost_per_patient),
        0
    ]
    utilities_state = [
        float(u_healthy),
        float(u_diag),
        float(u_comp),
        float(u_dead)
    ]

    # Default transition matrices (base vs screen)
    P_base = np.array([
        [0.90, 0.05, 0.04, 0.01],
        [0.00, 0.80, 0.15, 0.05],
        [0.00, 0.00, 0.85, 0.15],
        [0.00, 0.00, 0.00, 1.00]
    ])
    P_screen = np.array([
        [0.92, 0.07, 0.00, 0.01],
        [0.00, 0.88, 0.07, 0.05],
        [0.00, 0.00, 0.86, 0.14],
        [0.00, 0.00, 0.00, 1.00]
    ])

    # Run Markov model for both scenarios
    res_base = run_markov(P_base, int(cohort), int(years_cea), costs_state, utilities_state, float(discount_rate))
    res_screen = run_markov(P_screen, int(cohort), int(years_cea), costs_state, utilities_state, float(discount_rate))

    delta_cost = res_screen['total_cost'] - res_base['total_cost']
    delta_qaly = res_screen['total_qalys'] - res_base['total_qalys']
    icer = delta_cost / delta_qaly if delta_qaly != 0 else np.nan

    st.subheader(f"Markov CEA (per cohort = {int(cohort):,})")
    cea_summary = pd.DataFrame({
        'scenario': ['No Screening (base)', 'Screening (intervention)'],
        'total_cost_Rp': [res_base['total_cost'], res_screen['total_cost']],
        'total_QALYs': [res_base['total_qalys'], res_screen['total_qalys']]
    })
    st.dataframe(cea_summary.style.format({
        'total_cost_Rp': '{:,.0f}',
        'total_QALYs': '{:.3f}'
    }))

    st.write("Delta Cost (Rp): {:,.0f}".format(delta_cost))
    st.write("Delta QALY: {:.3f}".format(delta_qaly))
    st.write("ICER (Rp per QALY): {:,.0f}".format(icer) if not np.isnan(icer) else "ICER: NaN")

    # Plot cumulative discounted costs
    years_arr = np.arange(1, int(years_cea)+1)
    cum_cost_base = np.cumsum(res_base['yearly_costs'])/1e9
    cum_cost_screen = np.cumsum(res_screen['yearly_costs'])/1e9
    fig2, ax2 = plt.subplots(figsize=(8,3.5))
    ax2.plot(years_arr, cum_cost_base, label='No Screening')
    ax2.plot(years_arr, cum_cost_screen, label='Screening')
    ax2.set_title('Cumulative Discounted Costs (Billion IDR)')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Cumulative Cost (billion IDR)')
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

    # Download buttons for CSVs
    st.markdown("### Download outputs")
    st.download_button("Download BIA results (CSV)", to_csv_bytes(bia_df), file_name="bia_results.csv", mime="text/csv")
    st.download_button("Download CEA summary (CSV)", to_csv_bytes(cea_summary), file_name="cea_results.csv", mime="text/csv")

    # ---------------------
    # One-way sensitivity example (vary complication cost)
    # ---------------------
    st.subheader("One-way sensitivity: complication cost")
    cc_vals = np.array([4_000_000, 6_000_000, 8_000_000, 10_000_000, 12_000_000])
    sens_res = []
    for val in cc_vals:
        costs_state_var = [0, int(treat_early_cost_per_patient), int(val), 0]
        r_screen = run_markov(P_screen, int(cohort), int(years_cea), costs_state_var, utilities_state, float(discount_rate))
        r_base = run_markov(P_base, int(cohort), int(years_cea), costs_state_var, utilities_state, float(discount_rate))
        delta_cost_v = r_screen['total_cost'] - r_base['total_cost']
        delta_qaly_v = r_screen['total_qalys'] - r_base['total_qalys']
        icer_v = delta_cost_v / delta_qaly_v if delta_qaly_v != 0 else np.nan
        sens_res.append({'comp_cost': val, 'icer': icer_v, 'delta_cost': delta_cost_v, 'delta_qaly': delta_qaly_v})
    sens_df = pd.DataFrame(sens_res)
    st.dataframe(sens_df.style.format({'comp_cost': '{:,.0f}', 'icer': '{:,.0f}', 'delta_cost':'{:,.0f}', 'delta_qaly':'{:.3f}'}))

    fig3, ax3 = plt.subplots(figsize=(6,3))
    ax3.plot(sens_df['comp_cost']/1e6, sens_df['icer']/1e6, marker='o')
    ax3.set_title('One-way: comp cost vs ICER (Rp million per QALY)')
    ax3.set_xlabel('Complication cost (million Rp)')
    ax3.set_ylabel('ICER (million Rp per QALY)')
    ax3.grid(True)
    st.pyplot(fig3)

    st.success("Analysis completed. Download outputs or tweak assumptions on the left and re-run.")

else:
    st.info("Configure parameters on the left and click ▶️ Run analysis to compute BIA + CEA.")

# ==============
# Footer (copyright & small disclaimer)
# ==============
st.markdown(
    f"""
    <div class="footer">
    {watermark_text}  •  This is a demonstration model for portfolio purposes. Replace assumptions with validated data before using for policy or clinical decisions.
    </div>
    """,
    unsafe_allow_html=True,
)
