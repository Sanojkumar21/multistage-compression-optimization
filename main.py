import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

st.set_page_config(page_title="Module 1.2 - Gas Compression", layout="wide")

st.title("Module 1.2 – Multistage Gas Compression Optimization")

# -------------------------------------------------
# THEORY SECTION
# -------------------------------------------------
st.markdown("""
### Mechanical Energy Balance (Ideal Gas, Adiabatic Compression)

For each stage:

W = (γ / (γ - 1)) R T_in [ (p_out / p_in)^((γ-1)/γ) - 1 ]

For perfect intercooling:

p₂/p₁ = p₃/p₂ = p₄/p₃

(Equal pressure ratios minimize total work)
""")

# -------------------------------------------------
# INPUT PARAMETERS
# -------------------------------------------------
gamma = st.sidebar.slider("Heat Capacity Ratio (γ)", 1.1, 1.6, 1.4)
R = 287.0

T1 = st.sidebar.slider("Inlet Temperature (K)", 280, 350, 300)
T_cool_user = st.sidebar.slider("Cooling Temperature (K)", 250, 320, 290)

p1 = st.sidebar.number_input("Inlet Pressure (Pa)", value=100000.0)
p4 = st.sidebar.number_input("Outlet Pressure (Pa)", value=800000.0)

epsilon = st.sidebar.slider("Heat Exchanger Effectiveness (ε)", 0.0, 1.0, 1.0)

# For perfect intercooling enforce T_cool = T1
if epsilon == 1.0:
    T_cool = T1
else:
    T_cool = T_cool_user

exponent = (gamma - 1) / gamma

# -------------------------------------------------
# ANALYTICAL SOLUTION (Perfect Intercooling)
# -------------------------------------------------
st.subheader("Perfect Intercooling – Analytical Solution")

r_opt = (p4 / p1) ** (1 / 3)
p2_analytical = p1 * r_opt
p3_analytical = p2_analytical * r_opt

st.write(f"Analytical p₂ = {p2_analytical:,.0f} Pa")
st.write(f"Analytical p₃ = {p3_analytical:,.0f} Pa")

# -------------------------------------------------
# TOTAL WORK FUNCTION
# -------------------------------------------------
def total_work(vars, eps):
    p2, p3 = vars

    # Penalty constraint
    if not (p1 < p2 < p3 < p4):
        return 1e12

    # Stage 1
    r1 = p2 / p1
    T2 = T1 * r1**exponent
    W1 = (gamma / (gamma - 1)) * R * T1 * (r1**exponent - 1)

    T12 = T2 - eps * (T2 - T_cool)

    # Stage 2
    r2 = p3 / p2
    T3 = T12 * r2**exponent
    W2 = (gamma / (gamma - 1)) * R * T12 * (r2**exponent - 1)

    T23 = T3 - eps * (T3 - T_cool)

    # Stage 3
    r3 = p4 / p3
    W3 = (gamma / (gamma - 1)) * R * T23 * (r3**exponent - 1)

    return W1 + W2 + W3

# -------------------------------------------------
# NUMERICAL OPTIMIZATION
# -------------------------------------------------
st.subheader("Numerical Optimization")

bounds = [(p1 * 1.01, p4 * 0.99),
          (p1 * 1.02, p4 * 0.999)]

result = minimize(
    lambda x: total_work(x, epsilon),
    x0=[p2_analytical, p3_analytical],
    bounds=bounds,
    method="L-BFGS-B"
)

p2_opt, p3_opt = result.x
W_total_opt = result.fun

col1, col2, col3 = st.columns(3)

col1.metric("Optimal p₂ (Pa)", f"{p2_opt:,.0f}")
col2.metric("Optimal p₃ (Pa)", f"{p3_opt:,.0f}")
col3.metric("Minimum Work (J/kg)", f"{W_total_opt:,.0f}")

# -------------------------------------------------
# STAGE-WISE WORK BREAKDOWN
# -------------------------------------------------
st.subheader("Stage-wise Work Breakdown")

def compute_stage_work(p2, p3, eps):
    r1 = p2 / p1
    T2 = T1 * r1**exponent
    W1 = (gamma / (gamma - 1)) * R * T1 * (r1**exponent - 1)
    T12 = T2 - eps * (T2 - T_cool)

    r2 = p3 / p2
    T3 = T12 * r2**exponent
    W2 = (gamma / (gamma - 1)) * R * T12 * (r2**exponent - 1)
    T23 = T3 - eps * (T3 - T_cool)

    r3 = p4 / p3
    W3 = (gamma / (gamma - 1)) * R * T23 * (r3**exponent - 1)

    return W1, W2, W3

W1, W2, W3 = compute_stage_work(p2_opt, p3_opt, epsilon)

st.write(f"W₁ = {W1:,.0f} J/kg")
st.write(f"W₂ = {W2:,.0f} J/kg")
st.write(f"W₃ = {W3:,.0f} J/kg")

# -------------------------------------------------
# EFFECTIVENESS SENSITIVITY
# -------------------------------------------------
st.subheader("Effect of Heat Exchanger Effectiveness")

eps_vals = np.linspace(0.2, 1.0, 12)
work_vals = []

for eps in eps_vals:
    res = minimize(
        lambda x: total_work(x, eps),
        x0=[p2_analytical, p3_analytical],
        bounds=bounds,
        method="L-BFGS-B"
    )
    work_vals.append(res.fun)

fig, ax = plt.subplots(figsize=(6,4))
ax.plot(eps_vals, work_vals)
ax.set_xlabel("Effectiveness (ε)")
ax.set_ylabel("Minimum Work (J/kg)")
ax.grid(True)

st.pyplot(fig)

# -------------------------------------------------
# INTERPRETATION
# -------------------------------------------------
st.subheader("Interpretation")

st.markdown("""
• For ε = 1, equal pressure ratios minimize work.

• Stage-wise work becomes nearly equal.

• As ε decreases, intercooling becomes less effective.

• Gas enters next stage at higher temperature.

• Total compression work increases.

• Optimal intermediate pressures shift away from equal ratios.
""")

st.success("Module 1.2 Optimization Completed Successfully")
