import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

st.set_page_config(page_title="Module 1.2 - Gas Compression", layout="wide")

st.title("Module 1.2 – Multistage Gas Compression Optimization")

st.markdown("""
### Mechanical Energy Balance (Ideal Gas, Adiabatic Compression)

For each stage:

W = (γ/(γ-1)) R T_in [ (p_out/p_in)^((γ-1)/γ) - 1 ]

For perfect intercooling:

Optimal condition:
p2/p1 = p3/p2 = p4/p3
""")

# -------------------------------------------------
# INPUTS
# -------------------------------------------------
gamma = st.sidebar.slider("Heat Capacity Ratio (γ)", 1.1, 1.6, 1.4)
R = 287
T1 = st.sidebar.slider("Inlet Temperature (K)", 280, 350, 300)
T_cool = st.sidebar.slider("Cooling Temperature (K)", 280, 320, 290)

p1 = st.sidebar.number_input("Inlet Pressure (Pa)", value=1e5)
p4 = st.sidebar.number_input("Outlet Pressure (Pa)", value=8e5)

epsilon = st.sidebar.slider("Heat Exchanger Effectiveness", 0.0, 1.0, 1.0)

exponent = (gamma-1)/gamma

# -------------------------------------------------
# PERFECT INTERCOOLING (Analytical)
# -------------------------------------------------
st.subheader("Perfect Intercooling Case")

r_opt = (p4/p1)**(1/3)

p2_analytical = p1*r_opt
p3_analytical = p2_analytical*r_opt

st.write("Analytical Solution (Equal Pressure Ratios)")
st.write(f"p2 = {p2_analytical:,.0f} Pa")
st.write(f"p3 = {p3_analytical:,.0f} Pa")

# -------------------------------------------------
# NUMERICAL OPTIMIZATION
# -------------------------------------------------
def total_work(vars):
    p2, p3 = vars
    
    if not (p1 < p2 < p3 < p4):
        return 1e12
    
    # Stage 1
    r1 = p2/p1
    T2 = T1*r1**exponent
    W1 = (gamma/(gamma-1))*R*T1*(r1**exponent - 1)
    
    T12 = T2 - epsilon*(T2 - T_cool)
    
    # Stage 2
    r2 = p3/p2
    T3 = T12*r2**exponent
    W2 = (gamma/(gamma-1))*R*T12*(r2**exponent - 1)
    
    T23 = T3 - epsilon*(T3 - T_cool)
    
    # Stage 3
    r3 = p4/p3
    W3 = (gamma/(gamma-1))*R*T23*(r3**exponent - 1)
    
    return W1 + W2 + W3

bounds = [(p1*1.01,p4*0.99),(p1*1.02,p4*0.999)]

result = minimize(total_work,
                  x0=[p2_analytical,p3_analytical],
                  bounds=bounds,
                  method="L-BFGS-B")

p2_opt, p3_opt = result.x

st.subheader("Numerical Optimization Result")

st.metric("Optimal p2 (Pa)", f"{p2_opt:,.0f}")
st.metric("Optimal p3 (Pa)", f"{p3_opt:,.0f}")
st.metric("Minimum Work (J/kg)", f"{result.fun:,.0f}")

# -------------------------------------------------
# WORK vs EFFECTIVENESS
# -------------------------------------------------
st.subheader("Effect of Heat Exchanger Effectiveness")

eps_vals = np.linspace(0.2,1.0,10)
work_vals = []

for eps in eps_vals:
    res = minimize(lambda x: total_work(x),
                   x0=[p2_analytical,p3_analytical],
                   bounds=bounds,
                   method="L-BFGS-B")
    work_vals.append(res.fun)

fig, ax = plt.subplots(figsize=(5,4))
ax.plot(eps_vals, work_vals)
ax.set_xlabel("Effectiveness (ε)")
ax.set_ylabel("Minimum Work (J/kg)")
st.pyplot(fig)

# -------------------------------------------------
# INTERPRETATION
# -------------------------------------------------
st.subheader("Interpretation")

st.markdown("""
- For ε = 1, equal pressure ratios minimize work.
- As ε decreases, intercooling becomes imperfect.
- Total compression work increases.
- Optimal intermediate pressures shift away from equal ratios.
""")

st.success("Complete Module 1.2 Optimization Dashboard")
