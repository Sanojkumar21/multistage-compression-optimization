import streamlit as st
import numpy as np
from scipy.optimize import minimize

st.set_page_config(page_title="Multistage Compression Optimization", layout="wide")

st.title("Module 1.2 - Multistage Gas Compression Optimization")

# Sidebar Inputs
gamma = st.sidebar.slider("Heat Capacity Ratio (γ)", 1.1, 1.6, 1.4)
R = 287
T1 = st.sidebar.slider("Inlet Temperature (K)", 280, 350, 300)
T_cool = st.sidebar.slider("Cooling Temperature (K)", 280, 320, 290)

p1 = st.sidebar.number_input("Inlet Pressure (Pa)", value=1e5)
p4 = st.sidebar.number_input("Outlet Pressure (Pa)", value=8e5)

epsilon = st.sidebar.slider("Heat Exchanger Effectiveness", 0.0, 1.0, 0.8)

exponent = (gamma-1)/gamma

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
                  x0=[2e5,4e5],
                  bounds=bounds,
                  method="L-BFGS-B")

p2_opt, p3_opt = result.x

st.metric("Optimal p2 (Pa)", f"{p2_opt:,.0f}")
st.metric("Optimal p3 (Pa)", f"{p3_opt:,.0f}")
st.metric("Minimum Work (J/kg)", f"{result.fun:,.0f}")

st.success("Optimization Completed Successfully")
