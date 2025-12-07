import numpy as np
import pandas as pd
import streamlit as st

# -----------------------
# PAGE CONFIG
# -----------------------
st.set_page_config(
    page_title="Vero Cell Culture Growth – Vero on Cytodex 1",
    layout="wide"
)

st.title("Vero Cell Culture Growth Model")
st.caption("Conceptual model for Vero cells on Cytodex 1 microcarriers – for training & illustration, not for GMP use.")

st.markdown(
    """
This model simulates batch growth of Vero cells on Cytodex 1 microcarriers using a **logistic growth** equation.  
The effective growth rate μ and carrying capacity Xmax are estimated from **process conditions**:

- pH (5–9)  
- Cytodex 1 concentration (1–10 g/L)  
- Glucose concentration (0–6 g/L)  
- Glucosamine concentration (0–20 mM)  
- Temperature (30–39 °C)
"""
)

# -----------------------
# SIDEBAR – INPUTS
# -----------------------
st.sidebar.header("Process conditions")

# Time settings
st.sidebar.subheader("Time settings")
t_end = st.sidebar.slider(
    "Simulation time (hours)",
    min_value=48,
    max_value=336,
    value=168,
    step=24
)
n_points = st.sidebar.slider(
    "Number of time points",
    min_value=100,
    max_value=500,
    value=300,
    step=50
)

# Initial cell density
st.sidebar.subheader("Initial conditions")
X0 = st.sidebar.number_input(
    "Initial viable cell density X₀ (cells/mL)",
    min_value=1e4,
    max_value=1e7,
    value=3e5,
    step=1e5,
    format="%.3e"
)

# Process variables – Vero on Cytodex 1
st.sidebar.subheader("Culture parameters")

pH = st.sidebar.slider("pH", min_value=5.0, max_value=9.0, value=7.2, step=0.1)
temp = st.sidebar.slider("Temperature (°C)", min_value=30.0, max_value=39.0, value=37.0, step=0.1)
cytodex = st.sidebar.slider("Cytodex 1 (g/L)", min_value=1.0, max_value=10.0, value=3.0, step=0.5)
glucose = st.sidebar.slider("Glucose (g/L)", min_value=0.0, max_value=6.0, value=3.0, step=0.1)
glucosamine = st.sidebar.slider("Glucosamine (mM)", min_value=0.0, max_value=20.0, value=4.0, step=0.5)

# -----------------------
# SUPPORT FUNCTIONS
# -----------------------

def gaussian_factor(x, x_opt, width):
    """
    Symmetric bell-shaped factor between 0 and 1.
    width ~ half-width of 'good' region.
    """
    return np.exp(-0.5 * ((x - x_opt) / width) ** 2)

def monod_factor(S, Ks):
    """Simple Monod-like saturation between 0 and 1."""
    return S / (Ks + S) if S > 0 else 0.0

def bounded_saturation(x, x_opt, Ks_low, Ks_high):
    """
    Factor that rises with x (like Monod) and gently penalizes strong overdose.
    x_opt is where it’s roughly near 1.
    """
    # Low-side saturation
    low = monod_factor(x, Ks_low)
    # High-side penalty – gets <1 when x >> x_opt
    penalty = 1.0 / (1.0 + ((max(x - x_opt, 0) / Ks_high) ** 2))
    return low * penalty

def compute_mu_and_xmax(pH, temp, cytodex, glucose, glucosamine, X0):
    """
    Very simplified empirical-like model for Vero cells
    on Cytodex 1 in batch mode.
    """
    # Baseline parameters (tunable)
    mu_max = 0.04    # 1/h – rough typical for Vero log phase
    Xmax_base = 2e6  # cells/mL – base carrying capacity

    # Individual factors (0–1)
    # pH – Vero typically likes ~7.2–7.4
    f_pH_mu = gaussian_factor(pH, x_opt=7.2, width=0.6)
    f_pH_X = gaussian_factor(pH, x_opt=7.2, width=0.7)

    # Temperature – optimum around 37 °C, some tolerance
    f_T_mu = gaussian_factor(temp, x_opt=37.0, width=1.5)
    f_T_X = gaussian_factor(temp, x_opt=37.0, width=2.0)

    # Cytodex 1 – trade-off between surface area and mass transfer
    # Let’s say optimum ≈ 3 g/L, decent between 2–5 g/L
    f_cyto_mu = gaussian_factor(cytodex, x_opt=3.0, width=1.5)
    f_cyto_X = gaussian_factor(cytodex, x_opt=3.5, width=2.0)

    # Glucose – Monod-like
    # Assume Ks ≈ 0.5 g/L, saturates towards 1 above ~3 g/L
    f_glu_mu = monod_factor(glucose, Ks=0.5)
    f_glu_X = monod_factor(glucose, Ks=0.8)

    # Glucosamine – support factor with mild overdose penalty
    # Rough "nice" zone around 2–6 mM
    f_glcn_mu = bounded_saturation(glucosamine, x_opt=4.0, Ks_low=1.0, Ks_high=4.0)
    f_glcn_X = bounded_saturation(glucosamine, x_opt=4.0, Ks_low=1.5, Ks_high=5.0)

    # Combine factors multiplicatively (very simplified)
    mu_eff = mu_max * f_pH_mu * f_T_mu * f_cyto_mu * f_glu_mu * f_glcn_mu

    # Prevent unrealistic zero μ
    mu_eff = max(mu_eff, 1e-4)

    Xmax_eff = Xmax_base * f_pH_X * f_T_X * f_cyto_X * f_glu_X * f_glcn_X

    # Ensure Xmax at least a bit above X0
    Xmax_eff = max(Xmax_eff, X0 * 1.5)

    # Also return the individual factors for display
    factors = {
        "pH (μ)": f_pH_mu,
        "Temp (μ)": f_T_mu,
        "Cytodex (μ)": f_cyto_mu,
        "Glucose (μ)": f_glu_mu,
        "Glucosamine (μ)": f_glcn_mu,
        "pH (Xmax)": f_pH_X,
        "Temp (Xmax)": f_T_X,
        "Cytodex (Xmax)": f_cyto_X,
        "Glucose (Xmax)": f_glu_X,
        "Glucosamine (Xmax)": f_glcn_X,
    }

    return mu_eff, Xmax_eff, factors

def logistic_growth(t, X0, mu, Xmax):
    A = (Xmax - X0) / X0
    return Xmax / (1.0 + A * np.exp(-mu * t))

# -----------------------
# CALCULATIONS
# -----------------------
t = np.linspace(0, t_end, n_points)

mu_eff, Xmax_eff, factors = compute_mu_and_xmax(
    pH=pH,
    temp=temp,
    cytodex=cytodex,
    glucose=glucose,
    glucosamine=glucosamine,
    X0=X0,
)

X = logistic_growth(t, X0, mu_eff, Xmax_eff)

df = pd.DataFrame({"Time [h]": t, "Viable cell density [cells/mL]": X})

doubling_time = np.log(2) / mu_eff

# -----------------------
# LAYOUT
# -----------------------
col_plot, col_stats = st.columns([2.2, 1.3])

with col_plot:
    st.subheader("Growth curve – Vero on Cytodex 1")
    st.line_chart(df, x="Time [h]", y="Viable cell density [cells/mL]")

with col_stats:
    st.subheader("Derived culture performance")

    st.metric("Effective μ [1/h]", f"{mu_eff:.4f}")
    st.metric("Doubling time [h]", f"{doubling_time:.1f}")
    st.metric("Carrying capacity Xmax [cells/mL]", f"{Xmax_eff:.3e}")
    st.metric("Initial density X₀ [cells/mL]", f"{X0:.3e}")

    st.markdown("**Factor contributions to μ (0–1):**")
    df_mu = pd.DataFrame(
        {
            "Factor": ["pH", "Temperature", "Cytodex 1", "Glucose", "Glucosamine"],
            "Contribution": [
                factors["pH (μ)"],
                factors["Temp (μ)"],
                factors["Cytodex (μ)"],
                factors["Glucose (μ)"],
                factors["Glucosamine (μ)"],
            ],
        }
    ).set_index("Factor")
    st.bar_chart(df_mu)

st.markdown("---")
st.markdown(
    """
### Model notes

- **Logistic growth** is used to represent limited surface area, nutrients and waste accumulation.
- The effective growth rate **μ** and carrying capacity **Xmax** are **heuristic functions** of pH, temperature, Cytodex 1 load,
  glucose and glucosamine.  
- Each variable scales μ and Xmax with a factor between 0 and 1 (Gaussian/Monod-like).
- All shapes, optima and constants are **illustrative** and should be tuned against real experimental data if you want more realism.

Use it as an educational tool on your website: show how changing pH, feed or microcarrier load can conceptually
shift growth rate and final cell density for Vero cell cultures.
"""
)