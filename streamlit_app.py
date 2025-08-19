import streamlit as st
import numpy as np

st.title("Expected Annual Damage (EAD) Calculator")
st.write(
    "Compute expected annual damages using the U.S. Army Corps of Engineers trapezoidal method."
)


def ead_trapezoidal(prob, damages):
    """Return expected annual damage via trapezoidal integration."""
    prob = np.asarray(prob, dtype=float)
    damages = np.asarray(damages, dtype=float)
    return float(
        np.sum(0.5 * (damages[:-1] + damages[1:]) * (prob[:-1] - prob[1:]))
    )


st.subheader("Inputs")
prob_input = st.text_area(
    "Annual exceedance probabilities (descending, comma-separated)",
    value="0.50, 0.20, 0.10, 0.04, 0.01",
)
damage_input = st.text_area(
    "Damages at each probability (same length, comma-separated)",
    value="10000, 40000, 80000, 120000, 250000",
)

if st.button("Calculate EAD"):
    try:
        prob = [float(p.strip()) for p in prob_input.split(",") if p.strip() != ""]
        damages = [float(d.strip()) for d in damage_input.split(",") if d.strip() != ""]
        if len(prob) < 2 or len(prob) != len(damages):
            st.error(
                "Please supply the same number of probabilities and damages (at least two pairs)."
            )
        else:
            ead_value = ead_trapezoidal(prob, damages)
            st.success(f"Expected Annual Damage: ${ead_value:,.2f}")
    except ValueError:
        st.error("Inputs must be numeric and comma separated.")

