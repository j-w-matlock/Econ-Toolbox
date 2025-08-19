import streamlit as st
import numpy as np
import pandas as pd
from io import BytesIO

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


# ---------------------------------------------------------------------------
# Data input section
# ---------------------------------------------------------------------------
st.subheader("Inputs")

# Initialize session state for dynamic table
if "num_damage_cols" not in st.session_state:
    st.session_state.num_damage_cols = 1

if "table" not in st.session_state:
    st.session_state.table = pd.DataFrame(
        {
            "Frequency": [0.50, 0.20, 0.10, 0.04, 0.02, 0.01, 0.005, 0.002],
            "Damage 1": [
                10000,
                40000,
                80000,
                120000,
                160000,
                200000,
                250000,
                300000,
            ],
        }
    )

# Optionally include stage column
include_stage = st.checkbox(
    "Include stage column", value="Stage" in st.session_state.table.columns
)
if include_stage and "Stage" not in st.session_state.table.columns:
    st.session_state.table.insert(1, "Stage", [None] * len(st.session_state.table))
elif not include_stage and "Stage" in st.session_state.table.columns:
    st.session_state.table.drop(columns="Stage", inplace=True)

# Allow user to add additional damage columns
if st.button("Add damage column"):
    st.session_state.num_damage_cols += 1
    st.session_state.table[f"Damage {st.session_state.num_damage_cols}"] = [
        None
    ] * len(st.session_state.table)

# Editable table with column configuration
column_config = {
    "Frequency": st.column_config.NumberColumn(
        "Frequency", min_value=0, max_value=1, step=0.001
    )
}
for col in st.session_state.table.columns:
    if col.startswith("Damage"):
        column_config[col] = st.column_config.NumberColumn(
            col, min_value=0, step=1000, format="$%d"
        )

data = st.data_editor(
    st.session_state.table,
    num_rows="dynamic",
    use_container_width=True,
    key="table_editor",
    column_config=column_config,
)
st.session_state.table = data

# Plot damage-frequency curve
damage_cols = [c for c in st.session_state.table.columns if c.startswith("Damage")]
chart_data = (
    st.session_state.table.dropna(subset=["Frequency"])
    .sort_values("Frequency")
    .set_index("Frequency")[damage_cols]
)
if not chart_data.empty:
    st.subheader("Damage-Frequency Curve")
    st.line_chart(chart_data)

# Export table to Excel
buffer = BytesIO()
st.session_state.table.to_excel(buffer, index=False)
buffer.seek(0)
st.download_button(
    label="Download table as Excel",
    data=buffer,
    file_name="ead_data.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)


# ---------------------------------------------------------------------------
# Calculation section
# ---------------------------------------------------------------------------
if st.button("Calculate EAD"):
    df = st.session_state.table.dropna(subset=["Frequency"]).sort_values(
        "Frequency", ascending=False
    )
    freq = df["Frequency"].to_numpy()
    results = {}
    for col in df.columns:
        if col.startswith("Damage"):
            damages = df[col].fillna(0).to_numpy()
            if len(freq) >= 2 and len(freq) == len(damages):
                results[col] = ead_trapezoidal(freq, damages)
            else:
                results[col] = None

    if not results:
        st.error("No damage columns found.")
    else:
        for col, val in results.items():
            if val is None:
                st.error(
                    f"{col}: Ensure at least two paired frequency and damage values."
                )
            else:
                st.success(f"{col} Expected Annual Damage: ${val:,.2f}")

