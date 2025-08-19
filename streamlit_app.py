import streamlit as st
import numpy as np
import pandas as pd
from io import BytesIO

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a tool", ["Expected Annual Damage", "Cost of Storage"]
)


def ead_trapezoidal(prob, damages):
    """Return expected annual damage via trapezoidal integration."""
    prob = np.asarray(prob, dtype=float)
    damages = np.asarray(damages, dtype=float)
    return float(
        np.sum(0.5 * (damages[:-1] + damages[1:]) * (prob[:-1] - prob[1:]))
    )


def ead_page():
    st.title("Expected Annual Damage (EAD) Calculator")
    st.write(
        "Compute expected annual damages using the U.S. Army Corps of Engineers trapezoidal method."
    )

    # -----------------------------------------------------------------------
    # Data input section
    # -----------------------------------------------------------------------
    st.subheader("Inputs")

    # Initialize session state for dynamic table
    if "num_damage_cols" not in st.session_state:
        st.session_state.num_damage_cols = 1

    if "table" not in st.session_state:
        st.session_state.table = pd.DataFrame(
            {
                "Frequency": [1.00, 0.50, 0.20, 0.10, 0.04, 0.02, 0.01, 0.005, 0.002],
                "Damage 1": [
                    0,
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

    with st.form("data_table_form"):
        data = st.data_editor(
            st.session_state.table,
            num_rows="dynamic",
            use_container_width=True,
            key="table_editor",
            column_config=column_config,
        )
        submitted = st.form_submit_button("Save table")
    if submitted:
        st.session_state.table = data

    # Plot damage-frequency curve
    damage_cols = [c for c in st.session_state.table.columns if c.startswith("Damage")]
    chart_data = (
        st.session_state.table.dropna(subset=["Frequency"])
        .sort_values("Frequency")
        .set_index("Frequency")[damage_cols]
    )
    if not chart_data.empty and damage_cols:
        st.subheader("Damage-Frequency Curve")
        selected_damage = st.selectbox(
            "Select damage column", damage_cols, key="df_damage"
        )
        st.line_chart(chart_data[[selected_damage]])

    # Plot stage-related curves
    if "Stage" in st.session_state.table.columns:
        stage_df = (
            st.session_state.table.dropna(subset=["Stage"])
            .sort_values("Stage")
            .set_index("Stage")
        )
        if not stage_df.empty:
            if damage_cols:
                st.subheader("Stage-Damage Curve")
                dmg_col = st.selectbox(
                    "Select damage column (stage)", damage_cols, key="stage_damage"
                )
                st.line_chart(stage_df[[dmg_col]])
            if "Frequency" in stage_df.columns:
                st.subheader("Stage-Frequency Curve")
                st.line_chart(stage_df["Frequency"])

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

    # -----------------------------------------------------------------------
    # Calculation section
    # -----------------------------------------------------------------------
    if st.button("Calculate EAD"):
        df = st.session_state.table.dropna(subset=["Frequency"]).sort_values(
            "Frequency", ascending=False
        )
        freq = df["Frequency"].to_numpy()

        # Validate that frequencies begin at 1 and decrease towards 0
        if not np.isclose(freq[0], 1.0) or np.any(np.diff(freq) > 0):
            st.warning("Frequencies should start at 1 and monotonically decrease to 0.")

        missing_zero = freq[-1] != 0
        if missing_zero:
            st.info(
                "Final frequency not 0; appending zero-frequency point using last damage value."
            )

        results = {}
        for col in df.columns:
            if col.startswith("Damage"):
                damages = df[col].fillna(0).to_numpy()
                freq_use = np.append(freq, 0.0) if missing_zero else freq
                damages_use = np.append(damages, damages[-1]) if missing_zero else damages
                if len(freq_use) >= 2 and len(freq_use) == len(damages_use):
                    results[col] = ead_trapezoidal(freq_use, damages_use)
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


def cost_of_storage_page():
    st.title("Cost of Storage Calculator")
    st.write(
        "Update project costs to current price levels using USACE Civil Works Construction Cost Index System (CWCCIS) factors."
    )

    # Example CWCCIS index values (placeholder). Replace with official values as needed.
    cwccis_index = {
        2018: 284.5,
        2019: 289.9,
        2020: 295.6,
        2021: 302.3,
        2022: 328.4,
        2023: 350.0,
        2024: 360.5,
    }

    original_cost = st.number_input(
        "Original project cost ($)", min_value=0.0, value=1_000_000.0, step=1000.0
    )
    base_year = st.selectbox("Base year", options=sorted(cwccis_index.keys()))
    current_year = st.selectbox(
        "Current year", options=sorted(cwccis_index.keys()), index=len(cwccis_index) - 1
    )

    base_index = cwccis_index[base_year]
    current_index = cwccis_index[current_year]
    updated_cost = original_cost * (current_index / base_index)

    st.write(f"Base index: {base_index}")
    st.write(f"Current index: {current_index}")
    st.success(f"Updated cost of storage: ${updated_cost:,.2f}")

    discount_rate = st.number_input("Real discount rate (%)", value=2.5, step=0.01)
    period = st.number_input("Period of analysis (years)", value=50, step=1)
    r = discount_rate / 100.0
    if r == 0:
        annual_cost = updated_cost / period
    else:
        annual_cost = updated_cost * (r * (1 + r) ** period) / ((1 + r) ** period - 1)
    st.info(f"Equivalent annual cost: ${annual_cost:,.2f} per year")


if page == "Expected Annual Damage":
    ead_page()
elif page == "Cost of Storage":
    cost_of_storage_page()
