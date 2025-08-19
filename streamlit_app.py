import streamlit as st
import numpy as np
import pandas as pd
from io import BytesIO

from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.chart import LineChart, Reference

st.title("Expected Annual Damage (EAD) Calculator")
st.write(
    "Compute expected annual damages using the U.S. Army Corps of Engineers trapezoidal method."
)
st.caption(
    "Reference: U.S. Army Corps of Engineers, Engineering Manual 1110-2-1619 (1996)."
)


def ead_trapezoidal(prob, damages):
    """Return expected annual damage via trapezoidal integration.

    Implements the trapezoidal integration method described in the
    U.S. Army Corps of Engineers' risk analysis guidance
    (EM 1110-2-1619, *Risk-Based Analysis for Flood Damage Reduction Studies*,
    1996).
    """
    prob = np.asarray(prob, dtype=float)
    damages = np.asarray(damages, dtype=float)
    return float(
        np.sum(0.5 * (damages[:-1] + damages[1:]) * (prob[:-1] - prob[1:]))
    )


def storage_cost(quantity, unit_price, storage_rate, interest_rate, years):
    """Compute total storage cost given rates and time.

    The calculation follows financial analysis practices recommended in
    Office of Management and Budget Circular A-94 (1992).
    """
    quantity = float(quantity)
    unit_price = float(unit_price)
    storage_rate = float(storage_rate)
    interest_rate = float(interest_rate)
    years = float(years)
    return quantity * unit_price * (storage_rate + interest_rate) * years


# ---------------------------------------------------------------------------
# Data input section
# ---------------------------------------------------------------------------
st.subheader("Inputs")
st.info(
    "Fill in the frequency and damage values below. Use the checkbox to add a stage "
    "column and the button to insert additional damage columns as needed."
)

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
    "Include stage column",
    value="Stage" in st.session_state.table.columns,
    help="Add a column for stage values to enable stage-related charts.",
)
if include_stage and "Stage" not in st.session_state.table.columns:
    st.session_state.table.insert(1, "Stage", [None] * len(st.session_state.table))
elif not include_stage and "Stage" in st.session_state.table.columns:
    st.session_state.table.drop(columns="Stage", inplace=True)

# Allow user to add additional damage columns
if st.button("Add damage column", help="Insert another damage column to compare scenarios."):
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
    submitted = st.form_submit_button(
        "Save table", help="Apply edits to the table above."
    )
if submitted:
    st.session_state.table = data

# Plot damage-frequency curve
damage_cols = [c for c in st.session_state.table.columns if c.startswith("Damage")]
charts_for_export = []
chart_data = (
    st.session_state.table.dropna(subset=["Frequency"])
    .sort_values("Frequency")
    .set_index("Frequency")[damage_cols]
)
if not chart_data.empty and damage_cols:
    st.subheader("Damage-Frequency Curve")
    selected_damage = st.selectbox(
        "Select damage column",
        damage_cols,
        key="df_damage",
        help="Choose which damage column to visualize.",
    )
    st.line_chart(chart_data[[selected_damage]])
    charts_for_export.append(
        {
            "title": "Damage-Frequency Curve",
            "data": chart_data[[selected_damage]].reset_index(),
        }
    )

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
                "Select damage column (stage)",
                damage_cols,
                key="stage_damage",
                help="Damage column to plot against stage values.",
            )
            st.line_chart(stage_df[[dmg_col]])
            charts_for_export.append(
                {
                    "title": "Stage-Damage Curve",
                    "data": stage_df[[dmg_col]].reset_index(),
                }
            )
        if "Frequency" in stage_df.columns:
            st.subheader("Stage-Frequency Curve")
            st.line_chart(stage_df["Frequency"])
            charts_for_export.append(
                {
                    "title": "Stage-Frequency Curve",
                    "data": stage_df[["Frequency"]].reset_index(),
                }
            )

# Export table and charts to Excel
buffer = BytesIO()
wb = Workbook()
ws_data = wb.active
ws_data.title = "Data"
for row in dataframe_to_rows(st.session_state.table, index=False, header=True):
    ws_data.append(row)

if charts_for_export:
    ws_charts = wb.create_sheet("Charts")
    for chart_info in charts_for_export:
        df_chart = chart_info["data"]
        start_row = ws_charts.max_row + 2 if ws_charts.max_row > 1 else 1
        for row in dataframe_to_rows(df_chart, index=False, header=True):
            ws_charts.append(row)
        end_row = start_row + len(df_chart)
        chart = LineChart()
        chart.title = chart_info["title"]
        chart.y_axis.title = df_chart.columns[1]
        chart.x_axis.title = df_chart.columns[0]
        data_ref = Reference(ws_charts, min_col=2, min_row=start_row, max_row=end_row)
        chart.add_data(data_ref, titles_from_data=True)
        cats_ref = Reference(ws_charts, min_col=1, min_row=start_row + 1, max_row=end_row)
        chart.set_categories(cats_ref)
        ws_charts.add_chart(chart, f"E{start_row}")

wb.save(buffer)
buffer.seek(0)
st.download_button(
    label="Download data and charts as Excel",
    data=buffer,
    file_name="ead_data.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    help="Export the current table and generated charts as an Excel file.",
)


# ---------------------------------------------------------------------------
# Calculation section
# ---------------------------------------------------------------------------
st.subheader("EAD Results")
st.info(
    "Click the button below to compute expected annual damages for each damage "
    "column using trapezoidal integration."
)
if st.button(
    "Calculate EAD",
    help="Run the trapezoidal method on the frequency and damage data.",
):
    df = st.session_state.table.dropna(subset=["Frequency"]).sort_values(
        "Frequency", ascending=False
    )
    freq = df["Frequency"].to_numpy()

    # Validate that frequencies begin at 1 and decrease towards 0
    if not np.isclose(freq[0], 1.0) or np.any(np.diff(freq) > 0):
        st.warning("Frequencies should start at 1 and monotonically decrease to 0.")

    missing_zero = freq[-1] != 0
    if missing_zero:
        st.info("Final frequency not 0; appending zero-frequency point using last damage value.")

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


# ---------------------------------------------------------------------------
# Cost of storage calculator
# ---------------------------------------------------------------------------
st.header("Cost of Storage Calculator")
st.caption(
    "Reference: Office of Management and Budget, Circular A-94 (1992)."
)
st.info(
    "Estimate the total cost of storing items by providing the quantity, unit price, "
    "and annual rates."
)
with st.form("storage_form"):
    quantity = st.number_input(
        "Quantity",
        min_value=0.0,
        value=100.0,
        help="Number of items being stored.",
    )
    unit_price = st.number_input(
        "Unit price",
        min_value=0.0,
        value=10.0,
        help="Cost per item.",
    )
    storage_rate = st.number_input(
        "Annual storage cost (%)",
        min_value=0.0,
        value=2.0,
        help="Percentage of the unit price charged each year for storage.",
    ) / 100
    interest_rate = st.number_input(
        "Annual interest rate (%)",
        min_value=0.0,
        value=5.0,
        help="Opportunity cost of capital as a percentage.",
    ) / 100
    years = st.number_input(
        "Years in storage",
        min_value=0.0,
        value=1.0,
        help="Total time items remain in storage.",
    )
    compute_storage = st.form_submit_button(
        "Compute storage cost",
        help="Calculate the total cost of storage.",
    )
if compute_storage:
    cost = storage_cost(quantity, unit_price, storage_rate, interest_rate, years)
    st.success(f"Total storage cost: ${cost:,.2f}")

