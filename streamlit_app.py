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


def updated_storage_cost(tc, sp, storage_reallocated, total_usable_storage):
    """Compute updated cost of storage for reservoir reallocations.

    Updated Cost of Storage = (TC - SP) * Storage reallocated / Total usable storage space
    where:
    * TC – total costs of construction updated using CWCCIS and ENR.
    * SP – specific costs of identifiable project features updated using CWCCIS and ENR.
    """
    tc = float(tc)
    sp = float(sp)
    storage_reallocated = float(storage_reallocated)
    total_usable_storage = float(total_usable_storage)
    return (tc - sp) * storage_reallocated / total_usable_storage


def interest_during_construction(total_initial_cost, rate, months):
    """Compute interest during construction assuming uniform expenditures."""
    years = months / 12.0
    return total_initial_cost * rate * years / 2


def capital_recovery_factor(rate, periods):
    """Return capital recovery factor for a given discount rate and period."""
    if rate == 0:
        return 1 / periods
    return rate * (1 + rate) ** periods / ((1 + rate) ** periods - 1)

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
# Updated cost of storage calculator
# ---------------------------------------------------------------------------
st.header("Updated Cost of Storage Calculator")
st.caption(
    "Reference: Civil Works Construction Cost Index System (CWCCIS) and Engineering News Record (ENR)."
)
st.info(
    "Estimate the updated cost of storage for a reservoir reallocation."
)
with st.form("storage_form"):
    tc = st.number_input(
        "Total construction cost (TC)",
        min_value=0.0,
        value=1000000.0,
        help="Total costs of construction updated using CWCCIS and ENR.",
    )
    sp = st.number_input(
        "Specific costs (SP)",
        min_value=0.0,
        value=100000.0,
        help="Costs of identifiable project features for a specific purpose, updated using CWCCIS and ENR.",
    )
    storage_reallocated = st.number_input(
        "Storage reallocated (ac-ft)",
        min_value=0.0,
        value=1000.0,
        help="Volume of storage being reallocated.",
    )
    total_usable_storage = st.number_input(
        "Total usable storage space (ac-ft)",
        min_value=0.0,
        value=10000.0,
        help="Total usable storage capacity of the project.",
    )
    compute_storage = st.form_submit_button(
        "Compute updated cost",
        help="Calculate the updated cost of storage.",
    )
if compute_storage:
    cost = updated_storage_cost(tc, sp, storage_reallocated, total_usable_storage)
    st.success(f"Updated cost of storage: ${cost:,.2f}")

# ---------------------------------------------------------------------------
# Project cost annualizer
# ---------------------------------------------------------------------------
st.header("Project Cost Annualizer")
st.info("Calculate annualized project costs and benefit-cost ratio.")

if "num_future_costs" not in st.session_state:
    st.session_state.num_future_costs = 0

if st.button("Add planned future cost"):
    st.session_state.num_future_costs += 1

with st.form("annualizer_form"):
    first_cost = st.number_input("Project First Cost ($)", min_value=0.0, value=0.0)
    real_estate_cost = st.number_input("Real Estate Cost ($)", min_value=0.0, value=0.0)
    ped_cost = st.number_input("PED Cost ($)", min_value=0.0, value=0.0)
    monitoring_cost = st.number_input("Monitoring Cost ($)", min_value=0.0, value=0.0)
    idc_rate = st.number_input(
        "Interest Rate (%) - For Interest During Construction", min_value=0.0, value=0.0
    )
    construction_months = st.number_input(
        "Construction Period (Months)", min_value=0.0, value=0.0
    )
    annual_om = st.number_input("Annual O&M Cost ($)", min_value=0.0, value=0.0)
    annual_benefits = st.number_input("Benefits (Annual, $)", min_value=0.0, value=0.0)
    base_year = st.number_input("Base Year (Year)", min_value=0, step=1, value=0)
    discount_rate = st.number_input("Discount Rate (%)", min_value=0.0, value=0.0)
    period_analysis = st.number_input(
        "Period of Analysis (Years)", min_value=1, step=1, value=1
    )

    future_costs = []
    for i in range(st.session_state.num_future_costs):
        cost = st.number_input(
            f"Planned Future Cost {i + 1} ($)", min_value=0.0, value=0.0, key=f"fcost_{i}"
        )
        year = st.number_input(
            f"Year of Cost {i + 1}", min_value=0, step=1, value=base_year, key=f"fyear_{i}"
        )
        future_costs.append((cost, year))

    compute_annual = st.form_submit_button("Compute Annual Costs")

if compute_annual:
    initial_cost = first_cost + real_estate_cost + ped_cost + monitoring_cost
    idc = interest_during_construction(initial_cost, idc_rate / 100.0, construction_months)
    total_initial = initial_cost + idc
    dr = discount_rate / 100.0
    pv_future = sum(
        cost / ((1 + dr) ** (year - base_year)) for cost, year in future_costs
    )
    total_investment = total_initial + pv_future
    crf = capital_recovery_factor(dr, period_analysis)
    annual_construction = total_investment * crf
    annual_total = annual_construction + annual_om
    bcr = annual_benefits / annual_total if annual_total else np.nan
    st.success(f"Interest During Construction: ${idc:,.2f}")
    st.success(f"Total Cost/Investment: ${total_investment:,.2f}")
    st.success(f"Capital Recovery Factor: {crf:.4f}")
    st.success(
        f"Annual Construction Cost including O&M: ${annual_total:,.2f}"
    )
    st.success(f"Benefit-Cost Ratio: {bcr:,.2f}")

