import streamlit as st
import pandas as pd
import openpyxl
import pathlib, sys
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streamlit_app import build_excel, capital_recovery_factor


def test_build_excel_includes_storage_sheets():
    # Clear any existing state
    st.session_state.clear()

    # Populate session state with sample data for each storage section
    st.session_state.storage_capacity = {"STot": 100.0, "SRec": 50.0, "P": 0.5}
    st.session_state.joint_om = {
        "operations": 10.0,
        "maintenance": 5.0,
        "total": 15.0,
    }
    usc_df = pd.DataFrame(
        {
            "Category": ["A", "B"],
            "Actual Cost": [1.0, 2.0],
            "Update Factor": [1.0, 1.0],
        }
    )
    usc_df = usc_df.assign(**{"Updated Cost": usc_df["Actual Cost"] * usc_df["Update Factor"]})
    st.session_state.updated_storage = {"table": usc_df, "CTot": float(usc_df["Updated Cost"].sum())}
    rrr_df = pd.DataFrame(
        {
            "Item": ["A"],
            "Future Cost": [100.0],
            "Year": [2020],
            "PV Factor": [1.0],
            "Present Value": [100.0],
        }
    )
    st.session_state.rrr_mit = {
        "rate": 5.0,
        "periods": 30,
        "cwcci": 1.0,
        "base_year": 2020,
        "table": rrr_df,
        "total_pv": 100.0,
        "updated_cost": 100.0,
        "annualized": 10.0,
    }
    st.session_state.total_annual_cost_inputs = {
        "rate1": 5.0,
        "periods1": 30,
        "rate2": 6.0,
        "periods2": 40,
    }
    p = st.session_state.storage_capacity["P"]
    ctot = st.session_state.updated_storage["CTot"]
    inputs = st.session_state.total_annual_cost_inputs
    cap1 = ctot * p * capital_recovery_factor(inputs["rate1"] / 100.0, inputs["periods1"])
    cap2 = ctot * p * capital_recovery_factor(inputs["rate2"] / 100.0, inputs["periods2"])
    om_scaled = st.session_state.joint_om["total"] * p
    rrr_scaled = st.session_state.rrr_mit["annualized"] * p
    total1 = cap1 + om_scaled + rrr_scaled
    total2 = cap2 + om_scaled + rrr_scaled
    st.session_state.storage_cost = {"scenario1": total1, "scenario2": total2}

    buffer = build_excel()
    wb = openpyxl.load_workbook(buffer)

    expected = [
        "Storage Capacity",
        "Joint Costs O&M",
        "Updated Storage Costs",
        "RR&R and Mitigation",
        "Total Annual Cost",
    ]
    for name in expected:
        assert name in wb.sheetnames

    ws_sc = wb["Storage Capacity"]
    assert ws_sc["A1"].value == "Total Usable Storage (STot)"
    assert ws_sc["B3"].value == 0.5

    ws_jom = wb["Joint Costs O&M"]
    assert ws_jom["A3"].value == "Total Joint O&M"
    assert ws_jom["B1"].value == 10.0

    ws_usc = wb["Updated Storage Costs"]
    assert ws_usc["A1"].value == "Category"

    ws_rrr = wb["RR&R and Mitigation"]
    assert ws_rrr["A1"].value == "Federal Discount Rate (%)"
    assert ws_rrr["A4"].value == "Base Year"
    assert ws_rrr["A6"].value == "Item"
    assert ws_rrr["A9"].value == "Total Present Value Cost"

    ws_tac = wb["Total Annual Cost"]
    assert ws_tac["A2"].value == "Percent of Total Conservation Storage (P)"
    assert ws_tac["A3"].value == "Cost of Storage Recommendation"
    assert ws_tac["A4"].value == "Annualized Storage Cost"
    assert ws_tac["A5"].value == "Joint O&M"
    assert ws_tac["A6"].value == "Annualized RR&R/Mitigation"
    assert ws_tac["A7"].value == "Total Annual Cost"
    assert ws_tac["B2"].value == 0.5
    assert ws_tac["C2"].value == 0.5
    assert ws_tac["B3"].value == 1.5
    assert ws_tac["C3"].value == 1.5
    assert ws_tac["B4"].value == pytest.approx(cap1)
    assert ws_tac["C4"].value == pytest.approx(cap2)
    assert ws_tac["B5"].value == pytest.approx(om_scaled)
    assert ws_tac["C5"].value == pytest.approx(om_scaled)
    assert ws_tac["B6"].value == pytest.approx(rrr_scaled)
    assert ws_tac["C6"].value == pytest.approx(rrr_scaled)
    assert ws_tac["B7"].value == pytest.approx(total1)
    assert ws_tac["C7"].value == pytest.approx(total2)
    assert ws_tac["A8"].value == "Discount Rate (%) for Storage Cost"
    assert ws_tac["B8"].value == 5.0
    assert ws_tac["C8"].value == 6.0
    assert ws_tac["A9"].value == "Analysis Period (years)"
    assert ws_tac["B9"].value == 30
    assert ws_tac["C9"].value == 40
