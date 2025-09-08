import streamlit as st
import pandas as pd
import openpyxl
import pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from streamlit_app import build_excel


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
    st.session_state.rrr_mit = {
        "rate": 5.0,
        "periods": 30,
        "cwcci": 1.0,
        "base_cost": 100.0,
        "updated_cost": 100.0,
        "annualized": 10.0,
    }
    st.session_state.total_annual_cost_inputs = {
        "rate1": 5.0,
        "periods1": 30,
        "rate2": 6.0,
        "periods2": 40,
    }
    st.session_state.storage_cost = {"scenario1": 25.0, "scenario2": 30.0}

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

    ws_tac = wb["Total Annual Cost"]
    assert ws_tac["A2"].value == "Percent of Total Conservation Storage (P)"
    assert ws_tac["B6"].value == 25.0
    assert ws_tac["C6"].value == 30.0
