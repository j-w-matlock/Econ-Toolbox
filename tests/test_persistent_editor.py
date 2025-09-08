import pandas as pd
import streamlit as st
import streamlit_app


def test_persistent_data_editor_returns_deep_copies(monkeypatch):
    # Track the id of the object passed to the mock data_editor
    recorded = {}

    def mock_data_editor(df, key, **kwargs):
        # simulate user editing by mutating the provided df in place
        df.loc[0, "a"] = 2
        recorded["id"] = id(df)
        return df

    monkeypatch.setattr(st, "data_editor", mock_data_editor)

    original = pd.DataFrame({"a": [1]})
    edited = streamlit_app.persistent_data_editor(original, key="test")

    # original should remain unchanged
    assert original.loc[0, "a"] == 1
    # edit is applied in returned dataframe
    assert edited.loc[0, "a"] == 2
    # returned object should not be the same as the one passed to data_editor
    assert id(edited) != recorded["id"]
    # modifying the returned dataframe shouldn't affect the original
    edited.loc[0, "a"] = 3
    assert original.loc[0, "a"] == 1
