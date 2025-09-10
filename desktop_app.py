"""Tkinter-based desktop interface for core economic toolbox functions.

This module mirrors the capabilities of the Streamlit app while remaining
independent from it.  Each tab in the interface exposes one of the core
calculations provided in :mod:`toolbox` so the project can be used offline
as a traditional desktop program.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk, messagebox

from toolbox import (
    capital_recovery_factor,
    ead_trapezoidal,
    interest_during_construction,
    updated_storage_cost,
)


class EconToolboxApp(tk.Tk):
    """Desktop user interface for the economic toolbox.

    The application uses :class:`ttk.Notebook` to provide individual tabs for
    each calculation.  The design intentionally keeps the Streamlit version
    untouched; this script can be executed independently to launch a local
    desktop application.
    """

    def __init__(self) -> None:
        super().__init__()
        self.title("Economic Toolbox Desktop")
        self.geometry("420x300")

        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True)

        self._build_crf_tab(notebook)
        self._build_ead_tab(notebook)
        self._build_storage_tab(notebook)
        self._build_idc_tab(notebook)

    # ------------------------------------------------------------------
    # Tab builders
    # ------------------------------------------------------------------
    def _build_crf_tab(self, notebook: ttk.Notebook) -> None:
        """Capital recovery factor tab."""

        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Capital Recovery")

        ttk.Label(frame, text="Rate (%)").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.crf_rate = tk.DoubleVar(value=5.0)
        ttk.Entry(frame, textvariable=self.crf_rate).grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(frame, text="Periods").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.crf_periods = tk.IntVar(value=10)
        ttk.Entry(frame, textvariable=self.crf_periods).grid(row=1, column=1, padx=5, pady=5)

        ttk.Button(frame, text="Compute", command=self._compute_crf).grid(
            row=2, column=0, columnspan=2, pady=10
        )
        self.crf_result = ttk.Label(frame, text="")
        self.crf_result.grid(row=3, column=0, columnspan=2)

    def _build_ead_tab(self, notebook: ttk.Notebook) -> None:
        """Expected annual damage (EAD) tab."""

        frame = ttk.Frame(notebook)
        notebook.add(frame, text="EAD")

        ttk.Label(
            frame,
            text="Probabilities (comma separated, descending)",
        ).grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.ead_prob = tk.StringVar()
        ttk.Entry(frame, textvariable=self.ead_prob, width=40).grid(
            row=0, column=1, padx=5, pady=5
        )

        ttk.Label(
            frame,
            text="Damages (comma separated)",
        ).grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.ead_damages = tk.StringVar()
        ttk.Entry(frame, textvariable=self.ead_damages, width=40).grid(
            row=1, column=1, padx=5, pady=5
        )

        ttk.Button(frame, text="Compute", command=self._compute_ead).grid(
            row=2, column=0, columnspan=2, pady=10
        )
        self.ead_result = ttk.Label(frame, text="")
        self.ead_result.grid(row=3, column=0, columnspan=2)

    def _build_storage_tab(self, notebook: ttk.Notebook) -> None:
        """Updated storage cost tab."""

        frame = ttk.Frame(notebook)
        notebook.add(frame, text="Storage Cost")

        labels = [
            "Total Cost (Tc)",
            "Storage Price (Sp)",
            "Storage Reallocated",
            "Total Usable Storage",
        ]
        vars_ = []
        for idx, lbl in enumerate(labels):
            ttk.Label(frame, text=lbl).grid(row=idx, column=0, padx=5, pady=5, sticky="w")
            var = tk.DoubleVar(value=0.0)
            ttk.Entry(frame, textvariable=var).grid(row=idx, column=1, padx=5, pady=5)
            vars_.append(var)

        (
            self.tc_var,
            self.sp_var,
            self.storage_realloc_var,
            self.total_storage_var,
        ) = vars_

        ttk.Button(frame, text="Compute", command=self._compute_storage).grid(
            row=4, column=0, columnspan=2, pady=10
        )
        self.storage_result = ttk.Label(frame, text="")
        self.storage_result.grid(row=5, column=0, columnspan=2)

    def _build_idc_tab(self, notebook: ttk.Notebook) -> None:
        """Interest during construction tab."""

        frame = ttk.Frame(notebook)
        notebook.add(frame, text="IDC")

        ttk.Label(frame, text="Total Initial Cost").grid(
            row=0, column=0, padx=5, pady=5, sticky="w"
        )
        self.idc_total = tk.DoubleVar(value=0.0)
        ttk.Entry(frame, textvariable=self.idc_total).grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(frame, text="Rate (%)").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.idc_rate = tk.DoubleVar(value=5.0)
        ttk.Entry(frame, textvariable=self.idc_rate).grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(frame, text="Months").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self.idc_months = tk.IntVar(value=12)
        ttk.Entry(frame, textvariable=self.idc_months).grid(row=2, column=1, padx=5, pady=5)

        ttk.Label(
            frame, text="Monthly Costs (optional, comma separated)"
        ).grid(row=3, column=0, padx=5, pady=5, sticky="w")
        self.idc_costs = tk.StringVar()
        ttk.Entry(frame, textvariable=self.idc_costs, width=40).grid(
            row=3, column=1, padx=5, pady=5
        )

        ttk.Label(
            frame, text="Timings (optional, begin/middle/end)"
        ).grid(row=4, column=0, padx=5, pady=5, sticky="w")
        self.idc_timings = tk.StringVar()
        ttk.Entry(frame, textvariable=self.idc_timings, width=40).grid(
            row=4, column=1, padx=5, pady=5
        )

        ttk.Button(frame, text="Compute", command=self._compute_idc).grid(
            row=5, column=0, columnspan=2, pady=10
        )
        self.idc_result = ttk.Label(frame, text="")
        self.idc_result.grid(row=6, column=0, columnspan=2)

    # ------------------------------------------------------------------
    # Calculation callbacks
    # ------------------------------------------------------------------
    def _compute_crf(self) -> None:
        rate = self.crf_rate.get() / 100.0
        periods = self.crf_periods.get()
        try:
            crf = capital_recovery_factor(rate, periods)
        except Exception as exc:  # pragma: no cover - UI feedback
            messagebox.showerror("Error", str(exc))
        else:
            self.crf_result.config(text=f"Capital recovery factor: {crf:.6f}")

    def _compute_ead(self) -> None:
        try:
            probs = [float(x) for x in self.ead_prob.get().split(",") if x.strip()]
            damages = [
                float(x) for x in self.ead_damages.get().split(",") if x.strip()
            ]
            if len(probs) != len(damages):
                raise ValueError("Probability and damage counts must match")
            result = ead_trapezoidal(probs, damages)
        except Exception as exc:  # pragma: no cover - UI feedback
            messagebox.showerror("Error", str(exc))
        else:
            self.ead_result.config(text=f"Expected annual damage: {result:.2f}")

    def _compute_storage(self) -> None:
        try:
            result = updated_storage_cost(
                self.tc_var.get(),
                self.sp_var.get(),
                self.storage_realloc_var.get(),
                self.total_storage_var.get(),
            )
        except Exception as exc:  # pragma: no cover - UI feedback
            messagebox.showerror("Error", str(exc))
        else:
            self.storage_result.config(text=f"Updated cost: {result:.2f}")

    def _compute_idc(self) -> None:
        try:
            costs = (
                [float(x) for x in self.idc_costs.get().split(",") if x.strip()]
                if self.idc_costs.get().strip()
                else None
            )
            timings = (
                [x.strip() for x in self.idc_timings.get().split(",") if x.strip()]
                if self.idc_timings.get().strip()
                else None
            )
            result = interest_during_construction(
                self.idc_total.get(),
                self.idc_rate.get() / 100.0,
                self.idc_months.get(),
                costs=costs,
                timings=timings,
            )
        except Exception as exc:  # pragma: no cover - UI feedback
            messagebox.showerror("Error", str(exc))
        else:
            self.idc_result.config(text=f"Interest during construction: {result:.2f}")


if __name__ == "__main__":  # pragma: no cover - manual invocation
    app = EconToolboxApp()
    app.mainloop()

