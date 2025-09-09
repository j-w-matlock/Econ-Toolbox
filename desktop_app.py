import tkinter as tk
from tkinter import ttk, messagebox
from toolbox import capital_recovery_factor


class EconToolboxApp(tk.Tk):
    """Simple desktop interface for core toolbox calculations."""

    def __init__(self):
        super().__init__()
        self.title("Economic Toolbox Desktop")
        self.geometry("360x160")

        ttk.Label(self, text="Rate (%)").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.rate_var = tk.DoubleVar(value=5.0)
        ttk.Entry(self, textvariable=self.rate_var).grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(self, text="Periods").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        self.periods_var = tk.IntVar(value=10)
        ttk.Entry(self, textvariable=self.periods_var).grid(row=1, column=1, padx=5, pady=5)

        ttk.Button(self, text="Compute CRF", command=self.compute).grid(
            row=2, column=0, columnspan=2, pady=10
        )

    def compute(self):
        """Calculate and display the capital recovery factor."""
        rate = self.rate_var.get() / 100.0
        periods = self.periods_var.get()
        crf = capital_recovery_factor(rate, periods)
        messagebox.showinfo("Result", f"Capital recovery factor: {crf:.6f}")


if __name__ == "__main__":
    app = EconToolboxApp()
    app.mainloop()
