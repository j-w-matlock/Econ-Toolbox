# 🎈 Economic toolbox

A Streamlit application featuring an Expected Annual Damage (EAD) calculator,
an updated cost of storage calculator, and a project cost annualizer that
produces annualized construction costs and benefit–cost ratios.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://blank-app-template.streamlit.app/)

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```

## Formulas

The app implements standard economic engineering equations. Variables are
defined beneath each equation so results can be replicated by hand.

### Expected Annual Damage (EAD)

\[
\text{EAD} = \sum_{i=1}^{n-1} \tfrac{1}{2} (D_i + D_{i+1}) (P_i - P_{i+1})
\]

where \(P_i\) are exceedance probabilities listed from 1 down to 0 and \(D_i\)
are the corresponding damages. The summation applies the trapezoidal rule to
integrate the damage–frequency curve.

### Updated Cost of Storage

\[
\text{Updated Cost} = (TC - SP) \cdot \frac{S_r}{S_t}
\]

with total construction cost \(TC\), specific costs \(SP\), storage reallocated
\(S_r\), and total usable storage \(S_t\).

### Interest During Construction (IDC)

\[
\text{IDC} = \sum_{i=1}^{m} C_i \cdot \frac{r}{12} \cdot t_i
\]

where \(C_i\) is the cost incurred in month \(i\), \(r\) is the annual
interest rate expressed as a decimal, \(m\) is the construction period in
months, and \(t_i\) is the number of months the expenditure accrues interest.
Beginning-, middle-, and end-of-month expenditures correspond to
\(t_i = m - i + 1\), \(m - i + 0.5\), and \(m - i\), respectively. When costs
are normalized across the construction period, \(C_i = T/m\) with the first
month treated as a beginning-of-month expenditure and the remaining months at
midpoints.

### Present Value of Planned Future Costs

\[
PV = C \cdot (1 + r)^{-(y - b)}
\]

where \(C\) is a cost incurred in year \(y\), \(r\) is the discount rate, and
\(b\) is the base year.

### Capital Recovery Factor (CRF)

\[
CRF = \frac{r(1 + r)^n}{(1 + r)^n - 1}
\]

for discount rate \(r\) and period of analysis \(n\) years. If \(r = 0\), then
\(CRF = 1/n\).

### Annualized Costs and Benefit–Cost Ratio

\[
\begin{aligned}
\text{Annual Construction} &= \text{Total Investment} \cdot CRF \\
\text{Annual Total Cost} &= \text{Annual Construction} + \text{Annual O\&M} \\
\text{Benefit–Cost Ratio} &= \frac{\text{Annual Benefits}}{\text{Annual Total Cost}}
\end{aligned}
\]

## References

- U.S. Army Corps of Engineers. (1996). *Engineering Manual 1110-2-1619: Risk-Based Analysis for Flood Damage Reduction Studies*.
- U.S. Office of Management and Budget. (1992). *Circular A-94: Guidelines and Discount Rates for Benefit-Cost Analysis of Federal Programs*.
- U.S. Army Corps of Engineers. Civil Works Construction Cost Index System (CWCCIS) and Engineering News Record (ENR) cost indexes.
- U.S. Army Corps of Engineers. (2000). *Planning Guidance Notebook* (ER 1105-2-100).
