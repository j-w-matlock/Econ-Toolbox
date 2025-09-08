# 🎈 Economic toolbox

A Streamlit application featuring an Expected Annual Damage (EAD) calculator,
an updated cost of storage calculator, a project cost annualizer that produces
annualized construction costs and benefit–cost ratios, and a Unit Day Value
(UDV) analysis tab for estimating recreation benefits.

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

```
EAD = sum from i = 1 to n - 1 of 0.5 * (D_i + D_{i+1}) * (P_i - P_{i+1})
```

where P_i are exceedance probabilities listed from 1 down to 0 and D_i are the
corresponding damages. The summation applies the trapezoidal rule to integrate
the damage–frequency curve.

### Updated Cost of Storage

```
Updated Cost = (TC - SP) * (S_r / S_t)
```

with total construction cost TC, specific costs SP, storage reallocated S_r, and
total usable storage S_t.

### Interest During Construction (IDC)

```
IDC = sum from i = 1 to m of C_i * (r / 12) * t_i
```

where C_i is the cost incurred in month i, r is the annual interest rate
expressed as a decimal, m is the construction period in months, and t_i is the
number of months the expenditure accrues interest. Beginning-, middle-, and
end-of-month expenditures correspond to t_i = m - i + 1, m - i + 0.5, and m - i,
respectively. When costs are normalized across the construction period, C_i =
T / m with the first month treated as a beginning-of-month expenditure and the
remaining months at midpoints.

### Present Value of Planned Future Costs

```
PV = C * (1 + r)^(-(y - b))
```

where C is a cost incurred in year y, r is the discount rate, and b is the base
year.

### Capital Recovery Factor (CRF)

```
CRF = r * (1 + r)^n / ((1 + r)^n - 1)
```

for discount rate r and period of analysis n years. If r = 0, then CRF = 1 / n.

### Annualized Costs and Benefit–Cost Ratio

```
Annual Construction = Total Investment * CRF
Annual Total Cost = Annual Construction + Annual O&M
Benefit–Cost Ratio = Annual Benefits / Annual Total Cost
```

### Recreation Benefits via Unit Day Values

```
Annual Recreation Benefit = UDV * User Days
```

where UDV is the unit day value from the latest USACE schedule and User Days
are the expected annual recreation visitations.
The application converts recreation quality point rankings to unit day values
using USACE schedules for general recreation, fishing and hunting, and other
specialized activities such as boating.

## References

- U.S. Army Corps of Engineers. (1996). *Engineering Manual 1110-2-1619: Risk-Based Analysis for Flood Damage Reduction Studies*.
- U.S. Office of Management and Budget. (1992). *Circular A-94: Guidelines and Discount Rates for Benefit-Cost Analysis of Federal Programs*.
- U.S. Army Corps of Engineers. Civil Works Construction Cost Index System (CWCCIS) and Engineering News Record (ENR) cost indexes.
- U.S. Army Corps of Engineers. (2000). *Planning Guidance Notebook* (ER 1105-2-100).
