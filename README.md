# Economic Toolbox

An interactive [Streamlit](https://streamlit.io/) application that gathers common economic-engineering calculations into a single interface.  The toolbox is organized into tabs, each implementing a specific set of formulas used by practitioners in water-resource planning and benefit–cost analysis.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://blank-app-template.streamlit.app/)

---

## Getting Started

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Every tab stores its inputs in the session state so results can be exported or revisited later in the session.

---

## Tab Reference

### 1. Expected Annual Damage (EAD)
Estimate flood damages by integrating a damage–frequency curve.  Damages \(D_i\) are paired with exceedance probabilities \(P_i\) listed from 1 down to 0.  The trapezoidal rule yields

$$
\text{EAD} = \sum_{i=1}^{n-1} \tfrac{1}{2}\left(D_i + D_{i+1}\right)\left(P_i - P_{i+1}\right).
$$

Optional stage information allows plotting stage–damage and stage–frequency relations.  Frequencies must decrease monotonically for the integration to be valid.

### 2. Storage Cost and O&M Calculator
A multi-step worksheet for estimating the annual cost of reallocating reservoir storage.

1. **Storage Capacity** – computes the share of conservation storage to be reallocated.
   $$
   P = \frac{S_{\text{rec}}}{S_{\text{tot}}}
   $$
   where \(S_{\text{rec}}\) is recommended storage and \(S_{\text{tot}}\) is total conservation storage.

2. **Joint Costs O&M** – sums annual operations and maintenance.
   $$
   O\!M_{\text{total}} = O + M
   $$

3. **Updated Storage Costs** – updates historical construction costs using cost index factors.
   $$
   C_{\text{Tot}} = \sum_{j} C_{j}^{\text{act}} \times U_{j}
   $$
   with actual costs \(C_{j}^{\text{act}}\) and update factors \(U_j\).

4. **RR&R and Mitigation** – discounts future rehabilitation/replacement and mitigation costs to the base year, updates them with the CWCCIS ratio, and annualizes using a capital recovery factor (CRF).

   Present value for each item incurred in year \(y\):
   $$
   PV = \frac{F}{(1+r)^{(y-b)}}
   $$
   Updated cost: \(C^{\*} = \text{CWCCI} \times \sum PV\)  
   Annualized cost: \(C_a = C^{\*} \times \text{CRF}(r,n)\), where
   $$
   \text{CRF}(r,n) = \frac{r(1+r)^n}{(1+r)^n-1}.
   $$

5. **Total Annual Cost** – combines capital, O&M, and RR&R/mitigation.
   $$
   C_{\text{rec}} = C_{\text{Tot}} \times P\\
   C_{\text{capital}} = C_{\text{rec}} \times \text{CRF}(r,n)\\
   O\!M_{\text{scaled}} = O\!M_{\text{total}} \times P\\
   R_{\text{scaled}} = C_a \times P\\
   \text{Total Annual Cost} = C_{\text{capital}} + O\!M_{\text{scaled}} + R_{\text{scaled}}.
   $$
   A second scenario is provided for alternative discount rates and analysis periods (without RR&R).

### 3. Project Cost Annualizer
Calculates annualized construction costs and benefit–cost ratio.

* **Interest During Construction (IDC)** distributes costs over the construction period:
  $$
  \text{IDC} = \sum_{i=1}^{m} C_i \left(\frac{r}{12}\right) t_i
  $$
  where \(C_i\) is the cost incurred in month \(i\), \(r\) the annual interest rate, and \(t_i\) the months that funds accrue interest (beginning = \(m-i+1\), middle = \(m-i+0.5\), end = \(m-i\)).
* **Present Value of Planned Future Costs**
  $$
  PV = C (1+r)^{-(y-b)}
  $$
  for a cost \(C\) in year \(y\) discounted to base year \(b\).
* **Annualization and Benefit–Cost Ratio**
  $$
  \begin{aligned}
  \text{CRF}(r,n) &= \frac{r(1+r)^n}{(1+r)^n-1},\\
  C_{\text{annual}} &= (C_0 + \text{IDC} + \sum PV) \times \text{CRF}(r,n),\\
  \text{Annual Total Cost} &= C_{\text{annual}} + O\!M,\\
  \text{BCR} &= \frac{\text{Annual Benefits}}{\text{Annual Total Cost}}.
  \end{aligned}
  $$

### 4. Recreation Benefit (Unit Day Value)
Implements USACE Unit Day Value methodology.

Adjusted user days and annual benefit are computed as
$$
\begin{aligned}
\text{Adjusted User Days} &= U \times V,\\
\text{Annual Recreation Benefit} &= \text{UDV} \times \text{Adjusted User Days},
\end{aligned}
$$
where \(U\) is expected annual user days, \(V\) is a visitation multiplier, and UDV is the unit day value determined from ranking points.

A second tab lists ranking criteria and the point-to-dollar conversion table for transparency.

### 5. Water Demand Forecast
Projects combined municipal and industrial demand following ER 1105-2-100 guidance.

For base population \(P_0\) and annual growth rate \(g\):
$$
P_t = P_0 (1+g)^t
$$
Demands are then
$$
\begin{aligned}
M_t &= \frac{P_t \times u \times 365}{10^6},\\
I_t &= M_t \times f,\\
T_t &= (M_t + I_t) \times (1+\ell),
\end{aligned}
$$
where \(u\) is per-capita municipal use (gallons/person/day), \(f\) is the industrial demand factor, and \(\ell\) represents system losses.  Results are reported in million gallons per year (MGY).

---

## References
- U.S. Army Corps of Engineers. *Engineering Manual 1110-2-1619: Risk-Based Analysis for Flood Damage Reduction Studies* (1996).
- U.S. Office of Management and Budget. *Circular A-94: Guidelines and Discount Rates for Benefit-Cost Analysis of Federal Programs* (1992).
- U.S. Army Corps of Engineers. Civil Works Construction Cost Index System (CWCCIS) and Engineering News Record (ENR) indices.
- U.S. Army Corps of Engineers. *Planning Guidance Notebook* (ER 1105-2-100) (2000).

---

## Known Issues and Areas for Improvement

- **EAD:** Assumes exceedance probabilities are pre-sorted and monotonic; automatic sorting or validation could improve usability.
- **Storage Cost and O&M:** Cost update factors and CWCCI ratios must be entered manually; automatic retrieval from current indices would reduce errors.
- **Project Cost Annualizer:** IDC calculation requires monthly detail when costs are irregular; importing a schedule from CSV would streamline entry.
- **Recreation Benefit:** UDV schedules are hard-coded for a single fiscal year; updating values when new schedules are released is manual.
- **Water Demand Forecast:** Forecast assumes constant growth and demand factors; incorporating scenario analysis or time-varying inputs would enhance realism.

---

## License

Distributed under the terms of the [MIT License](LICENSE).

