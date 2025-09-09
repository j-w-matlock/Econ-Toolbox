import numpy as np


def ead_trapezoidal(prob, damages):
    """Return expected annual damage via trapezoidal integration."""
    prob = np.asarray(prob, dtype=float)
    damages = np.asarray(damages, dtype=float)
    return float(np.sum(0.5 * (damages[:-1] + damages[1:]) * (prob[:-1] - prob[1:])))


def updated_storage_cost(tc, sp, storage_reallocated, total_usable_storage):
    """Compute updated cost of storage for reservoir reallocations."""
    tc = float(tc)
    sp = float(sp)
    storage_reallocated = float(storage_reallocated)
    total_usable_storage = float(total_usable_storage)
    return (tc - sp) * storage_reallocated / total_usable_storage


def interest_during_construction(
    total_initial_cost,
    rate,
    months,
    *,
    costs=None,
    timings=None,
    normalize=True,
):
    """Compute interest during construction (IDC)."""
    if months <= 0:
        return 0.0

    monthly_rate = rate / 12.0

    if costs is None:
        if not normalize:
            years = months / 12.0
            return total_initial_cost * rate * years / 8

        monthly_cost = total_initial_cost / months
        costs = [monthly_cost] * months
        timings = ["beginning"] + ["middle"] * (months - 1)
    else:
        if timings is None:
            timings = ["middle"] * len(costs)

    idc = 0.0
    for i, cost in enumerate(costs, start=1):
        timing = timings[i - 1]
        if timing == "beginning":
            remaining = months - i + 1
        elif timing == "end":
            remaining = months - i
        else:  # middle
            remaining = months - i + 0.5
        idc += cost * monthly_rate * remaining
    return idc


def capital_recovery_factor(rate, periods):
    """Return capital recovery factor for a given discount rate and period."""
    if rate == 0:
        return 1 / periods
    return rate * (1 + rate) ** periods / ((1 + rate) ** periods - 1)
