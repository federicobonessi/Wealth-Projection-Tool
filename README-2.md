# UHNW Wealth Projection Tool

**A long-term wealth projection model for Ultra High Net Worth families, combining deterministic scenario analysis with Monte Carlo simulation.**

Built as part of [The Meridian Playbook](https://themeridianplaybook.com) — a research project on capital allocation, portfolio strategy and global financial systems.

---

## What It Does

This tool projects a family's wealth over a multi-decade horizon across three scenarios (base, optimistic, pessimistic), accounting for:

- Portfolio returns (net of tax drag)
- Annual withdrawals (inflation-linked)
- Capital events (business sales, inheritance, distributions)
- Inflation erosion of real purchasing power
- Monte Carlo uncertainty across 10,000 simulations

It replicates the kind of analysis a private banker or family office CIO uses in a client conversation about multigenerational wealth preservation.

---

## Output

The tool generates a single high-resolution report (`outputs/wealth_projection_report.png`) with six panels:

1. **Nominal Wealth Projection** — three scenarios with capital event markers
2. **Real Wealth** — inflation-adjusted purchasing power over time
3. **Monte Carlo Fan Chart** — P5/P25/P50/P75/P95 distribution for the base case
4. **Withdrawal Schedule** — inflation-linked annual withdrawals over the horizon
5. **Scenario Summary Table** — final wealth, CAGR, probability metrics
6. **Wealth Preservation Probability** — probability of preserving or depleting AUM at horizon

---

## Default Client Profile

```python
CLIENT = {
    "name":              "Rossi Family Office",
    "initial_wealth":    25_000_000,   # USD
    "annual_withdrawal": 500_000,      # annual living expenses + distributions
    "withdrawal_growth": 0.025,        # withdrawals grow with inflation
    "horizon_years":     30,
    "base_currency":     "USD",
}
```

---

## Scenarios

| Scenario | Net Return | Volatility | Inflation | Tax Drag |
|----------|-----------|------------|-----------|----------|
| Base Case | 7.2% | 11.0% | 2.5% | 0.8% |
| Optimistic | 9.5% | 13.0% | 2.0% | 0.6% |
| Pessimistic | 4.5% | 9.0% | 3.5% | 1.0% |

---

## Capital Events

You can model one-off events (business sales, inheritance, capital distributions):

```python
CAPITAL_EVENTS = [
    {"year": 5,  "amount":  2_000_000, "label": "Business sale proceeds"},
    {"year": 15, "amount": -1_500_000, "label": "Next-gen education & setup"},
    {"year": 20, "amount":  3_000_000, "label": "Inheritance received"},
]
```

Negative amounts represent capital outflows.

---

## Installation

```bash
git clone https://github.com/your-username/wealth-projection.git
cd wealth-projection
pip install -r requirements.txt
python src/wealth_projection.py
```

---

## Methodology

**Deterministic Projection**
Year-by-year compounding at the scenario's net return, with inflation-linked withdrawals and capital events applied sequentially.

**Real Wealth**
Nominal wealth deflated by cumulative inflation to express purchasing power in today's dollars.

**Monte Carlo Simulation**
10,000 paths using Geometric Brownian Motion with scenario-specific drift and volatility. Withdrawals and capital events are applied identically across all paths.

**Probability Metrics**
- *Prob. Preserve*: share of paths ending above initial AUM
- *Prob. Deplete*: share of paths ending at or below zero

---

## The Trilogy

This project is the third in a series of UHNW wealth management tools:

| Project | Focus |
|---------|-------|
| [Portfolio Optimizer](https://github.com/your-username/portfolio-analyzer) | Efficient frontier, Sharpe optimization |
| [Risk Scoring Model](https://github.com/your-username/risk-scoring-model) | Multi-dimensional UHNW risk assessment |
| **Wealth Projection Tool** | Long-term wealth preservation & scenario analysis |

---

## Context

This tool reflects how a private banker or family office CIO thinks about client wealth — not in terms of quarterly returns, but in terms of real purchasing power, withdrawal sustainability, and multigenerational capital preservation.

---

*Federico Bonessi — MSc Finance, IÉSEG School of Management*
*[LinkedIn](https://www.linkedin.com/in/federico-bonessi/) | [The Meridian Playbook](https://themeridianplaybook.com)*
