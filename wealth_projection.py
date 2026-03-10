"""
UHNW Wealth Projection Tool
==============================
Projects a family's wealth over a long-term horizon across three scenarios
(base, optimistic, pessimistic), accounting for portfolio returns, inflation,
annual withdrawals, capital events, and tax drag.

Designed to replicate the kind of tool a private banker or family office CIO
uses in a client conversation about multigenerational wealth preservation.

Author: Federico Bonessi | The Meridian Playbook
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# ─────────────────────────────────────────────
# COLOURS
# ─────────────────────────────────────────────
DARK_BG = "#0d1117"
GOLD    = "#c9a84c"
WHITE   = "#e6edf3"
GREY    = "#30363d"
MID     = "#8b949e"
RED     = "#f85149"
GREEN   = "#3fb950"
BLUE    = "#58a6ff"
ORANGE  = "#ffa657"
PURPLE  = "#a371f7"

# ─────────────────────────────────────────────
# CLIENT PROFILE
# ─────────────────────────────────────────────

CLIENT = {
    "name":              "Rossi Family Office",
    "initial_wealth":    25_000_000,   # USD
    "annual_withdrawal": 500_000,      # USD — living expenses + distributions
    "withdrawal_growth": 0.025,        # withdrawal grows with inflation
    "horizon_years":     30,
    "base_currency":     "USD",
}

# Three scenario definitions
SCENARIOS = {
    "Base Case": {
        "ann_return":   0.072,   # portfolio net return
        "ann_vol":      0.110,
        "inflation":    0.025,
        "tax_drag":     0.008,   # annual tax drag on returns
        "color":        GOLD,
        "linestyle":    "-",
    },
    "Optimistic": {
        "ann_return":   0.095,
        "ann_vol":      0.130,
        "inflation":    0.020,
        "tax_drag":     0.006,
        "color":        GREEN,
        "linestyle":    "--",
    },
    "Pessimistic": {
        "ann_return":   0.045,
        "ann_vol":      0.090,
        "inflation":    0.035,
        "tax_drag":     0.010,
        "color":        RED,
        "linestyle":    "--",
    },
}

# Capital events (inheritance, business sale, etc.)
CAPITAL_EVENTS = [
    {"year": 5,  "amount":  2_000_000, "label": "Business sale proceeds"},
    {"year": 15, "amount": -1_500_000, "label": "Next-gen education & setup"},
    {"year": 20, "amount":  3_000_000, "label": "Inheritance received"},
]

N_MONTE_CARLO = 10_000


# ─────────────────────────────────────────────
# PROJECTION ENGINE
# ─────────────────────────────────────────────

def project_deterministic(scenario: dict, client: dict, events: list) -> pd.DataFrame:
    """
    Year-by-year deterministic projection for a given scenario.
    Returns a DataFrame with wealth, real wealth, withdrawals, and events.
    """
    W0      = client["initial_wealth"]
    r       = scenario["ann_return"] - scenario["tax_drag"]
    inf     = scenario["inflation"]
    w0      = client["annual_withdrawal"]
    wg      = client["withdrawal_growth"]
    T       = client["horizon_years"]

    event_map = {e["year"]: e["amount"] for e in events}

    rows = []
    wealth = W0

    for yr in range(1, T + 1):
        # Portfolio return
        wealth = wealth * (1 + r)

        # Capital event
        event_amt = event_map.get(yr, 0)
        wealth += event_amt

        # Withdrawal (grows at withdrawal_growth rate)
        withdrawal = w0 * ((1 + wg) ** yr)
        wealth -= withdrawal

        # Real wealth (inflation-adjusted back to today)
        real_wealth = wealth / ((1 + inf) ** yr)

        rows.append({
            "year":        yr,
            "nominal":     max(wealth, 0),
            "real":        max(real_wealth, 0),
            "withdrawal":  withdrawal,
            "event":       event_amt,
        })

        if wealth <= 0:
            # Wealth depleted — fill remaining years with zeros
            for remaining in range(yr + 1, T + 1):
                rows.append({
                    "year": remaining, "nominal": 0,
                    "real": 0, "withdrawal": 0, "event": 0
                })
            break

    return pd.DataFrame(rows).set_index("year")


def project_monte_carlo(scenario: dict, client: dict,
                        events: list, n: int = N_MONTE_CARLO) -> dict:
    """
    Monte Carlo simulation of wealth paths for a given scenario.
    Returns distribution statistics at each year.
    """
    W0  = client["initial_wealth"]
    r   = scenario["ann_return"] - scenario["tax_drag"]
    vol = scenario["ann_vol"]
    inf = scenario["inflation"]
    w0  = client["annual_withdrawal"]
    wg  = client["withdrawal_growth"]
    T   = client["horizon_years"]

    event_map = {e["year"]: e["amount"] for e in events}

    # Shape: (n_simulations, horizon)
    paths = np.zeros((n, T))
    wealth = np.full(n, float(W0))

    for yr in range(1, T + 1):
        # Stochastic return
        annual_ret = np.random.normal(r, vol, n)
        wealth = wealth * (1 + annual_ret)

        # Capital event
        wealth += event_map.get(yr, 0)

        # Withdrawal
        withdrawal = w0 * ((1 + wg) ** yr)
        wealth -= withdrawal

        # Floor at zero
        wealth = np.maximum(wealth, 0)

        paths[:, yr - 1] = wealth

    years = np.arange(1, T + 1)
    return {
        "years":   years,
        "paths":   paths,
        "p5":      np.percentile(paths, 5,  axis=0),
        "p25":     np.percentile(paths, 25, axis=0),
        "p50":     np.percentile(paths, 50, axis=0),
        "p75":     np.percentile(paths, 75, axis=0),
        "p95":     np.percentile(paths, 95, axis=0),
        "mean":    paths.mean(axis=0),
        "prob_preserve": (paths[:, -1] > W0).mean(),
        "prob_deplete":  (paths[:, -1] <= 0).mean(),
    }


# ─────────────────────────────────────────────
# SUMMARY METRICS
# ─────────────────────────────────────────────

def compute_summary(det: dict, mc: dict, client: dict) -> dict:
    W0 = client["initial_wealth"]
    T  = client["horizon_years"]

    out = {}
    for name, df in det.items():
        final_nom  = df["nominal"].iloc[-1]
        final_real = df["real"].iloc[-1]
        total_withdrawn = df["withdrawal"].sum()
        cagr = (final_nom / W0) ** (1 / T) - 1 if final_nom > 0 else -1
        out[name] = {
            "final_nominal":  final_nom,
            "final_real":     final_real,
            "total_withdrawn": total_withdrawn,
            "cagr":           cagr,
            "prob_preserve":  mc[name]["prob_preserve"],
            "prob_deplete":   mc[name]["prob_deplete"],
        }
    return out


# ─────────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────────

def make_report(client, det, mc, summary, events):
    fig = plt.figure(figsize=(22, 26), facecolor=DARK_BG)
    gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

    def style_ax(ax, title=""):
        ax.set_facecolor(DARK_BG)
        ax.tick_params(colors=WHITE, labelsize=9)
        ax.spines[:].set_color(GREY)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_color(WHITE)
        if title:
            ax.set_title(title, color=GOLD, fontsize=11, fontweight="bold", pad=10)

    usd_m = FuncFormatter(lambda x, _: f"${x/1e6:.1f}M")
    pct_f = FuncFormatter(lambda x, _: f"{x:.0%}")
    years = np.arange(1, client["horizon_years"] + 1)
    W0    = client["initial_wealth"]

    # ── TITLE
    ax0 = fig.add_subplot(gs[0, :])
    ax0.set_facecolor(DARK_BG); ax0.axis("off")
    ax0.text(0.5, 0.80, "UHNW WEALTH PROJECTION TOOL",
             ha="center", color=GOLD, fontsize=22, fontweight="bold", transform=ax0.transAxes)
    ax0.text(0.5, 0.52,
             f"{client['name']}  |  Initial AUM: ${W0/1e6:.0f}M  |  "
             f"Annual Withdrawal: ${client['annual_withdrawal']/1e3:.0f}K  |  "
             f"Horizon: {client['horizon_years']} Years",
             ha="center", color=WHITE, fontsize=11, transform=ax0.transAxes)
    ax0.text(0.5, 0.22, "Three-Scenario Deterministic & Monte Carlo Analysis",
             ha="center", color=MID, fontsize=10, transform=ax0.transAxes)
    ax0.axhline(0.08, color=GOLD, linewidth=0.8, xmin=0.1, xmax=0.9)

    # ── 1. NOMINAL WEALTH — three scenarios
    ax1 = fig.add_subplot(gs[1, :2])
    for name, sc in SCENARIOS.items():
        df = det[name]
        ax1.plot(df.index, df["nominal"], color=sc["color"],
                 linestyle=sc["linestyle"], linewidth=2, label=name)
    ax1.axhline(W0, color=WHITE, linewidth=0.8, linestyle=":", alpha=0.4, label="Initial AUM")
    # Mark capital events
    for ev in events:
        ax1.axvline(ev["year"], color=GREY, linewidth=0.8, linestyle="--", alpha=0.6)
        ax1.text(ev["year"] + 0.3, ax1.get_ylim()[1] * 0.95 if ax1.get_ylim()[1] > 0 else W0 * 2,
                 ev["label"][:18], color=MID, fontsize=7, rotation=90, va="top")
    ax1.yaxis.set_major_formatter(usd_m)
    ax1.set_xlabel("Year", color=WHITE, fontsize=9)
    ax1.set_ylabel("Nominal Wealth (USD)", color=WHITE, fontsize=9)
    ax1.legend(fontsize=9, labelcolor=WHITE, facecolor=GREY, edgecolor=GREY)
    style_ax(ax1, "Nominal Wealth Projection — Three Scenarios")

    # ── 2. REAL WEALTH (inflation-adjusted)
    ax2 = fig.add_subplot(gs[1, 2])
    for name, sc in SCENARIOS.items():
        df = det[name]
        ax2.plot(df.index, df["real"], color=sc["color"],
                 linestyle=sc["linestyle"], linewidth=1.8, label=name)
    ax2.axhline(W0, color=WHITE, linewidth=0.8, linestyle=":", alpha=0.4)
    ax2.yaxis.set_major_formatter(usd_m)
    ax2.set_xlabel("Year", color=WHITE, fontsize=9)
    ax2.set_ylabel("Real Wealth (Today's USD)", color=WHITE, fontsize=9)
    ax2.legend(fontsize=7, labelcolor=WHITE, facecolor=GREY, edgecolor=GREY)
    style_ax(ax2, "Real Wealth (Inflation-Adjusted)")

    # ── 3. MONTE CARLO — BASE CASE fan chart
    ax3 = fig.add_subplot(gs[2, :2])
    mc_base = mc["Base Case"]
    ax3.fill_between(mc_base["years"], mc_base["p5"],  mc_base["p95"],
                     alpha=0.12, color=GOLD, label="P5–P95")
    ax3.fill_between(mc_base["years"], mc_base["p25"], mc_base["p75"],
                     alpha=0.25, color=GOLD, label="P25–P75")
    ax3.plot(mc_base["years"], mc_base["p50"],  color=GOLD,  linewidth=2,   label="Median")
    ax3.plot(mc_base["years"], mc_base["mean"], color=WHITE, linewidth=1.2,
             linestyle="--", label="Mean")
    ax3.plot(mc_base["years"], mc_base["p5"],   color=RED,   linewidth=1,
             linestyle=":", label="5th Percentile")
    ax3.axhline(W0, color=WHITE, linewidth=0.8, linestyle=":", alpha=0.4, label="Initial AUM")
    ax3.yaxis.set_major_formatter(usd_m)
    ax3.set_xlabel("Year", color=WHITE, fontsize=9)
    ax3.set_ylabel("Wealth (USD)", color=WHITE, fontsize=9)
    ax3.legend(fontsize=8, labelcolor=WHITE, facecolor=GREY, edgecolor=GREY, ncol=3)
    style_ax(ax3, f"Monte Carlo Fan Chart — Base Case ({N_MONTE_CARLO:,} Simulations)")

    # ── 4. WITHDRAWAL SCHEDULE
    ax4 = fig.add_subplot(gs[2, 2])
    w0   = client["annual_withdrawal"]
    wg   = client["withdrawal_growth"]
    wdrs = [w0 * ((1 + wg) ** yr) for yr in years]
    ax4.bar(years, wdrs, color=BLUE, alpha=0.7, width=0.8)
    ax4.yaxis.set_major_formatter(usd_m)
    ax4.set_xlabel("Year", color=WHITE, fontsize=9)
    ax4.set_ylabel("Annual Withdrawal (USD)", color=WHITE, fontsize=9)
    style_ax(ax4, f"Withdrawal Schedule\n(+{wg:.1%}/yr inflation-linked)")

    # ── 5. SUMMARY TABLE
    ax5 = fig.add_subplot(gs[3, :2])
    ax5.set_facecolor(DARK_BG); ax5.axis("off")
    ax5.set_title("Scenario Summary", color=GOLD, fontsize=11, fontweight="bold", pad=10)

    headers = ["Scenario", "Final Wealth (Nom.)", "Final Wealth (Real)",
               "Total Withdrawn", "CAGR", "Prob. Preserve", "Prob. Deplete"]
    col_x   = [0.01, 0.16, 0.31, 0.46, 0.60, 0.72, 0.86]

    for j, h in enumerate(headers):
        ax5.text(col_x[j], 0.92, h, transform=ax5.transAxes,
                 color=GOLD, fontsize=8, fontweight="bold")
    ax5.plot([0.01, 0.99], [0.88, 0.88], color=GOLD, linewidth=0.5,
             transform=ax5.transAxes)

    for i, (name, sc) in enumerate(SCENARIOS.items()):
        s   = summary[name]
        y   = 0.76 - i * 0.22
        col = sc["color"]
        vals = [
            name,
            f"${s['final_nominal']/1e6:.1f}M",
            f"${s['final_real']/1e6:.1f}M",
            f"${s['total_withdrawn']/1e6:.1f}M",
            f"{s['cagr']:.2%}",
            f"{s['prob_preserve']:.1%}",
            f"{s['prob_deplete']:.1%}",
        ]
        for j, v in enumerate(vals):
            c = col if j == 0 else WHITE
            if j == 6 and s["prob_deplete"] > 0.05: c = RED
            if j == 5 and s["prob_preserve"] > 0.7: c = GREEN
            ax5.text(col_x[j], y, v, transform=ax5.transAxes,
                     color=c, fontsize=9)

    # ── 6. PROBABILITY BARS
    ax6 = fig.add_subplot(gs[3, 2])
    scenario_names = list(SCENARIOS.keys())
    p_preserve = [summary[n]["prob_preserve"] for n in scenario_names]
    p_deplete  = [summary[n]["prob_deplete"]  for n in scenario_names]
    sc_colors  = [SCENARIOS[n]["color"] for n in scenario_names]

    x = np.arange(len(scenario_names))
    bars1 = ax6.bar(x - 0.2, p_preserve, width=0.35, color=GREEN,  alpha=0.8, label="Prob. Preserve")
    bars2 = ax6.bar(x + 0.2, p_deplete,  width=0.35, color=RED,    alpha=0.8, label="Prob. Deplete")
    ax6.set_xticks(x)
    ax6.set_xticklabels([n.replace(" ", "\n") for n in scenario_names], fontsize=8)
    ax6.yaxis.set_major_formatter(pct_f)
    ax6.legend(fontsize=8, labelcolor=WHITE, facecolor=GREY, edgecolor=GREY)
    for bar in bars1:
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{bar.get_height():.0%}", ha="center", color=GREEN, fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        if h > 0.001:
            ax6.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                     f"{h:.1%}", ha="center", color=RED, fontsize=8)
    style_ax(ax6, "Wealth Preservation Probability\nat 30-Year Horizon")

    # FOOTER
    fig.text(0.5, 0.005,
             "The Meridian Playbook  |  Research on Capital Allocation & Financial Systems"
             "  |  themeridianplaybook.com",
             ha="center", color=GREY, fontsize=8)

    import os; os.makedirs("outputs", exist_ok=True)
    out = "outputs/wealth_projection_report.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK_BG)
    plt.close()
    print(f"    ✓ Report saved → {out}\n")
    return out


# ─────────────────────────────────────────────
# PRINT CONSOLE SUMMARY
# ─────────────────────────────────────────────

def print_summary(summary, client):
    W0 = client["initial_wealth"]
    T  = client["horizon_years"]
    print("=" * 65)
    print(f"  WEALTH PROJECTION — {client['name'].upper()}")
    print(f"  Initial AUM: ${W0/1e6:.0f}M  |  Horizon: {T} years")
    print("=" * 65)
    for name, s in summary.items():
        print(f"\n  [{name}]")
        print(f"  Final Wealth (Nominal):  ${s['final_nominal']/1e6:.2f}M")
        print(f"  Final Wealth (Real):     ${s['final_real']/1e6:.2f}M")
        print(f"  Total Withdrawn:         ${s['total_withdrawn']/1e6:.2f}M")
        print(f"  Portfolio CAGR:          {s['cagr']:.2%}")
        print(f"  Prob. of Preserving AUM: {s['prob_preserve']:.1%}")
        print(f"  Prob. of Depletion:      {s['prob_deplete']:.2%}")
    print("\n" + "=" * 65)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("\n╔══════════════════════════════════════════════╗")
    print("║   UHNW WEALTH PROJECTION TOOL                ║")
    print("║   The Meridian Playbook                      ║")
    print("╚══════════════════════════════════════════════╝\n")

    print("📐  Running deterministic projections...")
    det = {name: project_deterministic(sc, CLIENT, CAPITAL_EVENTS)
           for name, sc in SCENARIOS.items()}

    print("🎲  Running Monte Carlo simulations...")
    mc = {name: project_monte_carlo(sc, CLIENT, CAPITAL_EVENTS)
          for name, sc in SCENARIOS.items()}

    summary = compute_summary(det, mc, CLIENT)
    print_summary(summary, CLIENT)

    print("📊  Generating report...")
    make_report(CLIENT, det, mc, summary, CAPITAL_EVENTS)

    print("✅  Analysis complete.")
    print("    Open outputs/wealth_projection_report.png\n")


if __name__ == "__main__":
    main()
