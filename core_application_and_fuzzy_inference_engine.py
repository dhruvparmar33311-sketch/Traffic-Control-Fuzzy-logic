import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


def trapmf(x, a, b, c, d):
    x = np.asarray(x, dtype=float)
    left = np.ones_like(x) if a == b else (x - a) / (b - a)
    right = np.ones_like(x) if c == d else (d - x) / (d - c)
    return np.maximum(0.0, np.minimum(np.minimum(left, 1.0), right))


def trimf(x, a, b, c):
    x = np.asarray(x, dtype=float)
    left = np.zeros_like(x) if a == b else (x - a) / (b - a)
    right = np.zeros_like(x) if b == c else (c - x) / (c - b)
    return np.maximum(0.0, np.minimum(left, right))


DENSITY_MFS = {
    "Low": ("trap", (0, 0, 20, 40)),
    "Medium": ("tri", (20, 50, 80)),
    "High": ("trap", (60, 80, 100, 100)),
}

WAIT_MFS = {
    "Short": ("trap", (0, 0, 25, 45)),
    "Moderate": ("tri", (25, 60, 95)),
    "Long": ("trap", (75, 95, 120, 120)),
}

GREEN_MFS = {
    "Very Short": ("trap", (10, 10, 15, 25)),
    "Short": ("tri", (15, 30, 45)),
    "Medium": ("tri", (35, 50, 65)),
    "Long": ("tri", (55, 70, 85)),
    "Very Long": ("trap", (75, 85, 90, 90)),
}

RULE_BASE = [
    ("R1", "Low", "Short", "Very Short"),
    ("R2", "Low", "Moderate", "Short"),
    ("R3", "Low", "Long", "Medium"),
    ("R4", "Medium", "Short", "Short"),
    ("R5", "Medium", "Moderate", "Medium"),
    ("R6", "Medium", "Long", "Long"),
    ("R7", "High", "Short", "Medium"),
    ("R8", "High", "Moderate", "Long"),
    ("R9", "High", "Long", "Very Long"),
]


def eval_mf(x, mf_spec):
    kind, params = mf_spec
    if kind == "trap":
        return trapmf(x, *params)
    return trimf(x, *params)


def fuzzify_inputs(density, wait_time):
    d_arr = np.array([density], dtype=float)
    w_arr = np.array([wait_time], dtype=float)
    return {
        "density": {k: float(eval_mf(d_arr, v)[0]) for k, v in DENSITY_MFS.items()},
        "wait_time": {k: float(eval_mf(w_arr, v)[0]) for k, v in WAIT_MFS.items()},
    }


def infer_traffic_control(density, wait_time, n_intervals=800):
    if not (0.0 <= density <= 100.0 and 0.0 <= wait_time <= 120.0):
        raise ValueError("Inputs outside operational domain.")

    fuzzified = fuzzify_inputs(density, wait_time)
    z = np.linspace(10.0, 90.0, n_intervals + 1)
    agg_mf = np.zeros_like(z)
    active_rules = []

    for rule_id, d_term, w_term, g_term in RULE_BASE:
        alpha = min(fuzzified["density"][d_term], fuzzified["wait_time"][w_term])
        if alpha > 0.0:
            active_rules.append({
                "Rule ID": rule_id,
                "Description": f"Density {d_term} AND Waiting {w_term} -> {g_term}",
                "Firing Strength": round(alpha, 2),
            })
            consequent_mf = eval_mf(z, GREEN_MFS[g_term])
            clipped_mf = np.minimum(alpha, consequent_mf)
            agg_mf = np.maximum(agg_mf, clipped_mf)

    area = float(np.sum(agg_mf))
    if area == 0.0:
        green_time = 10.0
    else:
        green_time = float(np.sum(z * agg_mf) / area)

    return round(green_time, 2), active_rules, fuzzified, z, agg_mf


def save_membership_plot(filepath):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle("Fuzzy Membership Functions (Traffic Congestion Controller Parameters)", fontsize=13)

    d_grid = np.linspace(0, 100, 1001)
    for label, spec in DENSITY_MFS.items():
        axes[0].plot(d_grid, eval_mf(d_grid, spec), linewidth=2, label=label)
    axes[0].set_title("Traffic Density (veh/km)")
    axes[0].set_xlabel("Density D in [0, 100]")
    axes[0].set_ylabel("Membership degree")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    w_grid = np.linspace(0, 120, 1201)
    for label, spec in WAIT_MFS.items():
        axes[1].plot(w_grid, eval_mf(w_grid, spec), linewidth=2, label=label)
    axes[1].set_title("Waiting Time (seconds)")
    axes[1].set_xlabel("Waiting Time W in [0, 120]")
    axes[1].set_ylabel("Membership degree")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    g_grid = np.linspace(10, 90, 801)
    for label, spec in GREEN_MFS.items():
        axes[2].plot(g_grid, eval_mf(g_grid, spec), linewidth=2, label=label)
    axes[2].set_title("Green Light Extension G (seconds)")
    axes[2].set_xlabel("Green Time G in [10, 90]")
    axes[2].set_ylabel("Membership degree")
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend()

    plt.tight_layout()
    fig.savefig(filepath, dpi=200)
    plt.close(fig)


def plot_aggregated_output(z, agg_mf, green_time, density, wait_time):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(z, agg_mf, color="#0284c7", linewidth=2.5)
    ax.fill_between(z, 0, agg_mf, color="#38bdf8", alpha=0.35)
    ax.axvline(
        green_time,
        color="#ef4444",
        linestyle="--",
        linewidth=2,
        label=f"Centroid Defuzzified Value: {green_time:.1f} s",
    )
    ax.set_title(f"Aggregated Fuzzy Output Membership (Case: D={density:.0f}, W={wait_time:.0f})")
    ax.set_xlabel("Green Light Duration G (seconds)")
    ax.set_ylabel("Membership degree")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")
    plt.tight_layout()
    return fig


def run_benchmark_suite():
    os.makedirs("outputs", exist_ok=True)
    save_membership_plot("outputs/membership_functions.png")

    scenarios = [
        ("Empty approach", 10.0, 15.0),
        ("Congested peak", 85.0, 105.0),
        ("Moderate traffic", 50.0, 60.0),
        ("Heavy queue build-up", 55.0, 100.0),
        ("Sudden burst", 80.0, 20.0),
    ]

    records = []
    for name, d_val, w_val in scenarios:
        g_val, rules, _, _, _ = infer_traffic_control(d_val, w_val)
        fired_ids = ", ".join(r["Rule ID"] for r in rules)
        records.append({
            "Scenario": name,
            "Density (veh/km)": d_val,
            "Wait Time (s)": w_val,
            "Allocated Green (s)": g_val,
            "Fired Rules": fired_ids,
        })

    df = pd.DataFrame(records)
    df.to_csv("outputs/test_results.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 4.2))
    bars = ax.bar(df["Scenario"], df["Allocated Green (s)"], color="#1d4ed8", width=0.45)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 2.0, f"{h:.1f} s", ha="center", fontweight="bold")
    ax.set_title("Prototype Test Outputs (Green Light Duration)")
    ax.set_ylabel("Allocated Green Light Time (seconds)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=15)
    plt.tight_layout()
    fig.savefig("outputs/test_outputs.png", dpi=200)
    plt.close(fig)
    return df


def main():
    st.set_page_config(page_title="Fuzzy Logic Traffic Congestion Controller", layout="wide")
    st.title("Fuzzy Logic-Based Traffic Congestion Controller")
    st.caption("Real-time adaptive green-light scheduling using Mamdani fuzzy inference and centroid defuzzification.")

    st.sidebar.header("Sensor Inputs")
    density = st.sidebar.slider("Traffic Density (veh/km)", min_value=0.0, max_value=100.0, value=55.0, step=1.0)
    wait_time = st.sidebar.slider("Waiting Time (s)", min_value=0.0, max_value=120.0, value=100.0, step=1.0)

    if st.sidebar.button("Calculate Green Time", type="primary"):
        green_time, active_rules, fuzzified, z, agg_mf = infer_traffic_control(density, wait_time)

        st.subheader("Decision Output")
        st.success(f"Recommended Green-Time Duration: {green_time:.2f} seconds")

        if green_time >= 65.0:
            st.warning("STATUS: HEAVY CONGESTION DETECTED -> LONG GREEN EXTENSION ASSIGNED")
        elif green_time <= 25.0:
            st.info("STATUS: LOW TRAFFIC VOLUME -> SHORT GREEN PHASE ASSIGNED")
        else:
            st.info("STATUS: BALANCED TRAFFIC FLOW -> MODERATE GREEN PHASE ASSIGNED")

        st.subheader("Activated Fuzzy Rules")
        if active_rules:
            st.table(pd.DataFrame(active_rules))

        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Membership Degrees")
            st.json(fuzzified)
        with col2:
            st.subheader("Aggregated Fuzzy Output & Centroid")
            fig = plot_aggregated_output(z, agg_mf, green_time, density, wait_time)
            st.pyplot(fig)
    else:
        st.info("System Status: Online. Set sensor values in the sidebar and evaluate adaptive timing.")
        if st.button("Run 5-Scenario Verification Suite"):
            df_results = run_benchmark_suite()
            st.dataframe(df_results, use_container_width=True)
            st.success("Generated outputs/test_results.csv, outputs/membership_functions.png, and outputs/test_outputs.png")


if __name__ == "__main__":
    main()