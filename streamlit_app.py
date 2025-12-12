# streamlit_app.py
# Full app: sliders → Monte Carlo CA → mean±SD plots → low/median/high plots
# → looping GIFs (low/median/high) → state-vs-capacity line plots for SELECTED steps

import io
import math
import random
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import imageio.v2 as imageio

# =========================
# Streamlit config
# =========================
st.set_page_config(page_title="Microcarrier CA – Monte Carlo", layout="wide")

# =========================
# Constants: cell states
# =========================
UNOCCUPIED = 0
OCCUPIED = 1
NEWLY_OCCUPIED = 2
INHIBITED = 3
MULTILAYER = 4

# Sphere colors (RGBA)
STATE_COLORS = np.array([
    [0.83, 0.83, 0.83, 1.0],  # UNOCCUPIED  light grey
    [0.50, 0.00, 0.50, 1.0],  # OCCUPIED    purple
    [0.18, 0.00, 0.30, 1.0],  # NEWLY_OCCUPIED dark purple
    [0.85, 0.00, 0.00, 1.0],  # INHIBITED   red
    [0.35, 0.00, 0.00, 1.0],  # MULTILAYER  dark red
], dtype=float)

# Defaults / fixed parameters (you can slider these too if you want)
TIME_STEPS = 10
MAX_CELLS_MEAN_DEFAULT = 140.0
MAX_CELLS_SD_DEFAULT = 23.0
MULTILAYER_THRESHOLD_DEFAULT = 1


# =========================
# Sphere helpers
# =========================
def map_to_sphere(U: np.ndarray, V: np.ndarray, r: float = 1.0):
    phi = U * np.pi
    theta = V * 2 * np.pi
    X = r * np.sin(phi) * np.cos(theta)
    Y = r * np.sin(phi) * np.sin(theta)
    Z = r * np.cos(phi)
    return X, Y, Z


def render_sphere_frame(grid: np.ndarray, title: str):
    fig = plt.figure(figsize=(5.2, 5.2))
    ax = fig.add_subplot(111, projection="3d")

    n = grid.shape[0]
    u = np.linspace(0, 1, n)
    v = np.linspace(0, 1, n)
    U, V = np.meshgrid(u, v)
    X, Y, Z = map_to_sphere(U, V)

    facecolors = STATE_COLORS[grid.astype(int)]
    ax.plot_surface(
        X, Y, Z,
        facecolors=facecolors,
        linewidth=0,
        antialiased=False,
        shade=False,
    )
    ax.set_title(title, pad=10)
    ax.set_axis_off()
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=22, azim=35)
    plt.tight_layout()
    return fig


def fig_to_png_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def make_gif_from_grids(grids, title_prefix: str, cap_value: float, duration_s: float = 0.8) -> bytes:
    frames = []
    for t, grid in enumerate(grids, start=1):
        fig = render_sphere_frame(grid, f"{title_prefix} (MAX≈{cap_value:.1f}) – step {t}")
        png_bytes = fig_to_png_bytes(fig)
        frames.append(imageio.imread(png_bytes))

    out = io.BytesIO()
    # loop=0 => infinite looping GIF (repeats in UI)
    imageio.mimsave(out, frames, format="GIF", duration=duration_s, loop=0)
    out.seek(0)
    return out.read()


# =========================
# CA simulation
# =========================
def run_experiment(
    time_steps: int,
    max_cells_mean: float,
    max_cells_sd: float,
    inoc_mean: float,
    inoc_sd: float,
    multilayer_threshold: int,
):
    cap = max(1.0, np.random.normal(max_cells_mean, max_cells_sd))
    inoc = max(0.0, np.random.normal(inoc_mean, inoc_sd))
    inoc_density = inoc / cap  # fraction of sites seeded on this bead

    grid_size = max(2, int(round(math.sqrt(cap))))
    grid = np.zeros((grid_size, grid_size), dtype=int)

    total_sites = grid_size**2
    n_seed = min(max(int(total_sites * inoc_density), 0), total_sites)

    if n_seed > 0:
        for pos in random.sample(range(total_sites), n_seed):
            x, y = divmod(pos, grid_size)
            grid[x, y] = OCCUPIED

    multilayer_counter = np.zeros_like(grid)
    snapshots = []

    counts = {k: [] for k in ["UNOCCUPIED","OCCUPIED","NEWLY_OCCUPIED","INHIBITED","MULTILAYER","OCCUPIED_averaged"]}

    def step(grid, counter):
        new = grid.copy()
        new_counter = counter.copy()

        for i in range(grid_size):
            for j in range(grid_size):

                if grid[i, j] == OCCUPIED:
                    neighbors = [(x % grid_size, y % grid_size)
                                 for x in range(i-1, i+2)
                                 for y in range(j-1, j+2)
                                 if (x, y) != (i, j)]
                    random.shuffle(neighbors)
                    for x, y in neighbors:
                        if new[x, y] == UNOCCUPIED:
                            new[x, y] = NEWLY_OCCUPIED
                            break
                    else:
                        new[i, j] = INHIBITED

                if new[i, j] == INHIBITED:
                    neighbors = [(x % grid_size, y % grid_size)
                                 for x in range(i-1, i+2)
                                 for y in range(j-1, j+2)
                                 if (x, y) != (i, j)]
                    if any(new[x, y] in [INHIBITED, MULTILAYER] for x, y in neighbors):
                        new_counter[i, j] += 1
                        if new_counter[i, j] >= multilayer_threshold:
                            new[i, j] = MULTILAYER
                            new_counter[i, j] = 0
                    else:
                        new_counter[i, j] = 0

        return new, new_counter

    for _ in range(time_steps):
        grid, multilayer_counter = step(grid, multilayer_counter)

        # snapshot BEFORE NEWLY->OCC so NEWLY is visible
        snapshots.append(grid.copy())

        uno = np.count_nonzero(grid == UNOCCUPIED)
        occ = np.count_nonzero(grid == OCCUPIED)
        newo = np.count_nonzero(grid == NEWLY_OCCUPIED)
        inh = np.count_nonzero(grid == INHIBITED)
        mul = np.count_nonzero(grid == MULTILAYER)
        occ_avg = occ + newo + inh + mul

        counts["UNOCCUPIED"].append(uno)
        counts["OCCUPIED"].append(occ)
        counts["NEWLY_OCCUPIED"].append(newo)
        counts["INHIBITED"].append(inh)
        counts["MULTILAYER"].append(mul)
        counts["OCCUPIED_averaged"].append(occ_avg)

        grid[grid == NEWLY_OCCUPIED] = OCCUPIED

    return cap, counts, snapshots


@st.cache_data(show_spinner=False)
def run_monte_carlo(
    num_experiments: int,
    time_steps: int,
    max_cells_mean: float,
    max_cells_sd: float,
    inoc_mean: float,
    inoc_sd: float,
    multilayer_threshold: int,
    seed: int | None,
):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    runs = [run_experiment(time_steps, max_cells_mean, max_cells_sd, inoc_mean, inoc_sd, multilayer_threshold)
            for _ in range(num_experiments)]

    capacities = np.array([cap for cap, _, _ in runs])
    sort_idx = np.argsort(capacities)

    idx_low = int(sort_idx[0])
    idx_med = int(sort_idx[len(sort_idx)//2])
    idx_high = int(sort_idx[-1])

    return runs, capacities, idx_low, idx_med, idx_high


# =========================
# UI
# =========================
st.title("Microcarrier cellular automaton (Vero) – Monte Carlo + looping sphere GIFs + capacity plots")

with st.sidebar:
    st.header("Controls (auto-runs on change)")

    num_experiments = st.slider("Number of experiments", 10, 2000, 120, 10)

    st.subheader("Inoculation (cells/MC)")
    inoc_mean = st.slider("Mean inoculation (cells/MC)", 0.0, 50.0, 4.004, 0.1)
    inoc_sd = st.slider("SD inoculation (cells/MC)", 0.0, 30.0, 3.0, 0.1)

    st.subheader("Optional (advanced)")
    with st.expander("Capacity + multilayer settings"):
        max_cells_mean = st.slider("Mean MAX cells/MC", 20.0, 400.0, MAX_CELLS_MEAN_DEFAULT, 1.0)
        max_cells_sd = st.slider("SD MAX cells/MC", 0.0, 120.0, MAX_CELLS_SD_DEFAULT, 1.0)
        multilayer_threshold = st.slider("Cycles until inhibited → multilayer", 1, 10, MULTILAYER_THRESHOLD_DEFAULT, 1)

    use_seed = st.checkbox("Use fixed seed (reproducible)", value=False)
    seed = st.number_input("Seed", min_value=0, max_value=10_000_000, value=42, step=1) if use_seed else None

# run (auto)
with st.spinner("Running Monte Carlo..."):
    runs, capacities, idx_low, idx_med, idx_high = run_monte_carlo(
        num_experiments=num_experiments,
        time_steps=TIME_STEPS,
        max_cells_mean=max_cells_mean,
        max_cells_sd=max_cells_sd,
        inoc_mean=inoc_mean,
        inoc_sd=inoc_sd,
        multilayer_threshold=multilayer_threshold,
        seed=seed,
    )

st.caption(
    f"Steps={TIME_STEPS} | MAX cells/MC ~ N({max_cells_mean:.1f}, {max_cells_sd:.1f}) | "
    f"Inoc ~ N({inoc_mean:.3f}, {inoc_sd:.3f}) | multilayer threshold={multilayer_threshold}"
)

# =========================
# Plot: mean ± SD vs time
# =========================
states_main = ["UNOCCUPIED", "OCCUPIED", "INHIBITED", "MULTILAYER"]
ts = np.arange(1, TIME_STEPS + 1)

avg = {s: [] for s in states_main}
sdv = {s: [] for s in states_main}

for s in states_main:
    for t in range(TIME_STEPS):
        vals = [counts[s][t] for _, counts, _ in runs]
        avg[s].append(np.mean(vals))
        sdv[s].append(np.std(vals))

fig1 = plt.figure(figsize=(9, 5.2))
ax = fig1.add_subplot(111)
for s in states_main:
    ax.errorbar(ts, avg[s], yerr=sdv[s], marker="o", linestyle="-", label=s)
ax.set_title(f"Monte Carlo mean ± SD (n={num_experiments})")
ax.set_xlabel("Time step")
ax.set_ylabel("Sites per MC")
ax.grid(True, linestyle="--", linewidth=0.5)
ax.legend()
st.pyplot(fig1)

# =========================
# Low / Median / High plots
# =========================
def plot_single(idx: int, label: str):
    cap, counts, _ = runs[idx]
    fig = plt.figure(figsize=(7.2, 4.6))
    ax = fig.add_subplot(111)
    for s in states_main:
        ax.plot(ts, counts[s], marker="o", linestyle="-", label=s)
    ax.set_title(f"{label} capacity MC (MAX≈{cap:.1f})")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Sites per MC")
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend()
    st.pyplot(fig)
    return cap, counts

c1, c2, c3 = st.columns(3)
with c1:
    cap_low, _ = plot_single(idx_low, "LOW")
with c2:
    cap_med, _ = plot_single(idx_med, "MEDIAN")
with c3:
    cap_high, _ = plot_single(idx_high, "HIGH")

# =========================
# Sphere GIFs: low / median / high (looping)
# =========================
st.subheader("Sphere visualization (animated looping GIFs)")

cap_low, _, grids_low = runs[idx_low]
cap_med, _, grids_med = runs[idx_med]
cap_high, _, grids_high = runs[idx_high]

with st.spinner("Rendering sphere GIFs (LOW / MEDIAN / HIGH)..."):
    low_gif = make_gif_from_grids(grids_low, "LOW capacity MC", cap_low, duration_s=0.8)
    med_gif = make_gif_from_grids(grids_med, "MEDIAN capacity MC", cap_med, duration_s=0.8)
    high_gif = make_gif_from_grids(grids_high, "HIGH capacity MC", cap_high, duration_s=0.8)

g1, g2, g3 = st.columns(3)

with g1:
    st.markdown(f"**LOW** (MAX≈{cap_low:.1f})")
    st.image(low_gif)
    st.download_button("Download LOW GIF", low_gif, "microcarrier_LOW_capacity.gif", "image/gif")

with g2:
    st.markdown(f"**MEDIAN** (MAX≈{cap_med:.1f})")
    st.image(med_gif)
    st.download_button("Download MEDIAN GIF", med_gif, "microcarrier_MEDIAN_capacity.gif", "image/gif")

with g3:
    st.markdown(f"**HIGH** (MAX≈{cap_high:.1f})")
    st.image(high_gif)
    st.download_button("Download HIGH GIF", high_gif, "microcarrier_HIGH_capacity.gif", "image/gif")

# =========================
# NEW: State counts vs capacity (selected steps)
# =========================
st.subheader("State counts vs capacity (selected steps)")

n_bins = st.slider("Capacity bins", 5, 60, 15, 1)

states_for_capacity_plot = st.multiselect(
    "States to plot vs capacity",
    ["UNOCCUPIED", "OCCUPIED", "NEWLY_OCCUPIED", "INHIBITED", "MULTILAYER", "OCCUPIED_averaged"],
    default=["OCCUPIED", "INHIBITED", "MULTILAYER"]
)

selected_steps = st.multiselect(
    "Select steps to plot",
    options=list(range(1, TIME_STEPS + 1)),
    default=[1, max(1, TIME_STEPS // 2), TIME_STEPS]
)

selected_steps = sorted(set(selected_steps))

if len(selected_steps) == 0:
    st.warning("Select at least one step to plot.")
else:
    # Flat table: one row per experiment x step
    rows = []
    for i, (cap, counts, _) in enumerate(runs):
        for t in range(TIME_STEPS):
            row = {
                "experiment": i,
                "capacity": float(cap),
                "step": t + 1,
            }
            for k in ["UNOCCUPIED","OCCUPIED","NEWLY_OCCUPIED","INHIBITED","MULTILAYER","OCCUPIED_averaged"]:
                row[k] = counts[k][t]
            rows.append(row)

    cap_step_df = pd.DataFrame(rows)

    # Bin capacities
    cap_min = float(cap_step_df["capacity"].min())
    cap_max = float(cap_step_df["capacity"].max())
    bins = np.linspace(cap_min, cap_max, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    cap_step_df["cap_bin"] = pd.cut(cap_step_df["capacity"], bins=bins, include_lowest=True)
    bin_intervals = pd.cut(bin_centers, bins=bins, include_lowest=True).cat.categories

    for state in states_for_capacity_plot:
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)

        for step in selected_steps:
            sub = cap_step_df[cap_step_df["step"] == step]
            means_by_bin = sub.groupby("cap_bin", observed=True)[state].mean()
            y = [float(means_by_bin.get(interval, np.nan)) for interval in bin_intervals]

            ax.plot(bin_centers, y, marker="o", linestyle="-", label=f"Step {step}")

        ax.set_title(f"{state}: mean count vs capacity (selected steps)")
        ax.set_xlabel("Capacity (MAX cells/MC)")
        ax.set_ylabel(f"Mean {state} sites")
        ax.grid(True, linestyle="--", linewidth=0.5)
        ax.legend(ncol=2, fontsize=9)
        st.pyplot(fig)

# =========================
# Optional export
# =========================
with st.expander("Export raw experiment table (CSV)"):
    rows = []
    for i, (cap, counts, _) in enumerate(runs):
        r = {"experiment": i, "MAX_capacity": cap}
        for t in range(TIME_STEPS):
            for k in ["UNOCCUPIED","OCCUPIED","NEWLY_OCCUPIED","INHIBITED","MULTILAYER","OCCUPIED_averaged"]:
                r[f"{k}_{t+1}"] = counts[k][t]
        rows.append(r)
    df = pd.DataFrame(rows)
    st.download_button(
        "Download CSV",
        df.to_csv(index=False).encode("utf-8"),
        "mc_microcarrier_experiments.csv",
        "text/csv",
    )