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
st.set_page_config(page_title="Microcarrier CA (Vero) – Monte Carlo", layout="wide")

# =========================
# Constants: cell states
# =========================
UNOCCUPIED = 0
OCCUPIED = 1
NEWLY_OCCUPIED = 2
INHIBITED = 3
MULTILAYER = 4

STATE_NAMES = {
    UNOCCUPIED: "UNOCCUPIED",
    OCCUPIED: "OCCUPIED",
    NEWLY_OCCUPIED: "NEWLY_OCCUPIED",
    INHIBITED: "INHIBITED",
    MULTILAYER: "MULTILAYER",
}

# Colors (RGBA) for sphere as requested
STATE_COLORS = np.array([
    [0.83, 0.83, 0.83, 1.0],  # UNOCCUPIED  light grey
    [0.50, 0.00, 0.50, 1.0],  # OCCUPIED    purple
    [0.18, 0.00, 0.30, 1.0],  # NEWLY_OCCUPIED dark purple
    [0.85, 0.00, 0.00, 1.0],  # INHIBITED   red
    [0.35, 0.00, 0.00, 1.0],  # MULTILAYER  dark red
], dtype=float)


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
    """Return a matplotlib figure for a single sphere frame."""
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
    """Create GIF bytes from list of grid snapshots."""
    frames = []
    for t, grid in enumerate(grids, start=1):
        fig = render_sphere_frame(grid, f"{title_prefix} (MAX≈{cap_value:.1f}) – step {t}")
        png_bytes = fig_to_png_bytes(fig)
        frames.append(imageio.imread(png_bytes))
    out = io.BytesIO()
    imageio.mimsave(out, frames, format="GIF", duration=duration_s)
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
    """
    Returns:
      cap (float), counts_over_time (dict[str]->list), sphere_snapshots (list of grids)
    """
    cap = max(1.0, np.random.normal(max_cells_mean, max_cells_sd))
    inoc = max(0.0, np.random.normal(inoc_mean, inoc_sd))
    inoc_density = inoc / cap

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

                # Growth from OCCUPIED
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

                # Multilayer on inhibited
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

        # snapshot BEFORE NEWLY->OCC conversion so you can see NEWLY_OCCUPIED
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
st.title("Microcarrier cellular automaton (Vero) – Monte Carlo + sphere GIFs")

with st.sidebar:
    st.header("Simulation controls")

    num_experiments = st.slider("Number of Monte Carlo experiments", 10, 2000, 120, 10)
    time_steps = st.slider("Time steps", 3, 30, 10, 1)

    st.subheader("Capacity distribution (MAX cells/MC)")
    max_cells_mean = st.slider("Mean MAX cells/MC", 20.0, 400.0, 140.0, 1.0)
    max_cells_sd = st.slider("SD MAX cells/MC", 0.0, 120.0, 23.0, 1.0)

    st.subheader("Inoculation distribution (cells/MC)")
    inoc_mean = st.slider("Mean inoculation cells/MC", 0.0, 50.0, 4.004, 0.1)
    inoc_sd = st.slider("SD inoculation cells/MC", 0.0, 30.0, 3.0, 0.1)

    st.subheader("Multilayer rule")
    multilayer_threshold = st.slider("Cycles until inhibited → multilayer", 1, 10, 1, 1)

    st.subheader("Reproducibility")
    use_seed = st.checkbox("Use fixed RNG seed", value=False)
    seed = st.number_input("Seed", min_value=0, max_value=10_000_000, value=42, step=1) if use_seed else None

    run_btn = st.button("Run simulation", type="primary")


# Auto-run once on load
if "has_run" not in st.session_state:
    st.session_state.has_run = True
    run_btn = True

if run_btn:
    with st.spinner("Running Monte Carlo..."):
        runs, capacities, idx_low, idx_med, idx_high = run_monte_carlo(
            num_experiments=num_experiments,
            time_steps=time_steps,
            max_cells_mean=max_cells_mean,
            max_cells_sd=max_cells_sd,
            inoc_mean=inoc_mean,
            inoc_sd=inoc_sd,
            multilayer_threshold=multilayer_threshold,
            seed=seed,
        )

    st.success("Done.")

    # =========================
    # Plot 1: Mean ± SD
    # =========================
    states_main = ["UNOCCUPIED", "OCCUPIED", "INHIBITED", "MULTILAYER"]
    ts = np.arange(1, time_steps + 1)

    avg = {s: [] for s in states_main}
    sd = {s: [] for s in states_main}

    for s in states_main:
        for t in range(time_steps):
            vals = [counts[s][t] for _, counts, _ in runs]
            avg[s].append(np.mean(vals))
            sd[s].append(np.std(vals))

    fig1 = plt.figure(figsize=(8.5, 5.2))
    ax = fig1.add_subplot(111)
    for s in states_main:
        ax.errorbar(ts, avg[s], yerr=sd[s], marker="o", linestyle="-", label=s)
    ax.set_title(f"Monte Carlo mean ± SD (n={num_experiments})")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Sites per MC")
    ax.grid(True, linestyle="--", linewidth=0.5)
    ax.legend()
    st.pyplot(fig1)

    # =========================
    # Plots 2–4: Low/Median/High
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

    colA, colB, colC = st.columns(3)
    with colA:
        cap_low, _ = plot_single(idx_low, "LOW")
    with colB:
        cap_med, _ = plot_single(idx_med, "MEDIAN")
    with colC:
        cap_high, _ = plot_single(idx_high, "HIGH")

    # =========================
    # Sphere GIFs: low/high
    # =========================
    st.subheader("Sphere visualization (GIFs)")

    cap_low, _, grids_low = runs[idx_low]
    cap_high, _, grids_high = runs[idx_high]

    with st.spinner("Rendering LOW and HIGH sphere GIFs..."):
        low_gif_bytes = make_gif_from_grids(grids_low, "LOW capacity MC", cap_low, duration_s=0.8)
        high_gif_bytes = make_gif_from_grids(grids_high, "HIGH capacity MC", cap_high, duration_s=0.8)

    g1, g2 = st.columns(2)
    with g1:
        st.markdown(f"**LOW capacity** (MAX≈{cap_low:.1f})")
        st.image(low_gif_bytes)
        st.download_button(
            "Download LOW GIF",
            data=low_gif_bytes,
            file_name="microcarrier_LOW_capacity.gif",
            mime="image/gif",
        )

    with g2:
        st.markdown(f"**HIGH capacity** (MAX≈{cap_high:.1f})")
        st.image(high_gif_bytes)
        st.download_button(
            "Download HIGH GIF",
            data=high_gif_bytes,
            file_name="microcarrier_HIGH_capacity.gif",
            mime="image/gif",
        )

    # =========================
    # Optional: export aggregated data
    # =========================
    st.subheader("Export (optional)")
    export_rows = []
    for i, (cap, counts, _) in enumerate(runs):
        row = {"experiment": i, "MAX_capacity": cap}
        for t in range(time_steps):
            for k in ["UNOCCUPIED","OCCUPIED","NEWLY_OCCUPIED","INHIBITED","MULTILAYER","OCCUPIED_averaged"]:
                row[f"{k}_{t+1}"] = counts[k][t]
        export_rows.append(row)
    export_df = pd.DataFrame(export_rows)

    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download raw experiment table (CSV)",
        data=csv_bytes,
        file_name="mc_microcarrier_experiments.csv",
        mime="text/csv",
    )

else:
    st.info("Adjust sliders and click **Run simulation**.")