"""Linear 1.5-layer equatorial waves on a closed beta-plane basin.

Required packages: numpy, matplotlib, pillow.
Run: python equatorial_waves_WWB.py
Output: see OUTPUT_FILE (one frame per simulated day).
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


# ============================================================
# 1. Physical and numerical parameters
# ============================================================

GRAVITY = 9.81                 # m/s^2
REDUCED_GRAVITY = 0.025        # m/s^2
MEAN_DEPTH = 250.0            # m
EARTH_RADIUS = 6371000.0    # m
EARTH_ROTATION = 7.3e-5 # rad/s

BETA = 2.0 * EARTH_ROTATION / EARTH_RADIUS
WAVE_SPEED = np.sqrt(REDUCED_GRAVITY * MEAN_DEPTH)
DEFORMATION_RADIUS = np.sqrt(WAVE_SPEED / BETA)

BASIN_LENGTH = 16000000.0   # m, east-west extent
BASIN_WIDTH = 6640000.0     # m, north-south extent
NX = 240                      # number of x cells
NY = 200                      # number of y cells
DX = BASIN_LENGTH / NX
DY = BASIN_WIDTH / NY

SECONDS_PER_DAY = 86400.0
DT = 3600.0                  # s, 1 hour
SIMULATION_DAYS = 240
FPS = 12
DRAG = 1.0 / (3.0 * 365.0 * SECONDS_PER_DAY)
MU = 200.0                      # m^2/s, lateral viscosity (Fortran ah)
OUTPUT_FILE = r"D:\desktop_file\PGFDLab\equatorial_waves.gif"

# Central-Pacific Westerly Wind Burst.
OCEAN_DENSITY = 1025.0            # kg/m^3
WIND_STRESS_PEAK = 0.10           # N/m^2
WIND_DURATION = 14.0 * SECONDS_PER_DAY

STEPS_PER_DAY = int(SECONDS_PER_DAY / DT)
assert STEPS_PER_DAY * DT == SECONDS_PER_DAY
assert WAVE_SPEED * DT * np.sqrt(1.0 / DX**2 + 1.0 / DY**2) < 0.7


# ============================================================
# 2. Fortran-style Arakawa C grid with one ghost-cell layer
# ============================================================

# All three arrays have the same (NY+2, NX+2) shape, but their
# physical positions are staggered:
#
#   h[j,k] : cell center                    (1<=j<=NY, 1<=k<=NX)
#   u[j,k] : EAST face of h[j,k]            (1<=j<=NY, 0<=k<=NX)
#   v[j,k] : NORTH face of h[j,k]           (0<=j<=NY, 1<=k<=NX)
#
# u[:,0] / u[:,NX] are the west/east walls;
# v[0,:] / v[NY,:] are the south/north walls.
# Index 0 and index NY+1/NX+1 are the land/ghost-cell ring for h.

GRID_SHAPE = (NY + 2, NX + 2)

x_center = (np.arange(NX + 2) - 0.5) * DX
x_u_face = np.arange(NX + 2) * DX

y_center = (np.arange(NY + 2) - 0.5 - NY / 2.0) * DY
y_v_face = (np.arange(NY + 2) - NY / 2.0) * DY
coriolis_v = BETA * y_v_face


# Land (dry) ghost cells, matching the all-four-sides Fortran wet mask.
wet = np.zeros(GRID_SHAPE, dtype=bool)
wet[1:NY + 1, 1:NX + 1] = True


# ============================================================
# 3. Initial condition and boundary conditions
# ============================================================

x0 = BASIN_LENGTH / 2.0
zonal_width = 850_000.0
meridional_width = DEFORMATION_RADIUS
wind_shape_x = np.exp(-0.5 * ((x_u_face - x0) / zonal_width)**2)
wind_shape_y = np.exp(-0.5 * (y_center / meridional_width)**2)
wind_shape_u = wind_shape_y[:, None] * wind_shape_x[None, :]


def enforce_wall_velocities(h, u, v):
    """Fill land/ghost cells and impose wall velocities on the C grid.

    Normal velocity is explicitly zero AT the four coastal faces.
    Tangential no-slip is represented by an odd reflection across each
    wall: (u_ghost + u_first)/2 = 0 at the N/S walls and likewise for
    v at the W/E walls. The first WATER row/column is NOT fixed to 0.
    """

    # Free-surface anomaly: mirror at the wall (zero normal derivative).
    h[0, :] = h[1, :]
    h[NY + 1, :] = h[NY, :]
    h[:, 0] = h[:, 1]
    h[:, NX + 1] = h[:, NX]

    # Impermeable coast: velocity on the physical normal face is zero.
    u[:, 0] = 0.0       # west coast
    u[:, NX] = 0.0      # east coast
    v[0, :] = 0.0       # south coast
    v[NY, :] = 0.0      # north coast

    # Tangential no-slip 
    u[0, :] = -u[1, :]          # south wall
    u[NY + 1, :] = -u[NY, :]    # north wall
    v[:, 0] = -v[:, 1]          # west wall
    v[:, NX + 1] = -v[:, NX]    # east wall

    # Outside-face halo values (not involved in calculation).
    u[:, NX + 1] = -u[:, NX - 1]
    v[NY + 1, :] = -v[NY - 1, :]


def initial_state():
    """Resting ocean: h is thickness anomaly, initially zero everywhere."""
    h = np.zeros(GRID_SHAPE, dtype=np.float64)
    u = np.zeros(GRID_SHAPE, dtype=np.float64)
    v = np.zeros(GRID_SHAPE, dtype=np.float64)
    enforce_wall_velocities(h, u, v)
    return h, u, v


# ============================================================
# 4. Finite-difference tendencies
# ============================================================
def tendencies(h, u, v, time_seconds):
    """Return linear shallow-water tendencies with lateral viscosity."""
    # Each RK4 stage must independently satisfy the wall constraints.
    enforce_wall_velocities(h, u, v)

    dh = np.zeros_like(h)
    du = np.zeros_like(u)
    dv = np.zeros_like(v)

    # Continuity, on wet h centers: j=1..NY, k=1..NX.
    divergence_x = (u[1:NY + 1, 1:NX + 1]
                    - u[1:NY + 1, 0:NX]) / DX
    divergence_y = (v[1:NY + 1, 1:NX + 1]
                    - v[0:NY, 1:NX + 1]) / DY
    dh[1:NY + 1, 1:NX + 1] = -MEAN_DEPTH * (divergence_x + divergence_y)

    # Zonal momentum at INTERNAL u-faces k=1..NX-1.
    # Fortran C-grid interpolation: four surrounding v-faces,
    # with f evaluated at the corresponding v latitude.
    fv_at_u = 0.25 * (
        coriolis_v[0:NY, None]
        * (v[0:NY, 1:NX] + v[0:NY, 2:NX + 1])
        + coriolis_v[1:NY + 1, None]
        * (v[1:NY + 1, 1:NX] + v[1:NY + 1, 2:NX + 1])
    )

    # Rigid layer approximation
    pressure_x = (h[1:NY + 1, 2:NX + 1]
                  - h[1:NY + 1, 1:NX]) / DX
    
    # Simplified viscous fluxes for u, with constant layer depth.
    laplacian_u = (
        (u[1:NY + 1, 2:NX + 1] - 2.0 * u[1:NY + 1, 1:NX]
         + u[1:NY + 1, 0:NX - 1]) / DX**2
        + (u[2:NY + 2, 1:NX] - 2.0 * u[1:NY + 1, 1:NX]
           + u[0:NY, 1:NX]) / DY**2
    )


    du[1:NY + 1, 1:NX] = (
        fv_at_u - REDUCED_GRAVITY * pressure_x
        - DRAG * u[1:NY + 1, 1:NX]
        + MU * laplacian_u
    )

    # WWB
    if 0.0 < time_seconds < WIND_DURATION:
        temporal_envelope = np.sin(np.pi * time_seconds / WIND_DURATION)
        wind_acceleration = WIND_STRESS_PEAK / (OCEAN_DENSITY * MEAN_DEPTH)
        du[1:NY + 1, 1:NX] += (
            wind_acceleration * temporal_envelope
            * wind_shape_u[1:NY + 1, 1:NX]
        )

    # Meridional momentum at INTERNAL v-faces j=1..NY-1.
    u_at_v = 0.25 * (
        u[1:NY, 0:NX] + u[1:NY, 1:NX + 1]
        + u[2:NY + 1, 0:NX] + u[2:NY + 1, 1:NX + 1]
    )

    # Rigid layer approximation
    pressure_y = (h[2:NY + 1, 1:NX + 1]
                  - h[1:NY, 1:NX + 1]) / DY
    
    # Simplified viscous flux for v, with constant layer depth.
    laplacian_v = (
        (v[1:NY, 2:NX + 2] - 2.0 * v[1:NY, 1:NX + 1]
         + v[1:NY, 0:NX]) / DX**2
        + (v[2:NY + 1, 1:NX + 1] - 2.0 * v[1:NY, 1:NX + 1]
           + v[0:NY - 1, 1:NX + 1]) / DY**2
    )

    dv[1:NY, 1:NX + 1] = (
        -coriolis_v[1:NY, None] * u_at_v
        - REDUCED_GRAVITY * pressure_y
        - DRAG * v[1:NY, 1:NX + 1]
        + MU * laplacian_v
    )

    # Reflect tendency ghosts as well.
    enforce_wall_velocities(dh, du, dv)
    return dh, du, dv


# ============================================================
# 5. Fourth-order Runge-Kutta time step
# ============================================================

def rk4_step(h, u, v, time_seconds):
    """Advance (h, u, v) through one time step of length DT."""
    k1_h, k1_u, k1_v = tendencies(h, u, v, time_seconds)

    k2_h, k2_u, k2_v = tendencies(
        h + 0.5 * DT * k1_h,
        u + 0.5 * DT * k1_u,
        v + 0.5 * DT * k1_v,
        time_seconds + 0.5 * DT,
    )

    k3_h, k3_u, k3_v = tendencies(
        h + 0.5 * DT * k2_h,
        u + 0.5 * DT * k2_u,
        v + 0.5 * DT * k2_v,
        time_seconds + 0.5 * DT,
    )

    k4_h, k4_u, k4_v = tendencies(
        h + DT * k3_h,
        u + DT * k3_u,
        v + DT * k3_v,
        time_seconds + DT,
    )

    next_h = h + (DT / 6.0) * (k1_h + 2 * k2_h + 2 * k3_h + k4_h)
    next_u = u + (DT / 6.0) * (k1_u + 2 * k2_u + 2 * k3_u + k4_u)
    next_v = v + (DT / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)

    enforce_wall_velocities(next_h, next_u, next_v)
    return next_h, next_u, next_v


# ============================================================
# 6. Integrate the model, storing one SSH frame each day
# ============================================================

def run_simulation():
    h, u, v = initial_state()
    ssh_frames = []

    for day in range(SIMULATION_DAYS + 1):
        # SSH is related to thickness anomaly by eta = (g'/g) * h (rigid layer approximation).
        ssh_cm = 100.0 * (REDUCED_GRAVITY / GRAVITY) * h[1:NY + 1, 1:NX + 1]
        ssh_frames.append(ssh_cm.astype(np.float32).copy())

        if day == SIMULATION_DAYS:
            break

        for step_in_day in range(STEPS_PER_DAY):
            time_seconds = (day * STEPS_PER_DAY + step_in_day) * DT
            h, u, v = rk4_step(h, u, v, time_seconds)

        if (day + 1) % 30 == 0:
            print("Integrated day {}/{}".format(day + 1, SIMULATION_DAYS), flush=True)

    return ssh_frames


# ============================================================
# 7. Save a GIF using Matplotlib + Pillow
# ============================================================
def save_animation(ssh_frames):
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.subplots_adjust(right=0.87, bottom=0.17, top=0.88)

    image = ax.imshow(
        ssh_frames[0],
        origin="lower",
        extent=(0, BASIN_LENGTH / 1_000.0, -30, 30),
        aspect="auto",
        cmap="RdBu_r",
        vmin=-3.5,
        vmax=3.5,
        interpolation="bilinear",
    )

    ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.6)
    ax.set_xlabel("Distance east of western coast (km)")
    ax.set_ylabel("Latitude (degrees, approx.)")
    ax.set_xticks(np.arange(0, 16_001, 4_000))
    ax.set_yticks([-30, -15, 0, 15, 30])

    colorbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03)
    colorbar.set_label("Sea-surface height anomaly (cm)")

    title = ax.set_title("Equatorial 1.5-layer waves | Day 0")

    # Display sampled WIND STRESS (not ocean velocity) only while it acts.
    arrow_x_km = np.linspace((x0 - 1.6 * zonal_width) / 1_000.0,
                             (x0 + 1.6 * zonal_width) / 1_000.0, 5)
    arrow_latitude = np.array([-4.0, -2.0, 0.0, 2.0, 4.0])
    arrow_x, arrow_y = np.meshgrid(arrow_x_km, arrow_latitude)
    arrow_y_m = arrow_y / 60.0 * BASIN_WIDTH
    arrow_footprint = np.exp(
        -0.5 * ((arrow_x * 1_000.0 - x0) / zonal_width)**2
        -0.5 * (arrow_y_m / meridional_width)**2
    )

    wind_arrows = ax.quiver(
        arrow_x, arrow_y,
        np.zeros_like(arrow_footprint), np.zeros_like(arrow_footprint),
        color="black", pivot="middle", width=0.003,
        scale=2.0, scale_units="width", zorder=5,
    )
    wind_arrows.set_visible(False)

    def update(frame_index):
        image.set_data(ssh_frames[frame_index])
        title.set_text("Equatorial 1.5-layer waves | Day {}".format(frame_index))

        time_seconds = frame_index * SECONDS_PER_DAY
        wind_is_active = 0.0 < time_seconds < WIND_DURATION
        wind_arrows.set_visible(wind_is_active)
        if wind_is_active:
            envelope = np.sin(np.pi * time_seconds / WIND_DURATION)
            stress = WIND_STRESS_PEAK * envelope * arrow_footprint
            wind_arrows.set_UVC(stress, np.zeros_like(stress))

        return image, title, wind_arrows

    animation = FuncAnimation(
        fig,
        update,
        frames=len(ssh_frames),
        interval=1000.0 / FPS,
        blit=False,
    )

    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    animation.save(OUTPUT_FILE, writer=PillowWriter(fps=FPS), dpi=90)
    plt.close(fig)
    print("Saved: {}".format(OUTPUT_FILE), flush=True)


if __name__ == "__main__":
    print("Kelvin-wave speed: {:.2f} m/s".format(WAVE_SPEED))
    print("Equatorial deformation radius: {:.0f} km".format(DEFORMATION_RADIUS / 1000))
    frames = run_simulation()
    save_animation(frames)
