# Active Elastic Fluid Solver

A 2D continuum solver for an active nematic fluid coupled to an elastic membrane via the Immersed Boundary Method (IBM). The fluid obeys a Stokes-like equation driven by nematic activity, and the membrane evolves under stretch, bending, and pinning forces.

---

## File Overview

| File | Role |
|---|---|
| `solverElasticActiveFluid.py` | Main solver: time-integration of velocity, nematic order, and membrane position |
| `solver_fields.py` | Membrane geometry initializer (`membraneField`) |
| `basic_tools.py` | Numerical primitives: derivatives, Laplacian, IBM delta functions, fluid diagnostics |
| `library_visualization.py` | Post-processing and movie generation |
| `plot_defect.py` | Nematic streamline plotting and topological defect detection |
| `solver_parameters.json` | All user-facing simulation parameters |

---

## `solver_parameters.json` — Parameter Reference

### Fluid Parameters

| Parameter | Type | Description |
|---|---|---|
| `DYN_VISCOSITY` | float | Dynamic (shear) viscosity `μ`. Scales the Laplacian damping of the velocity field. |
| `BLK_VISCOSITY` | float | Bulk viscosity `β`. When set to `0`, the solver enforces **incompressible flow** by activating the Fourier-space pressure projector. Any nonzero value switches to a compressible regime and disables the projector. |
| `FLUID_FRICTION` | float | Linear (Darcy) friction `Γ`. Adds a `−Γu` drag term; useful for quasi-2D or substrate-friction models. Set to `0` for no friction. |
| `ACTIVITY` | float | Magnitude of active stress `α` (stored as `−ACTIVITY` internally). Drives flow via gradients of the nematic tensor. Positive values correspond to **contractile** activity in the sign convention used here. |

### Membrane Elasticity

| Parameter | Type | Description |
|---|---|---|
| `STRETCH_STIFFNESS` | float | Tensile (stretch) stiffness `Y`. Penalizes deviation of arc-length from the rest length `ds = SPACE_STEP/2`. |
| `BENDING_STIFFNESS` | float | Bending rigidity `B`. Penalizes curvature deviation from the spontaneous curvature `κ₀`. Set to `0` to disable bending forces entirely. |
| `PINNING_STIFFNESS` | float | Harmonic pinning stiffness `G`. Tethers each membrane node to its **initial position** with a restoring force `−G(X − X₀)`. Set to `0` for a freely floating membrane. |

### Nematic Parameters

| Parameter | Type | Description |
|---|---|---|
| `FLOW_ALIGNMENT` | float | Flow-alignment parameter `λ`. Controls co-rotation vs. co-deformation of the director with the flow. `λ = 1` is the tumbling/flow-aligning limit. |
| `ROT_VISCOSITY` | float | Rotational viscosity. Divides all nematic relaxation rates (`NEM_ENERGY`, `FRANK_CONST`, `ANCHORING`). Effectively sets the nematic timescale. |
| `NEM_ENERGY` | float | Landau–de Gennes bulk free energy coefficient `C = NEM_ENERGY / ROT_VISCOSITY`. Drives the order parameter toward its equilibrium value. |
| `FRANK_CONST` | float | Frank elastic constant `κ = FRANK_CONST / ROT_VISCOSITY`. Penalizes spatial gradients in the director field (one-constant approximation). |
| `ANCHORING` | float | Surface anchoring strength `W = ANCHORING / ROT_VISCOSITY`. Couples the nematic at the membrane interface to the preferred orientation set by `ALIGNMENT`. |
| `ALIGNMENT` | int | `+1` for **parallel** (tangential) anchoring at the membrane; `−1` for **perpendicular** (homeotropic) anchoring. |
| `S_INITIAL` | float | Initial scalar order parameter magnitude. Values `< 0.5` initialize an **isotropic** state (random director); values `≥ 0.5` initialize a **weakly aligned** state (small perturbations around `θ = 0`). |
| `ACTIVE_IN` | float | `1.0` → active fluid is **inside** the membrane; `0.0` → active fluid is **outside**. Internally sets `Λ = 2·ACTIVE_IN − 1`. A value of `Λ = 0.0` (`ACTIVE_IN=0.5`) also disables the mask (uniform activity everywhere). |
| `ISO` | bool | `true` → use the linearized (isotropic) free energy `A = C`; `false` → use the full nonlinear free energy `A = C(S²M − (2M−1))`. |

### Domain and Time

| Parameter | Type | Description |
|---|---|---|
| `X_SIZE` | float | Domain length in x. |
| `Y_SIZE` | float | Domain length in y. |
| `MAX_TIME` | float | Total simulation time. |
| `SPACE_STEP` | float | Eulerian grid spacing `dh`. The membrane arc-length spacing is fixed at `ds = dh / 2`. |
| `TIME_STEP` | float | Time step `dt` for the predictor-corrector integrator. |
| `SKIP` | int | Save a snapshot every `SKIP` time steps. |
| `BATCH` | int | Number of time steps per output HDF5 file. A new file is opened every `BATCH` steps. |

### Membrane Geometry (`MEMBRANE` sub-object)

| Parameter | Type | Description |
|---|---|---|
| `count` | int | `1` for a single membrane; `2` for two concentric membranes (inner + outer). |
| `shape` | str | `"circle"` or `"line"`. A `"line"` spans the full domain width with periodic ends. |
| `radius` | float | Radius of the inner membrane (or the single membrane). For `count=2` with `shape="circle"`, the outer membrane radius is `radius + width`. |
| `width` | float | Gap between inner and outer membranes (`count=2`). For `count=1`, unused. |
| `stiff` | str | Which membrane segment(s) carry pinning stiffness `G`: `"all"`, `"one"` (outer only), or `"two"` (inner only). **Must be `"all"` when `count=1`**. |
| `slack` | str | Adds excess arc-length (slack) to `"one"` (outer), `"two"` (inner), or `"none"`. Controls undulation amplitude `a_q`. **`"one"` is not valid for `shape="line"`**. |
| `ratio` | float | Fractional slack `ν`. The excess arc-length relative to the rest circumference/length. Only active when `slack ≠ "none"`. |
| `mode` | int | Wavenumber `q` of the initial undulation mode seeded by the slack. |
| `off_center` | float | Percentage offset (0–100) of the inner membrane center relative to the gap width. Only applies to `count=2, shape="circle"`. |

### I/O and Preloading

| Parameter | Type | Description |
|---|---|---|
| `preload` | bool | `true` → restart from a previously saved snapshot. Loads velocity, nematic, membrane, and mask from HDF5 files. |
| `where` | str | Base path (without file index) of the restart files. Used only when `preload=true`. |
| `which` | int | File batch index to load from. Used only when `preload=true`. |
| `when` | int | Snapshot index within the batch (e.g., `-1` for the last frame). Used only when `preload=true`. |
| `save` | bool | `true` → write HDF5 output files and a `__parameters.json` copy. `false` → dry run with no output. |
| `folder` | str | Directory where output files are written. The solver appends a `YYMMDD_HHMMSS` timestamp automatically. |

---

## Running the Solver (`solverElasticActiveFluid.py`)

1. Edit `solver_parameters.json` with your desired parameters.
2. Place `solver_parameters.json` in the working directory.
3. Run:

```bash
python solverElasticActiveFluid.py
```

Output files are written to `folder/YYMMDD_HHMMSS/` and follow the naming pattern:

```
YYMMDD_HHMMSS_{N}__velocity.h5
YYMMDD_HHMMSS_{N}__nematic.h5
YYMMDD_HHMMSS_{N}__membrane.h5
YYMMDD_HHMMSS_{N}__mask.h5
YYMMDD_HHMMSS__parameters.json
```

where `{N}` is the batch index (1, 2, 3, …).

---

## Generating Movies (`library_visualization.py`)

Edit the configuration block near the bottom of `library_visualization.py`:

```python
path = '/path/to/data/folder'   # directory containing output files
tag  = 'YYMMDD_HHMMSS'         # timestamp tag of the run to visualize
```

Then configure the visualization options:

```python
file_list = [1]            # which batch file(s) to load (list of ints)
t_sample  = [0.0, 1.0, 10] # [start_fraction, end_fraction, temporal_skip]
skip      = 1              # spatial downsampling factor
```

And toggle what to plot:

```python
ap = {
    'fps': 25,
    'plot_velocity': True,   # vorticity field + streamlines
    'plot_nematics': True,   # elastic energy + nematic director + defects
    'plot_membrane': True,   # membrane particle positions
    'remove_frames': False   # delete PNG frames after stitching
}
```

Run with:

```bash
python library_visualization.py
```

This creates a `YYMMDD_HHMMSS_media/` directory for PNG frames and outputs:

- `YYMMDD_HHMMSS__velocity.mov` — vorticity field with streamlines and vortex cores
- `YYMMDD_HHMMSS__nematic.mov` — elastic energy density with director lines and ±½ defect markers

### Color Map Defaults

| Field | Color Map | Quantity Shown |
|---|---|---|
| Velocity | `coolwarm_r` | Vorticity `(∂v/∂x − ∂u/∂y)/2` |
| Nematic | `GnBu_r` | Frank elastic energy density |

---

## Known Hard-Coded Caveats

### Solver

- **Membrane arc-length spacing** is always fixed at `ds = SPACE_STEP / 2`. There is no independent parameter for this.
- **IBM stencil** uses a fixed 4-point regularized delta function. The stencil width is always 4 grid cells; this cannot be changed without modifying `discrete_delta` in `basic_tools.py`.
- **Predictor-corrector scheme** is always a 2nd-order Adams-Bashforth/trapezoidal method. The integrator order is not configurable.
- **Spatial derivatives** use a staggered (MAC) grid. Velocity components `u` and `v` live on face-centered points; `Q` and the mask live at cell centers. Mixing these conventions without using `interpolate_grid` will produce errors.
- **Fourier projector** is only activated when `BLK_VISCOSITY = 0`. Any nonzero bulk viscosity silently disables incompressibility enforcement.
- **`stiff="all"` is mandatory when `count=1`**. Using any other value will raise a `ValueError` at initialization.
- **`slack="one"` is not valid for `shape="line"`**. This combination raises a `ValueError`.
- **Nematic initialization noise** uses a hardcoded perturbation of `0.02 * seed` on the order parameter and `0.2 * (2*seed − 1)` on the angle in the aligned branch. These are not exposed as parameters.
- **Active mask normalization**: the mask is always linearly rescaled to `[0, 1]` via `(mask − min) / (max − min)`. If the membrane configuration produces a nearly uniform divergence field, this can amplify numerical noise.

### Visualization

- **FFmpeg is called via `os.system`** with hardcoded output settings: `1920px` width, `libx264` codec, `yuv420p` pixel format, CRF 18, slow preset. FFmpeg must be installed and on `PATH`.
- **Figure DPI** is hardcoded: `100` for velocity plots, `600` for nematic plots.
- **Colormap range** is computed from the global min/max of the loaded time window. Comparing movies from different runs requires manual normalization.
- **Vorticity scaling** applies a factor of `0.5` to `(∂v/∂x − ∂u/∂y)` (i.e., uses half the curl). This matches the solver's strain-rate convention but differs from the standard definition.
- **`plot_membrane=True`** requires `solver_fields.membraneField` to be importable; it is used to recover `n_b`, `fore`, and `back` indices. The membrane data is always loaded from batch file index hardcoded in `file_list`, not necessarily consistent with `t_sample` for multi-file runs.
- **Frame files** are named with a leading dot (`.u_00001.png`) making them hidden on Unix systems. They are stored in `{tag}_media/` which must exist before running; the script creates it via `os.system('mkdir ...')`, which will silently fail if the path already exists.
