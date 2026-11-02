# Workflow reference

Detailed controls, defaults, file conventions and implementation notes for
[AA2CG](../README.md), first public beta **0.1.0 beta 1** (`0.1.0b1`).
Commands below assume the application directory
(the directory containing `run_gui.py`), not this `docs` directory.

AA2CG is a self-contained desktop interface for the coarse-grained
force-field workflow in this repository. The GUI bundles its own copy of the
molecular species viewer, OpenMSCG wrapper, table postprocessor, LAMMPS generator,
per-system Optuna corrector, and multi-system merger. It does not import code
from sibling repository folders.

## Workflow

The eight consistently numbered tabs keep the existing workflow order and group the existing
backend jobs without changing their artifact directories or scientific calculations.
Project actions live in **File**, and the project name appears in the window title.
The Systems bar remains visible. Stage titles, descriptions, status and run actions
share one compact row, leaving the main area for the workspace.
Next-step guidance and navigation stay in that same header row. Navigation never
starts a job. Refine is optional; Validate still requires an actual CG simulation.

Every 3D panel has **Show simulation box** and **Make molecules whole (PBC)**
in its existing **View** menu—no extra control row. These are display-only;
they never change source trajectories, mapping rules or simulation settings.
Box edges follow the current frame's cell, including tilted cells. Whole-molecule
display is on by default; turning it off shows coordinates as stored (already
unwrapped input is not forcibly wrapped). Boundary-spanning bonds are hidden in
the split display, and mapped bead centres/radii still use whole-molecule geometry.
Single-molecule views always keep the molecule whole and disable the PBC toggle.
Their box is initially hidden but can be shown without zooming out from the molecule.
Returning to the full system restores its previous PBC/box choices. Choices are
local to each viewer for the session; both controls are unavailable without cell data.

1. **Inspect** — load AA inputs, detect species, and explore the trajectory in
   3D beside its 2D species drawing. **Load / change inputs…** opens a separate
   input dialog. A compact toolbar shows the system, frame/species counts and
   check status; **Details / checks…** opens the full diagnostics.
2. **Map** — draw custom beads or choose automatic united-atom mapping. The
   canvas and 3D overlay show the mapping, including any still-unmapped atoms.
   **Create bead rule** stays visible below the 2D canvas. Its **View** menu
   opens mapping settings and saved-rule deletion in a separate window.
   **Build mapped trajectory** generates the CG mapping and histograms.
3. **Configure fit** — browse histograms and visually edit the fitting plan.
   The generated defaults stay unchanged until edited. Automatic offers **Trim
   tails** (lower/upper histogram-weight percentiles; defaults 0.5% and 99.5%) and
   **Relative peak height** (first/last bins above a tunable percentage of the
   maximum; default 5%). For Pair, Bond and Angle interactions, drag a green
   boundary or choose Manual to adjust the bounds. Independently choose **Default
   spacing** or **Custom spacing**, and set **Order** per interaction (editor range
   2–6; original default 6). Custom spacing works with automatic bounds too.
   Editing only spacing/order preserves the exact original bounds; it does not
   silently recalculate an older plan's automatic interval.
   Field edits save when finished; dragging saves on release, per interaction and
   per system. **Reset** removes only that interaction's override and restores
   its original bounds, spacing, order and automatic recipe. Gray dashed lines always
   show the original bounds; green lines and shading show the active interval.
   Peak-height mode also shows a dotted horizontal threshold. The histogram is
   unchanged: short ticks are spacing guides, not a predicted force. Percentiles
   use histogram weight, not sample counts (especially important for pair RDFs).
   Interior bins below the threshold are not individually discarded.
   Dihedral bounds retain their original periodic domain; spacing/order can be edited.
   Histogram generation still belongs to the mapping job.
4. **Fit** — run force matching directly, with no subtabs.
   A compact interaction list shows the requested bounds, spacing, order and
   Original/Custom state beside the actual raw energy/force curves. Edit these
   settings in Configure fit. Mapping reminders and input-file checks remain
   available under **Mapping / inputs…**, leaving the workspace for results.
   Changed settings, a different method or an invalidated stage mark existing
   output as previous/unverified, never as a preview of the new settings.
   Independent fitting is the default for new systems. Coupled fitting remains
   available in the visible method selector; saved projects keep their
   selected method.
5. **Prepare tables** — prepare and review interaction tables:
   select from a searchable list on the left to display a large plot on the right.
   Energy (blue, left Y-axis) and force (orange, right Y-axis) share one X-axis;
   raw overlays are dashed. The Y scales are independent, not directly comparable.
6. **LAMMPS** — generate and smoke-test the setup. The smoke test checks
   initialization only; it does not generate a simulation trajectory.
7. **Refine (optional)** — configure property objectives and run the existing
   Optuna optimization directly, without nested tabs.
8. **Validate** — compare properties and distributions, load AA property data,
   and explore AA, mapped beads, and independent CG simulation trajectories.
   **3D visuals** places AA and independent CG side by side with one playback
   strip and shared **View** menu for species/molecule selection. AA can also
   show translucent mapped beads. Play/Pause and scrubbing use relative progress:
   50% selects the nearest halfway frame of each trajectory, not equal physical
   times or matching instantaneous conformations. Each panel retains its own
   frame/timestep label. Playback waits for both views and pauses when hidden
   or unavailable. Rotation, pan and zoom are linked in physical coordinate
   units, preserving visible size differences; **Frame view** fits both together.
   Box/PBC display options are shared, with whole-molecule display still forced
   during single-molecule viewing. Inspect/Map retain their own AA timeline.
   To skip Refine, run the generated LAMMPS setup externally and retain the
   compatible CG trajectory at the system's `output/lammps/cg.lammpstrj`.
   A current merged result or completed refinement takes priority over this
   baseline; invalidated refinement outputs are not silently selected.

**Parallel jobs** in Map settings defaults to **Auto (all available)**. Mapping,
histograms and independent fitting receive the available logical CPU count,
respecting process affinity where supported. Pools are still limited by the
number of frames/interactions; this does not make every calculation parallel.
Auto is resolved on the machine running the worker, so moving a project between
machines does not preserve an obsolete one-core default. Legacy projects saved
with `cores=1` migrate to Auto; larger saved limits remain manual. Disable Auto
to select a manual limit, including one core; this choice is saved per system.
Changing only the worker limit does not invalidate scientific results.
Coupled fitting remains one joint regression, and the LAMMPS/MPI launch command
is unchanged and separately configurable in LAMMPS or Refine.

Fitting setting changes never launch a calculation. They mark Fit and downstream
results pending (also after reopening the project); run Fit and the subsequent
stages explicitly. Spacing must be positive, within the interval, and produce no
more than 10,000 intervals; pair upper bounds cannot exceed the existing neighbor
cutoff, and angles stay in 0–180 degrees. With Default spacing, automatic range changes retain the
original spline interval count (200 if absent from an older plan), using the
backend's existing rounding and spacing calculation. Manual bounds retain their
spacing until edited; choosing Default spacing there restores the generated
spacing. Custom spacing is kept fixed when automatic bounds change. Order-only
edits do not change spacing. Neighbor cutoffs and solver defaults stay unchanged.
Exported table spacing continues to follow
spline spacing, as before. Editing outside sampled histogram regions does not
create additional data.

Overrides are saved as `fit_overrides` in each system's project record. The
generated `fit_plan.json` remains the automatic baseline; a force-matching run
writes `output/work/effective_fit_plan.json` and passes it to either the independent
or coupled backend. Successful fits also record the effective plan in
`output/work/force_match.json`, separately from the next requested settings.
Tuned automatic recipes are resolved from the histogram using
the same calculation as the preview. New plans record the original automatic
recipe; older plans keep their exact generated bounds and spacing on load/Reset,
with the current default recipe shown until explicitly edited.
Overrides for names absent from a regenerated plan remain
saved but inactive. Check retained overrides after changing the mapping.

**Combine systems** in the systems bar opens the existing multi-system merger.
It is a project-level action, separate from the single-system workflow.

### Refinement simulation protocol

The dialog's compact **Trial storage** row is saved per system and applies to future
Refine runs (not existing results):

- **Keep everything** (default): retain all trial files.
- **Keep improvement milestones**: retain the first valid scored trial,
  each new best whose objective improves by at least the chosen percentage since
  the last retained milestone (default 5%), and always the current/final best.
  Small improvements accumulate; 0% keeps every strict best-so-far improvement.
  Improvement is `(milestone score - new score) / abs(milestone score)`; at zero,
  any strict improvement qualifies. Ties are not new milestones.
- **Keep only the best**: retain full data only for the best valid scored trial.

A valid trial completed with finite scores and passed all configured property
limits. Crashes, incomplete scores and property-limit failures cannot become
milestones or the exported best. Optuna's objective and penalties are unchanged;
if no valid trial finishes, refinement reports that no valid best is available.

Cleanup is incremental, after scoring, including failed trials even before any
trial succeeds. Before removing a non-retained run folder, its `input.data`,
`correction.json`, `score.json` (including property summaries), bead signatures,
logs and `trial_record.json` (parameters, timestamps and outcome) are archived to
`optimization/trial_records/run_*/trial_####/`. Each study gets its own record
directory. Other files, including trajectories, sampled time series, structures,
tables and plots, are permanently deleted; `storage.json` records cleanup. They
cannot be reanalysed without rerunning. Original inputs, active trials,
links/junctions and trial folders not visited in the current run are excluded.
Copy/cleanup failures are reported and leave original files wherever possible.

New studies use unique `optimization/studies/study_*/` directories. Their `runs/`
and `trial_records/` follow the storage policy above. A final best result is saved
as an immutable `best/trial_####/` checkpoint. Cleanup modes move the best run
into its checkpoint instead of duplicating its trajectory; **Keep everything**
also retains its original run directory. `optimization/active_study.json` and the
study's `state.json` identify the current best; Validate and Combine resolve that
path automatically. Older `optimization/best` results remain readable.

Continuation never cleans earlier trials or checkpoints. The selected policy
applies only to new trials, so earlier best checkpoints can consume additional
space across multiple continuations. Starting a new study does not remove old
studies. No storage setting retroactively deletes historical data.

**Simulation protocol…** shares the existing Optuna-trials row in Refine; its
single-line summary shows ramp-up/production duration, timestep, temperature and
pressure. **Ramp-up** and **Production** sit side by side, with the production-only
scoring window alongside them. The dialog stores settings **per system**, without
adding a new panel or control row to the main workspace.

**Unmodified baseline first** is enabled by default. It uses one slot inside the
Explore budget (Explore 1 / Refine 0 means baseline only), with the same simulation
protocol and scoring as every candidate. Its pair-table energy AND force columns
are copied unchanged: it does not invoke the Gaussian table writer. Untick this
to recover the previous all-candidate trial budget.

**Gaussian search** defaults to **Explore → Refine**, with separate **Explore**
and **Refine** budgets (30 and 20 respectively for a new default configuration).
Explore uses the original broad Gaussian search; its optional baseline occupies
one Explore slot. Refine starts from the best valid result found across Explore
and the baseline. If Explore finds no valid candidate, Refine is skipped with a
clear message; add Explore trials using Continue rather than refining a failed
candidate. A zero Refine budget runs Explore only. Explicitly saved broad/local
modes remain available; their existing single trial-count control is retained.

Refine changes one parameter of one interaction per trial around the best valid
candidate, not around an arbitrary zero correction. It cycles through interactions
and amplitude, center and width; inactive
Gaussians receive an amplitude change instead of an ineffective center/width edit.
Initial centers use the lowest tabulated energy within the existing center range;
initial widths use the middle of the existing width range. **Step %** (default 10%)
limits the change to that percentage of the original parameter range; the original
global limits still apply. Optuna proposes relative steps, not full parameter vectors.

Only a strictly better, finite score with passing property limits moves the
reference. Worse, crashed or invalid trials leave it unchanged. Until a valid trial
exists, proposals remain around the unmodified starting vector. Each candidate is
rebuilt from the original source tables, not by stacking corrections on previous
trials. Uncorrected interactions remain byte-for-byte unchanged in incremental
mode. The actual full Gaussian vector is saved in `correction.json`; relative
Optuna steps, the changed interaction/parameter and reference trial are recorded
in trial metadata. This is a local search, not a guarantee of finding the global
optimum or preventing every crash. Broad search and its original ranges remain
available. No cutoff, short-range core or fitted table is automatically repaired.

**Continue…**, beside the protocol/results buttons, opens one compact row:
**Add Explore** and **Add Refine**. These counts are additional trials, not new
totals. Saved Optuna parameter distributions, scores, failure states and Gaussian
vectors are restored before new trials are appended. Trial numbers never restart;
an interrupted attempt keeps its number and is represented as failed if it did
not reach a durable checkpoint. The baseline is not repeated. Add Explore runs
first if both counts are nonzero; Refine then uses the best across all earlier
and newly added trials. Worse results never replace a better historical best.

Trial checkpoints are plain JSON, loaded using Optuna's public serialization API,
not executable pickle files. An OS-owned lock prevents simultaneous optimizers
for the same system. Checkpoints survive process termination; reconstruction uses
the saved history with a continuation-specific sampler seed, not a promise of
bit-for-bit equivalence to one uninterrupted run. Source setup snapshots and
content fingerprints prevent continuation after changes to tables, structure,
masses, protocol, objectives, Gaussian limits or optional structural references.
Changing budgets or local step size is allowed; incompatible scientific changes
require a new study and do not alter earlier outputs.

Older JSON-only runs can be imported on their first Continue if a complete valid
best result, compatible input/settings, and verifiable tables remain available.
Available trial records are imported without moving or rewriting legacy files;
missing trial numbers are reserved as failed. Legacy structural-loss runs lack
enough reference provenance and require a new study. A failed compatibility check
never launches LAMMPS or overwrites the old run.

**Trial results…**, next to the protocol button, opens a compact trial list and
log viewer. It includes archived/cleaned trials across saved studies, distinguishes
baseline, Explore and Refine candidates, and reports property-limit rejections separately from
LAMMPS crashes. Cutoff errors identify the pair using the structure's type labels
where possible, while preserving LAMMPS's original message. A pair cutoff error
is not interpreted as a bond error or an automatic diagnosis of weak repulsion.
Only small records and a bounded log tail are read; trajectories are not loaded.

By default, **Customize simulation protocol** is off: each trial copies the
source `output/lammps/input.data` unchanged, including older or hand-edited setups.
The current generator defaults remain 500,000 production steps at 1 fs (500 ps),
298.15 K and 1 atm, with no separate equilibration. The summary reads a supported
source script rather than assuming those defaults apply to an existing project.

Enable customization to edit production steps, timestep, optional ramp-up and NPT
settling steps, temperature and pressure. **Enable ramp-up** is off by default;
when first enabled it suggests 23,335 total steps (about 10 ps at the default
1 fs production timestep). All simulation lengths are entered in steps; changing
the timestep changes their displayed elapsed time, not the requested step counts.
Three approximately equal-duration NVT stages split the ramp steps in a 4:2:1
ratio and use quarter, half and full production timestep at the production target
temperature. The box remains fixed during ramp-up; this is not a heating ramp.
The total ramp step count is preserved exactly. Older time-based ramp protocols
remain readable and unchanged unless customized. Optional **Equilibration** then adds
NPT settling at the final timestep and requested pressure, preserving the existing
settling behavior. Neither startup phase records property samples or trajectory
frames. The step counter resets to zero before production, without recreating
velocities. Production duration, output cadence and scoring windows are unchanged.
**Use source input** removes the simulation override without changing property
objectives, search or storage settings. Ramp-up helps startup stability but cannot
make an intrinsically unstable potential physically valid.

The same dialog exposes **Discard initial %** separately for each active property.
It edits the existing `skip_fraction`: samples are discarded from production only,
without adding simulation time. These settings remain accessible in **Configure
selected…** and appear in the property table's summary. If the target is blank,
the first *retained* production sample remains the reference. State-point edits
do not silently change property targets.

Overrides apply only to copied optimization trials, never to the source LAMMPS
setup or a completed trial. Sampling/averaging (1, 99, 100), damping, velocity seed
and trajectory-output frequency are preserved. Unsupported custom scripts can
still be inherited unchanged; overrides fail with a clear message rather than
rewriting unknown simulation commands. A supported customized protocol requires
at least two retained property records, which is a basic validity check, **not a
convergence test**. Increasing timestep is not a substitute for longer sampling.

Changing the protocol, search or scoring windows marks Refine, Validate and combined
results pending; it does not invalidate mapping, force matching, prepared tables
or the base LAMMPS setup. Each trial's `correction.json` records the requested
overrides and the effective recognized protocol alongside the property settings.
No simulation is launched by opening or saving the dialog.

### Interactive table processing

In **Prepare tables**, choose **Processing preview** to edit the selected interaction
without overwriting exported tables. Dashed curves are raw energy/force; solid
curves are the processing preview on the same two Y-axes. Gray shading marks
trimmed samples and green shading marks extrapolation beyond the retained range.
The preview reports the existing acceptance checks, not a simulation-stability guarantee.

**Automatic** retains the original ordered search: trim 2%, 5%, 10%, 15%, 20%,
25% from each end; smoothing windows 5, 7, 9; endpoint fit points 2, 3, 5, 7, 15,
21. The first accepted candidate is selected, or the last rejected candidate is
shown. The disabled fields show the recipe selected by that search.
**Custom** exposes the energy smoothing window, trimming percentage per end,
left/right extension targets and endpoint-fit point count. Force smoothing,
damping, polynomial orders, extension spacing and acceptance heuristics remain
unchanged. Targets inside retained data do not shorten it; use trimming for that.
Dihedrals expose smoothing only and retain their existing periodic domain.

Changes preview after a short debounce and save when a field is finished, per
interaction and per system. Invalid numeric settings restore the saved recipe;
valid recipes that fail the existing checks remain visibly rejected. **Reset**
removes only that interaction's custom recipe and restores the automatic search.
No force matching, LAMMPS or table export is launched by editing. Click **Prepare
tables** explicitly to export with the same calculation used by the preview.
Rejected candidates are not written to the accepted-table directory.

Recipes are stored in each system's `table_overrides` record. Editing them marks
table processing and downstream results pending, leaving the force match intact.
Overrides whose table names are absent remain saved but inactive. Original raw
tables are never modified; exported results remain available under the other
source choices. Successful processing records the selected recipes/checks in
`output/processed_tables/processing_settings.json`. Older projects retain the
original automatic behavior without migration or a mandatory custom setting.

Long-running scientific jobs retain their application-modal progress/log window
and cancellation controls. Trajectory parsing and frame preparation run in
background threads; opening a 3D view never launches a scientific job.

### Molecule Focus

Inspect and Map give their 2D and 3D canvases equal initial width; the divider
remains adjustable. The 2D canvas uses a compact **Species** menu and a **View**
menu for framing and species/mapping details; mapping status stays visible.
All 3D canvases put their configuration in a **View ▾** menu beside playback.
Plot canvases use a text-only **Plot tools** menu below the plot for navigation,
pan/zoom, settings and export; the active pan/zoom mode remains visible.

Every 3D view offers **Whole system**, **Selected species**, and **Single
molecule** scopes in **View**. Choose a species and then a molecule using
**Previous**, **Next**, **Random molecule**, or **Go to molecule…**;
double-clicking a visible molecule also focuses it. The same instance stays
selected during playback. **View** also contains representation, dim-surroundings, follow-molecule,
and camera-framing controls. Orbit with left drag, pan with right/middle drag,
and zoom with the wheel.

Bonded molecules are unwrapped across periodic boundaries before centering.
The mapping overlay uses the same AA frame for atoms and beads, with geometric
bead centres and translucent visual envelopes. These preview radii and centres
do not change the backend's mapping weights or force-field parameters.
Unmapped atoms remain visible in the beads representation.

In Validate, AA and independent CG trajectories share playback by relative
progress, with separate frame/timestep labels. Their instantaneous configurations
are not synchronized physical counterparts. Species and molecule focus are shared
where the generated CG topology matches the current mapping. Inspect and Map
retain their own AA timeline.

The bundled GPU viewer uses ModernGL/OpenGL 3.3. LAMMPS text dumps are byte-indexed
on demand with a three-frame cache per reader; views share completed indices.
Only opened views load data, old source changes cancel indexing, and hidden
views pause playback. Atom IDs, rather than dump row order, determine identity.

## Installation

The provided installation recipes use **Conda** to provision dependencies.
Conda is not required or checked at runtime: an existing Python installation or
virtual environment with compatible dependencies can run `python run_gui.py`
as before, on Windows or Linux. Nothing is installed, created or updated automatically.

**Existing Linux Python 3.7 + OpenMSCG environment:** keep it. Activate it and
run the updated source with `python run_gui.py` from this directory. The 3D
viewer no longer uses Python 3.10-only dataclass slots. Existing 3.7 environments
need older compatible dependencies (for example Matplotlib 3.5.x and a
Python-3.7-compatible PySide6 release), not the newest releases. TOML reading on
Python below 3.11 uses `tomli` when installed. These compatibility changes have
not been execution-tested on Python 3.7; they are not a full environment certification.

For a **fresh Linux x86-64 environment**:

```bash
conda env create -f environment-linux.yml
conda activate openmscg-workbench
python run_gui.py
```

The fresh Linux specification uses Python 3.8: the
[OpenMSCG channel publishes 0.9.0 builds for both 3.7 and 3.8](https://anaconda.org/channels/vothgroup/packages/openmscg/files).
It does not require upgrading an existing 3.7 setup.

For a **fresh Windows GUI-only environment**:

```powershell
conda env create -f environment-windows.yml
conda activate openmscg-workbench-preview
python run_gui.py
```

The Windows specification intentionally excludes OpenMSCG. The YAML files are
Conda solver specifications, not tested lockfiles; no environment solve or
installation was performed while adding them. Do not apply the fresh Linux file
to a working 3.7 environment. They contain no pip/PyPI dependency step.

`pyproject.toml` retains local packaging metadata, but no longer asks pip to
install runtime dependencies or OpenMSCG. The Conda environment files declare
the installation dependencies; the application does not restrict how existing
dependencies were installed. Running from source requires no package installation.

LAMMPS and an MPI launcher
remain external programs; set their full command in **LAMMPS** or **Refine**. The
default is:

```text
mpiexec -np 16 lmp -in input.data
```

## Using the application

### Welcome and guided introduction

The welcome dialog opens on startup unless **Show welcome on startup** is
unchecked. Its three choices are **New project**, **Open project**, and **Follow
the tutorial**. **Help → Welcome…**, **Help → Guided introduction…** and
**Help → Squalane tutorial…** remain available during normal use.

The movable guide uses the current system's real stage states. Next becomes
available after a required stage succeeds; Configure fit is a review step and
Refine can be skipped. Back revisits earlier instructions. The guide contains no
duplicate project or workflow buttons and does not switch tabs for the user.
Each step names the actual GUI control to use. Completed actions show completion
text instead of asking the user to run them again. Input loading and inspection
are one step; the next step reviews the inspection results.
Selecting a different system or opening
a different project returns the guide to project/system confirmation. Exit guide
or the window's close button stops it. The guide remains readable beside manual
input/bead dialogs; navigation waits until those dialogs close. It stays owned by
the main window, so closing an editor cannot close the guide. It yields to
file pickers, errors and the modal job monitor, then returns afterwards. No
permanent tutorial panel is added to the main layout.

The general introduction works with user inputs. The Squalane tutorial starts
with an external download link defined by `download_url` in
`src/openmscg_gui/tutorial_data/example.json`. Replace `<LINK>` with the published
URL; the adjacent README documents the raw input ZIP. The tutorial is available
even before that URL is configured, with an inactive download-link placeholder.
Clicking a configured link opens the browser; the app never downloads or extracts
the archive automatically. Users create their own blank project, load the raw
structure/trajectory and manually create CH3, CH2 and CH bead rules. No mapping,
inputs, objectives or settings are assigned by the guide. Keep suggested labels
and all existing defaults; optionally add Density by hand before refinement.
The described mapping yields 30 beads per Squalane molecule, not a saved mapping
that the app injects. Optional simulation/validation instructions may be read and
skipped without claiming a computed result. Startup preferences remain in Qt's
user settings, independently of scientific project settings.

### Standard workflow

1. Create a project in an empty or dedicated folder.
2. In **Inspect**, select the atomistic LAMMPS `structure.data` and trajectory. The
   data file must contain `Atoms`, `Masses`, and `Bonds`; molecule IDs are used
   when present, otherwise bonded connected components are used.
3. Select **Inspect inputs**. Review detected species and the trajectory checks.
   If no data file exists, the GUI offers an explicit connectivity-inference
   fallback and saves the inferred structure for inspection.
4. In **Map**, create bead rules using the focused atom-selection editor, or
   choose automatic united-atom mapping in **Mapping settings and saved rules**.
   Interaction levels are cumulative: Pair, Bond, Angle, Dihedral.
5. Build the mapped trajectory, review **Configure fit**, then run **Fit force
   field** in **Fit** and **Prepare tables** in its separate tab.
6. In **LAMMPS**, generate the setup and optionally run its zero-step smoke test.
   Then optionally configure property objectives and run optimization in **Refine**,
   or simulate the generated setup externally before validation.
7. Use **Validate** for the quantitative comparison and **3D visuals** for
   molecule inspection. Load AA property statistics to compare matching
   quantities beside the CG result.
8. Add more systems as needed. **Combine systems** accepts at least two
   optimized systems and retains the existing optional global optimization.

The data file is authoritative for atom IDs, molecule membership, and bonds.
RDKit infers bond orders only to recover canonical SMILES and UA chemical types;
the UI marks a species `connectivity-only` if that inference is uncertain. The
generated OpenMSCG map explicitly references the original atom IDs, so mixed
species and shuffled dump rows do not change connectivity. The trajectory must
contain a contiguous `id` column matching the data file and the same atom count.
The fallback is inherently less reliable than supplied bonding information. It
currently expects a LAMMPS-style dump containing coordinates and either
`element`, `el`, or `mass`; other ab-initio formats must first be converted with
explicit attention to force and unit conventions.

Before force matching, the generated `structure.data` is checked against its
bond coordinates using minimum-image distances. A mismatched topology now stops
Stage 2 immediately instead of reaching LAMMPS or producing failed Optuna trials.

In **Refine**, use **Add property** to build the objective from columns written to
`simulation_data.out`. New systems start with an empty objective. Target and
weight are visible in the active-property table; double-click a row to configure
separate tail, drift, noise, and slope tolerances, the skipped sampling fraction,
hard minimum/maximum limits, maximum absolute drift, and the hard-limit penalty.
A blank target means that the first sampled value is the reference. The complete
objective is stored per system and printed to the workflow log.

## Project data

Each project has a portable JSON manifest:

```text
openmscg-project.json
systems/
  system-<id>/
    workflow-snapshot.json
    output/
    optimization/
    validation/
      metrics.json
merge/
  combined_forcefield/
  runs/
  best/
```

Removing a system from the GUI removes only its project record. Existing output
files are deliberately preserved.

### Sharing a project

**File → Export project…** saves current GUI settings and creates a portable ZIP,
without deleting or moving working data. Choose **Best result** (default),
**Inputs + settings**, or **Everything**. The popup reports selected bytes before
compression; planning and ZIP creation run in a cancellable background thread.
Existing archives are never overwritten. A failed or cancelled export removes
its temporary output, not project files.

Inputs referenced outside the project are copied into archive-local `inputs/`
folders, with shared files bundled once. Missing inputs must be relinked before
export. On opening the extracted project, relative input paths resolve against
its own folder; saving retains relative paths for project-owned inputs. Nothing
starts automatically. Check the LAMMPS command before running on another machine.

**Best result** preserves full data only for each system's current valid best
trial and the merged best, while keeping small score/settings records and saved
study history. Other trial trajectories, time series and tables are omitted.
Scientific source snapshots and fingerprints remain unchanged; the existing
compatibility checks still apply to continuation. **Inputs + settings** resets
generated stages to pending but retains mapping rules and configuration.
**Everything** includes all retained trials for the current systems. Presets
cover referenced inputs and managed system/merge folders, not unrelated root
files, other archives, environments, caches or locks. Links/junctions encountered
in selected data are rejected, not followed. Custom scripts may need additional
external dependencies beyond the application's linked inputs.

Extract the ZIP into a separate folder and open `openmscg-project.json` normally.
`EXPORT-README.md` explains the preset; `export-manifest.json` records SHA-256
checksums and available software versions. Original paths can remain in scientific
snapshots and logs: export makes active input/validation links portable, but is
not an anonymization tool. LAMMPS executables are never invoked to collect versions.

## Source layout

```text
src/openmscg_gui/
  app.py, main_window.py, pages.py    Qt application and workflow UI
  workflow_views.py                  distributions and stage grouping
  trajectory_view.py                 asynchronous 3D UI and molecule focus
  trajectory_scene.py                read-only species/mapping projections
  trajectory_gpu.py                  bundled ModernGL renderer and dump reader
  project.py                         persistent project/system model
  process_runner.py, worker.py       background stage orchestration
  molecule_drawer/                   bundled Qt/RDKit AA/UA species viewer
  backend/
    openmscg_wrap/                   topology detection, mapping, and force matching
    table_processing.py              interaction-table postprocessing
    lammps_generation.py             LAMMPS setup generation
    corrector/                        per-system Optuna correction
    merger/                           multi-system merge and optimization
```


## GUI checks

With GUI dependencies installed and `src` on `PYTHONPATH`:

```powershell
python -m unittest discover -s tests -v
python tests/review_workflow_ui.py
```

The second command uses synthetic water fixtures and a hidden Qt window to check
OpenGL rendering, frame/focus synchronization, workflow navigation, and compact layout.
It prints its screenshot directory. Neither command runs a real LAMMPS simulation.
