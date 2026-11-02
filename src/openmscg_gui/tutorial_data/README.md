# Squalane tutorial download

The tutorial uses an external ZIP containing raw inputs only:

- `structure.data`: atomistic LAMMPS structure, including masses and bonds.
- `aa.lammpstrj`: matching text trajectory with atomic forces (`fx`, `fy`, `fz`).
- `properties.txt`: named AA property time series for comparison in Validate.

Replace `download_url: "<LINK>"` in `example.json` with the published HTTPS URL.
The popup's **Download Squalane inputs** link opens that URL in the user's browser
only when clicked. Until a real URL is supplied, an inactive placeholder remains
visible. The tutorial itself is always available;
users with existing local inputs can continue.

Do not place a project JSON, saved bead rules, fitted tables or trial results in
this input archive. Those belong in a separate completed-reference download.
No large input files need to be bundled with the application.

The welcome dialog offers **New project**, **Open project**, and **Follow the
tutorial**. **Help → Squalane tutorial…** reopens the manual guide. It is a
movable, non-blocking popup: users extract the ZIP, create a project, select
files, draw all three bead rules and start every calculation themselves. Bead
steps describe CH3, CH2 and CH groups, accepting the suggested names and default
matching options. All fit, processing, protocol, search, scoring and storage
defaults stay unchanged. An optional Density objective is added only if the user
chooses to do so manually; the guide never applies one.

The guide contains instructions and Back/Next navigation, not duplicate project
or workflow buttons. Users find the controls and tabs in the main window. The
input-loading step includes Inspect inputs in the input popup; the following
step reviews its results instead of running inspection again. Completed actions
show a short completion message. Closing an editor does not close the tutorial.

The full example contains 100 C30H62 molecules, each with 92 atoms. The described
mapping produces 30 beads per molecule (3,000 beads total). These are expected
checks, not values silently assigned by the tutorial. Inspect still performs the
real data checks. Fitting needs OpenMSCG; simulation needs LAMMPS. Skipping optional
simulation/validation instructions never marks those calculations complete.

Before publishing, verify the paired structure/trajectory and property metadata,
document AA force-field provenance and data attribution, and follow the tutorial
manually. Default settings are a starting point, not a promise of identical
optimization scores, successful trials, or agreement with a separately tuned
reference model.
