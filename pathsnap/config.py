"""All tunable constants in one place. Every value is justified in notebooks 01 (EDA) / 02 (preprocessing)."""
SEED = 42

# --- geometry -------------------------------------------------------------------------------
R_EARTH = 6371008.8          # mean Earth radius (m); per-trip equirectangular projection (D5, max err 0.72 m)

# --- candidate generation (D7, notebook 01 section 10) ----------------------------------------
CAND_DELTA_M = 20.0          # candidates = arcs within d_nearest + DELTA ...
CAND_CAP_M = 150.0           # ... but never further than CAP
DENSIFY_STEP_M = 5.0         # KD-tree densification step for the spatial index
MIN_ARC_LEN_M = 0.01         # zero-length arcs are dropped (D11)

# --- heading -----------------------------------------------------------------------------------
HEADING_LAG = 2              # heading from positions i-LAG .. i+LAG (4-s baseline at 1 Hz)
MOVING_SPEED_MS = 3.0        # below this the GPS heading is noise → treated as unknown
SPEED_CLIP_MS = 60.0         # speed feature is clipped (trips 1/13/32 have sustained >60 m/s)

# --- label alignment (D6, D9) -----------------------------------------------------------------
ALIGN_DIST_CAP_M = 200.0     # distance term is capped so far-off (parked) points can't dominate the alignment
ALIGN_HEADING_PENALTY_M = 100.0   # cost added when a moving point would be matched to an arc pointing the other way
LABEL_MAX_DIST_M = 50.0      # aligned arc farther than this → point is "unlabelled" (no trustworthy reference)
HEADING_RUN_MIN = 10         # heading-mismatch makes a point unlabelled only inside a run of >= this many points
                             # (single-point mismatches are heading noise at sharp turns, median turn 113°)

# --- time ----------------------------------------------------------------------------------------
TIME_TOL_S = 0.05            # timestamps compared with tolerance (D10)
WINDOW_S = 1.0               # temporal window (±s) for the neighbour-distance feature (±1 s best: notebook 02 §6)
GAP_S = 10.0                 # a step longer than this is a "gap"

# --- CV grouping (D12) ------------------------------------------------------------------------
CV_GROUP_MERGE = {16: 10}    # trips 10 and 16 drive the same roads → same CV group
