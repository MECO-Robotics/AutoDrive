# Route scratch planner input gathering

The scratch planner used to index and stack robot pose, velocity, angular
limit and angular acceleration separately for every query. It now gathers the
rows in batches, reusing the validated device indices in the device route
evaluator. It still constructs a fresh planner for each route chunk.

On an exclusively leased WX 9100 using `pickup_simple/single`, seed 71, the
old and new planner inputs were exactly equal for 1 and 64 repeated rows.
Twelve warmed alternating-order construction pairs gave median times of
0.639 ms and 0.589 ms for one row, and 1.597 ms and 0.595 ms for 64 rows.
The timing excludes goal indexing in the old caller and does not measure the
complete route evaluation or training. Other GPU work was not controlled
globally during this component comparison.

The focused fake-planner test checks multi-world, multi-slot and repeated-row
input ordering. Full route-output parity and a fresh end-to-end simulator
evaluation with a current dashboard remain required before this change is
considered validated for deployment.
