# Chief Delphi strategy notes for siloed 3v3 offense

Read-only research collected 2026-10-11. Forum observations are hypotheses,
not controlled performance evidence. Any implementation still needs a frozen,
paired multi-seed simulator evaluation. The ideas below exclude teammate
intent, assigned lanes, reservations, and coordination.

## Relevant discussions

- The [algorithmic-driving discussion](https://www.chiefdelphi.com/t/what-would-it-take-to-have-the-most-optimal-driving-an-algorithmic-approach-2026-rebuilt/524572)
  argues that the nearest fuel target may not be the best choice. It proposes
  considering reachable fuel quantity, remaining capacity, pickup and return
  time, opponent arrival, and the scoring window. Its teammate-awareness
  suggestions do not fit the siloed decision constraint.
- In [coaching advice](https://www.chiefdelphi.com/t/rebuilt-coaching-advice/515940),
  one poster suggests sweeping behind the Hub and checking for fuel near the
  driver-station wall. Treat this as a prompt for local occlusion-aware search,
  not as evidence that those locations are generally profitable.
- [Week 0 takeaways](https://www.chiefdelphi.com/t/week-0-2026-what-can-we-learn/514788)
  include reports of robots getting stuck or beached on fuel and of trench
  robots pushing many pieces across. These are early-season observations, not
  controlled measurements.
- [GalacTech's Week 0 observations](https://www.chiefdelphi.com/t/galactech-4926-oa-build-blog-2026/508067/39)
  mention seeing large fuel mounds near trench edges and robots pushing a mound
  across in one pass. The post notes limited visibility, and its quantity
  estimate is anecdotal.
- A [strategy discussion](https://www.chiefdelphi.com/t/rebuilt-might-be-rock-paper-scissors/511558)
  proposes that self-cycling may be less vulnerable to disruption than passing
  chains. This is a design hypothesis, not a measured result.

## Testable robot-local hypotheses

1. Rank a target by estimated collectible fuel, clipped to the robot's
   remaining capacity, divided by velocity-aware pickup-plus-return time.
   Compare this with nearest-target ranking and retain the robot's own observed
   velocity only.
2. When visible nearby fuel is depleted, try a local search pass for fuel
   pockets occluded by field geometry, including behind the Hub and along wall
   edges. Avoid fixed teammate lanes or reservations.
3. Adjust approach speed and angle when entering dense piles to reduce
   beached/stuck time. Measure acquisition count, stuck ticks, and final score.
4. Revisit the FMS-window return decision only with fresh seeds; the standalone
   FMS filter's latest holdout was negative, so forum strategy advice does not
   override that evidence.

For each candidate, keep tuning seeds separate from holdouts; preregister at
least three to five unseen seed clusters with four worlds per cluster; pair
baseline and candidate on identical scenario IDs; and report per-world and
per-seed fuel and score deltas plus sign consistency and uncertainty. Do not
promote a result driven by a single seed. Verify rule legality before testing
any behavior that relocates or transfers fuel.
