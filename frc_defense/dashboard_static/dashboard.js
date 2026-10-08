import { renderComparisons, renderTrainingChart } from "./training_rendering.js";
import { renderField } from "./field_rendering.js";
const $ = (id) => document.getElementById(id);
const field = $("field");
const fx = field.getContext("2d");
const initialRun = new URLSearchParams(location.search).get("run");
const scenarioJobStorageKey = "autodrive.activeScenarioJob";
const focusedSettingsStorageKey = "autodrive.focusedTestSettings";
const dualGpuStorageKey = "autodrive.dualGpuMode";
const defaultWorld = {
  length: 16.54,
  width: 8.07,
  alliance_zone_depth: 4.028,
  colliders: [],
};

let history = [];
let frames = [];
let loadedScenarios = [];
let world = { ...defaultWorld };
let playbackTask = "counter_defense";
let playbackTimes = [];
let playbackSimTime = 0;
let frameAlpha = 0;
let index = 0;
let playing = !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
let playbackSpeed = 1;
let playbackRobotTypes = [];
let playbackFuelCapacities = [];
let zoneLoading = false;
let activeRequest = null;
let requestRevision = 0;
let pollTimer = null;
let pollDelay = 1500;
let latestTraining = null;
let lastScenarioGeneration = null;
let trendTaskChosen = false;
let scenarioReplays = [];

async function refreshScenarioReplays() {
  const select = $("scenarioReplay");
  if (!select) return;
  try {
    const response = await fetch("/api/scenario-replays", { cache: "no-store" });
    if (!response.ok) throw new Error(`Replay list failed (${response.status})`);
    const data = await response.json();
    scenarioReplays = data.replays || [];
    select.replaceChildren(new Option("Choose a saved replay", ""),
      ...scenarioReplays.map((replay) => new Option(
        `${replay.outdated ? "⚠ Outdated · " : ""}${replay.label} · seed ${replay.seed ?? "—"}`,
        replay.simulation_id,
      )));
    const status = $("scenarioReplayStatus");
    if (status) status.textContent = scenarioReplays.length
      ? `${scenarioReplays.length} saved replay${scenarioReplays.length === 1 ? "" : "s"}. Outdated means the commit or dirty files differ from this checkout.`
      : "No saved replays yet. Generate one when you are ready.";
  } catch (error) {
    const status = $("scenarioReplayStatus");
    if (status) status.textContent = `Could not list saved replays: ${error.message}`;
  }
}

function showScenarioProgress(progress) {
  const panel = $("scenarioProgress");
  if (!panel) return;
  panel.hidden = false;
  const percent = Math.max(0, Math.min(100, Number(progress.percent) || 0));
  const progressBar = $("scenarioProgressBar");
  if (progressBar) progressBar.value = percent;
  const progressPercent = $("scenarioProgressPercent");
  if (progressPercent) progressPercent.textContent = `${percent.toFixed(1)}%`;
  const simulated = Number(progress.simulated_seconds) || 0;
  const behaviorMode = progress.behavior_mode || $("behaviorMode")?.value || "match";
  const behaviorProbe = behaviorMode !== "match";
  const duration = Number(progress.match_seconds) || (behaviorProbe ? 30 : 160);
  const progressTime = $("scenarioProgressTime");
  if (progressTime) progressTime.textContent =
    `${simulated.toFixed(1)} / ${duration.toFixed(1)} simulated seconds`;
  const status = progress.status || "waiting";
  const [probeBehavior, probeHubState] = behaviorProbe ? behaviorMode.split("_") : [];
  const playbackLabel = behaviorProbe
    ? `Focused HUB ${probeHubState}`
    : "Randomized match";
  const progressTitle = $("scenarioProgressTitle");
  if (progressTitle) progressTitle.textContent = ["completed", "ready"].includes(status)
    ? `${playbackLabel} ready`
    : status === "error"
      ? "Scenario simulation failed"
      : status === "waiting"
        ? behaviorProbe ? "Behavior probe queued for simulator" : "Match queued for simulator"
        : behaviorProbe ? "Simulating isolated behavior" : "Simulating randomized match";
  const progressState = $("scenarioProgressState");
  if (progressState) progressState.textContent = status === "waiting"
    ? "Waiting for GPU time"
    : ["completed", "ready"].includes(status)
      ? "Complete"
      : status === "error"
        ? (progress.error || "Simulation failed")
        : `Physics tick ${integer(progress.tick)} / ${integer(progress.total_ticks)}`;
}

async function waitForZonePlayback(simulationId, revision, signal, query,
                                   ensureStart = false) {
  let retryDelay = 1000;
  while (revision === requestRevision) {
    try {
      if (ensureStart) {
        const startResponse = await fetch(`/api/zone-playback?${query}`, {
          cache: "no-store", signal,
        });
        const startRecord = await startResponse.json();
        if (startResponse.status === 409 || !startResponse.ok)
          throw new Error(startRecord.error || `Scenario start failed (${startResponse.status})`);
        ensureStart = false;
      }
      const response = await fetch(
        `/api/zone-playback-progress?simulation_id=${encodeURIComponent(simulationId)}`,
        { cache: "no-store", signal },
      );
      if (!response.ok) throw new Error(`Progress request failed (${response.status})`);
      const progress = await response.json();
      showScenarioProgress(progress);
      if (progress.status === "ready") {
        const resultResponse = await fetch(
          `/api/zone-playback-result?simulation_id=${encodeURIComponent(simulationId)}`,
          { cache: "no-store", signal },
        );
        const record = await resultResponse.json();
        if (!resultResponse.ok) throw new Error(record.error || "Scenario result could not be loaded");
        return record;
      }
      if (progress.status === "error")
        throw new Error(progress.error || "Scenario simulation failed");
      if (progress.status === "cancelled")
        throw new Error("This scenario was superseded. Reconnect to the active request or choose New scenario.");
      retryDelay = 1000;
    } catch (error) {
      if (error.name === "AbortError") throw error;
      if (error instanceof TypeError || /fetch|network|progress request/i.test(error.message)) {
        ensureStart = true;
        $("scenarioProgressState").textContent = "Connection interrupted · reconnecting to this match";
        retryDelay = Math.min(5000, retryDelay * 1.5);
      } else {
        throw error;
      }
    }
    await new Promise((resolve) => setTimeout(resolve, retryDelay));
  }
  throw new DOMException("Scenario request superseded", "AbortError");
}

const integer = (value) => Number(value || 0).toLocaleString();
const finite = (value) => Number.isFinite(Number(value));
const labelRole = (task) =>
  task === "3v3"
    ? "3v3 match"
    : task === "counter_defense"
    ? "Offense"
    : task === "defense"
      ? "Defense"
      : "Training";
const describeRobotSetup = (value) => {
  if (value === "none") return "None";
  const [role, controller] = value.includes("_")
    ? value.split("_")
    : [null, value];
  const controllerLabel = controller === "nn" ? "NN" : "Deterministic";
  return role ? `${role[0].toUpperCase()}${role.slice(1)} ${controllerLabel}` : controllerLabel;
};
const metricValue = (value, digits = 2) =>
  value == null || !finite(value) ? "—" : Number(value).toFixed(digits);

function ensurePlaybackClock() {
  playbackTimes = frames.map((frame, i) =>
    finite(frame.match_elapsed)
      ? Number(frame.match_elapsed)
      : frames.length > 1
        ? (160 * i) / (frames.length - 1)
        : 0,
  );
  for (let i = 1; i < playbackTimes.length; i++) {
    playbackTimes[i] = Math.max(playbackTimes[i], playbackTimes[i - 1]);
  }
  playbackSimTime = playbackTimes[index] || 0;
}

function setPlaybackTime(seconds) {
  if (!playbackTimes.length) return;
  playbackSimTime = Math.max(
    playbackTimes[0],
    Math.min(playbackTimes.at(-1), seconds),
  );
  let low = 0;
  let high = playbackTimes.length - 1;
  while (low < high) {
    const middle = (low + high + 1) >> 1;
    if (playbackTimes[middle] <= playbackSimTime) low = middle;
    else high = middle - 1;
  }
  index = low;
  const next = Math.min(index + 1, playbackTimes.length - 1);
  const span = playbackTimes[next] - playbackTimes[index];
  frameAlpha = span > 0 ? (playbackSimTime - playbackTimes[index]) / span : 0;
}

function currentPlaybackFrame() {
  const frame = frames[index];
  if (!frame) return frame;
  const next = frames[Math.min(index + 1, frames.length - 1)];
  if (!next?.robots || !frame.robots || frameAlpha <= 0) return frame;
  const robots = frame.robots.map((robot, i) => {
    const following = next.robots[i];
    if (!robot || !following) return robot;
    const angle = Math.atan2(
      Math.sin(following[2] - robot[2]),
      Math.cos(following[2] - robot[2]),
    );
    return [
      robot[0] + (following[0] - robot[0]) * frameAlpha,
      robot[1] + (following[1] - robot[1]) * frameAlpha,
      robot[2] + angle * frameAlpha,
    ];
  });
  return { ...frame, robots, match_elapsed: playbackSimTime };
}

function zoneName(x) {
  const depth = world.alliance_zone_depth || 4.028;
  return x <= depth ? "Red" : x >= world.length - depth ? "Blue" : "Center";
}

function drawTrainingChart() {
  renderTrainingChart({
    canvas: $("chart"),
    rows: history,
    selectedMetric: $("trendMetric").value,
  });
}

function renderExposure(counts = {}) {
  const list = $("opponentExposure");
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  $("exposureEmpty").hidden = entries.length > 0;
  list.replaceChildren(
    ...entries.map(([name, count]) => {
      const item = document.createElement("li");
      const label = document.createElement("span");
      label.textContent = name.replaceAll("_", " ");
      const bar = document.createElement("span");
      bar.className = "exposure-bar";
      const fill = document.createElement("i");
      fill.style.width = `${Math.max(2, (100 * count) / entries[0][1])}%`;
      bar.append(fill);
      const value = document.createElement("strong");
      value.textContent = integer(count);
      item.append(label, bar, value);
      return item;
    }),
  );
  $("opponentSummary").textContent = entries.length
    ? `${entries.length} opponent${entries.length === 1 ? "" : "s"} · ${integer(entries.reduce((sum, [, count]) => sum + count, 0))} transitions`
    : "—";
}

function renderActions(rates = []) {
  const list = $("actionDistribution");
  const values = Array.isArray(rates) ? rates : [];
  $("actionEmpty").hidden = values.length > 0;
  list.replaceChildren(
    ...values.map((rate, action) => {
      const item = document.createElement("li");
      const label = document.createElement("span");
      label.textContent = `A${action}`;
      const bar = document.createElement("span");
      bar.className = "action-bar";
      const fill = document.createElement("i");
      fill.style.width = `${Math.max(0, Math.min(100, Number(rate) * 100))}%`;
      bar.append(fill);
      const value = document.createElement("strong");
      value.textContent = `${(Number(rate || 0) * 100).toFixed(1)}%`;
      item.append(label, bar, value);
      return item;
    }),
  );
}

function renderTraining(status) {
  latestTraining = status;
  const campaign = status.campaign || {};
  const activeTasks = status.game_runs || [];
  if (!trendTaskChosen && ["offense", "defense"].includes(status.task))
    $("trendTask").value = status.task;
  const selectedTask = $("trendTask").value || status.task;
  const selectedCampaignTask = campaign.tasks?.[selectedTask] || {};
  const latestMetrics = selectedCampaignTask.latest_generation_metrics ||
    (selectedTask === status.task ? status.latest_generation_metrics : null) || {};
  const selectedLiveStatus = selectedTask === status.task ? status : {};
  const completed = Number(status.completed_timesteps || 0);
  const requested = Number(status.requested_timesteps || 0);
  const uncapped = campaign.uncapped === true;
  const ratio = requested > 0 ? Math.min(1, completed / requested) : 0;
  const generation = status.current_generation ?? status.generation;
  const totalGenerations = uncapped ? "∞" : status.total_generations;
  const isEvaluating =
    campaign.status === "running" && status.status === "completed";
  const statusLabel = isEvaluating
    ? "Generation evaluation"
    : status.status || "Waiting";
  const connection = $("connection");
  $("state").textContent = statusLabel;
  connection?.setAttribute(
    "data-status",
    status.status === "failed" ? "error" : status.status,
  );
  $("runPhase").textContent = statusLabel;
  $("runPhase").dataset.status =
    status.status === "failed" ? "error" : status.status;
  $("runIdentity").textContent = status.run_name
    ? `${labelRole(status.task)} · ${status.run_name} · ${status.architecture || "PPO"}`
    : "No active PPO run was found.";
  $("progressLabel").textContent =
    status.algorithm === "generational"
      ? "Strategic transitions"
      : "Completed steps";
  $("progress").textContent = uncapped
    ? `${integer(completed)} transitions · ongoing`
    : `${integer(completed)} / ${integer(requested)}`;
  $("progressBar").value = ratio;
  $("progressBar").hidden = uncapped;
  $("generationSummary").textContent =
    generation == null
      ? "Generation data unavailable"
      : `${selectedTask[0].toUpperCase()}${selectedTask.slice(1)} · generation ${generation} · ` +
        `${integer(selectedCampaignTask.generations_completed ?? latestMetrics.generation ?? 0)} completed` +
        `${isEvaluating ? " · evaluating opponents" : ""}`;
  $("updated").textContent =
    status.elapsed_seconds == null
      ? "Waiting for generation update"
      : `${Math.floor(Number(status.elapsed_seconds) / 60)}m ${Math.floor(Number(status.elapsed_seconds) % 60)}s elapsed`;
  $("rate").textContent = integer(
    Math.round(
  selectedLiveStatus.strategic_transitions_per_second ||
        selectedLiveStatus.transitions_per_second ||
        latestMetrics.strategic_transitions_per_second || 0,
    ),
  );
  $("device").textContent = status.device_name || status.device || "—";
  $("backend").textContent = status.accelerator_backend || "";
  $("setup").textContent =
    `${status.num_envs ? integer(status.num_envs) + " envs · " : ""}${status.architecture || "—"}`;
  $("opp").textContent = status.opponents?.length
    ? status.opponents.join(" · ")
    : `Current opponent: ${status.opponent || "—"}`;
  const campaignTasks = Object.entries(campaign.tasks || {});
  $("trainingEpochs").textContent = campaignTasks.length
    ? campaignTasks.map(([task, info]) =>
        `${task[0].toUpperCase()}${task.slice(1)} · ${integer(info.generations_completed ?? 0)} complete`)
        .join(" · ")
    : activeTasks.length
      ? activeTasks.map((run) =>
          `${labelRole(run.task)} ${run.current_generation ?? run.generation ?? 0}/${run.total_generations ?? "—"}`)
          .join(" · ")
      : generation == null ? "—" : `${generation} / ${totalGenerations ?? "—"}`;
  $("trainingRun").textContent = campaignTasks.length
    ? campaignTasks.map(([task, info]) =>
        `${task[0].toUpperCase()}${task.slice(1)} · ${info.status || "—"} · ` +
        `generation ${info.generation ?? "—"} · ${integer(info.complete_matches_observed)} complete matches`)
        .join(" | ")
    : activeTasks.length
      ? activeTasks.map((run) =>
          `${labelRole(run.task)} · ${run.status} · ${integer(run.completed_timesteps)} transitions`)
          .join(" | ")
      : status.status || "—";
  const drivetrain = status.drivetrain_config;
  $("drivetrain").textContent = drivetrain
    ? `${drivetrain.name || "configured robot"}${drivetrain.randomize ? " · randomized" : " · fixed"}`
    : "Unavailable";
  $("curriculum").textContent =
    [
      status.strategic_decision_rate_hz
        ? `${status.strategic_decision_rate_hz} Hz strategy`
        : null,
      status.gamma ? `γ ${status.gamma}` : null,
      selectedLiveStatus.entropy == null && latestMetrics.entropy == null
        ? null
        : `entropy ${Number(selectedLiveStatus.entropy ?? latestMetrics.entropy).toFixed(3)}`,
    ]
      .filter(Boolean)
      .join(" · ") || "—";
  const generationHistory = selectedCampaignTask.generation_history ||
    (selectedTask === status.task ? status.generation_history : []) || [];
  history = [...generationHistory].sort(
    (a, b) => Number(a.generation) - Number(b.generation),
  );
  const metricGeneration = latestMetrics.generation;
  $("latestMetricsSummary").textContent = metricGeneration == null
    ? "No completed-generation metrics are available yet."
    : `Latest completed ${selectedTask[0].toUpperCase()}${selectedTask.slice(1)} generation ${metricGeneration}: ` +
      `mean return ${metricValue(latestMetrics.mean_return)} · ` +
      `FUEL acquired ${integer(latestMetrics.fuel_acquired)} · ` +
      `FUEL scored ${integer(latestMetrics.fuel_scored)} · ` +
      `entropy ${metricValue(latestMetrics.entropy, 3)} · ` +
      `${integer(latestMetrics.complete_matches_observed)} complete matches`;
  drawTrainingChart();
  renderExposure(selectedLiveStatus.opponent_exposure_counts || latestMetrics.opponent_exposure_counts || {});
  renderActions(
    selectedLiveStatus.per_action_selection_rate || selectedLiveStatus.action_distribution ||
      latestMetrics.per_action_selection_rate || latestMetrics.action_distribution || [],
  );
}

function renderComparisonRecords(records) {
  renderComparisons({
    table: $("ablations"),
    records,
    labelRole,
    metricValue,
    createElement: (tag) => document.createElement(tag),
  });
}

function updateRobotAudit() {
  const panel = $("robotAudit");
  if (!panel) return;
  const robot = Number($("auditRobot").value);
  const frame = frames[index];
  if (!frame?.robots?.[robot]) {
    panel.textContent = "Load a playback to inspect its control state.";
    return;
  }
  const pose = frame.robots[robot];
  const fuel = frame.robot_fuel_targets?.[robot];
  const target = frame.robot_targets?.[robot];
  const routeGoal = frame.robot_route_goals?.[robot];
  const action = Number(frame.robot_actions?.[robot]);
  const actionNames = ["Approach fuel", "Approach fuel", "Approach fuel", "Approach fuel",
    "Score", "Defend", "Seek fuel / ferry", "Idle"];
  const point = (value) => Array.isArray(value) && value.length >= 2
    ? `${Number(value[0]).toFixed(2)}, ${Number(value[1]).toFixed(2)} m` : "—";
  const close = Array.isArray(fuel) && Array.isArray(routeGoal) &&
    Math.hypot(fuel[0] - routeGoal[0], fuel[1] - routeGoal[1]) < 0.2;
  const collecting = frame.robot_collecting?.[robot] === true;
  const state = collecting ? "Collecting" : action === 4 ? "Scoring"
    : action === 5 ? "Defending" : action === 6 ? "Searching / ferrying" : "Approaching fuel";
  const clusterCount = frame.robot_cluster_counts?.[robot];
  panel.textContent = `${["Red 1", "Red 2", "Red 3", "Blue 1", "Blue 2", "Blue 3"][robot]} · ${state} · ${actionNames[action] || "Unknown"}` +
    ` · cluster tracks ${clusterCount ?? "—"} · pose ${point(pose)}` +
    ` · selected FUEL ${point(fuel)} · behavior aim before AD* ${point(target)} (not the drive target)` +
    ` · AD* route destination ${point(routeGoal)}${close ? " (near selected FUEL)" : " (may be a staging waypoint)"}`;
}

function drawField() {
  renderField({
    canvas: field,
    context: fx,
    world,
    frames,
    currentFrame: currentPlaybackFrame,
    index,
    playbackTask,
    loadedScenarios,
    playbackRobotTypes,
    playbackFuelCapacities,
    playbackTimes,
    playbackSimTime,
    elements: {
      showGrid: $("showGrid"),
      redScore: $("redScore"),
      blueScore: $("blueScore"),
      redScoreLabel: $("redScoreLabel"),
      blueScoreLabel: $("blueScoreLabel"),
      matchClock: $("matchClock"),
      showFuel: $("showFuel"),
      showClusterBoundary: $("showClusterBoundary"),
      gameState: $("gameState"),
      showGhosts: $("showGhosts"),
      scenario: $("scenario"),
      frame: $("frame"),
      frameCount: $("frameCount"),
      auditRobot: $("auditRobot"),
    },
    integer,
  });
  updateRobotAudit();
}

function setPlaybackControls() {
  const ready = frames.length > 0;
  const canGenerate = $("playbackSource").value === "randomized";
  $("play").disabled = !ready && (!canGenerate || zoneLoading);
  $("play").textContent = ready
    ? playing
      ? "Pause"
      : "Play"
    : zoneLoading
      ? "Simulating…"
      : "Generate & play";
  $("play").setAttribute("aria-pressed", String(playing));
  $("frame").disabled = !ready;
  $("nextScenario").disabled = zoneLoading;
}

function setNotice(message, state = "") {
  $("matchupNotice").textContent = message;
  $("matchupNotice").dataset.state = state;
}

function saveFocusedTestSettings() {
  if ($("behaviorMode").value === "match") {
    localStorage.removeItem(focusedSettingsStorageKey);
    return;
  }
  localStorage.setItem(focusedSettingsStorageKey, JSON.stringify({
    behavior_mode: $("behaviorMode").value,
    robot_type: $("focusedRobotType").value,
    hopper_capacity: Number($("focusedHopperCapacity").value),
    scoring_bps: Number($("focusedScoringBps").value),
    random_gamepiece_placement: $("focusedRandomPlacement").checked,
  }));
}

async function loadZonePlayback({ autoplay = true, resumeJob = null,
                                  newScenario = false, batchMode = null,
                                  batchPosition = null,
                                  skipPreparedCache = false } = {}) {
  const requestedBehavior = batchMode || $("behaviorMode")?.value || "match";
  const focusedType = $("focusedRobotType")?.value || "dumper";
  const seedInput = $("scenarioSeed");
  const enteredSeed = seedInput.value.trim();
  if (enteredSeed &&
      (!/^\d+$/.test(enteredSeed) || Number(enteredSeed) > 2147483647)) {
    seedInput.setCustomValidity("Enter a whole-number seed from 0 to 2147483647.");
    seedInput.reportValidity();
    return;
  }
  seedInput.setCustomValidity("");
  activeRequest?.abort();
  const controller = new AbortController();
  activeRequest = controller;
  const revision = ++requestRevision;
  let savedJob = resumeJob;
  if (!savedJob && !newScenario && !batchMode) {
    try { savedJob = JSON.parse(localStorage.getItem(scenarioJobStorageKey) || "null"); }
    catch (_) { savedJob = null; }
  }
  if (savedJob && enteredSeed && Number(enteredSeed) !== Number(savedJob.seed)) {
    savedJob = null;
  }
  let behaviorMode = batchMode || savedJob?.behavior_mode || $("behaviorMode")?.value || "match";
  if (behaviorMode === "collect") behaviorMode = "collect_active";
  const behaviorProbe = behaviorMode !== "match";
  const [, probeHubState] = behaviorProbe ? behaviorMode.split("_") : [];
  const behaviorControl = $("behaviorMode");
  if (behaviorControl) behaviorControl.value = behaviorMode;
  const start = savedJob?.start || "red";
  const goal = savedJob?.goal || "blue";
  const task = savedJob?.task || "3v3";
  const seed = savedJob
    ? Number(savedJob.seed)
    : enteredSeed
      ? Number(enteredSeed)
      : Math.floor(Math.random() * 2147483647);
  seedInput.value = String(seed);
  const simulationId = `${behaviorProbe ? "focused" : "scenario"}-${seed}-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
  renderFocusedControls();
  zoneLoading = true;
  playing = false;
  setPlaybackControls();
  setNotice(behaviorProbe
    ? `Requesting the 30-second focused HUB ${probeHubState} test.`
    : "Requesting GPU time for a randomized six-robot match. PPO will yield while it simulates.");
  const requestedTicks = behaviorProbe ? 1500 : 8000;
  const requestedSeconds = behaviorProbe ? 30 : 160;
  showScenarioProgress({ status: "waiting", tick: 0, total_ticks: requestedTicks,
    simulated_seconds: 0, match_seconds: requestedSeconds, percent: 0 });
  try {
    const savedSetups = Array.isArray(savedJob?.control_modes)
      ? savedJob.control_modes : [];
    const savedTypes = Array.isArray(savedJob?.robot_types)
      ? savedJob.robot_types : [];
    const selectedSetups = Array.from({ length: 6 }, (_, robot) => {
      const saved = savedSetups[robot] || $("robotMode" + robot)?.value ||
        (robot < 3 ? "offense_deterministic" : "defense_deterministic");
      return saved === "offense_nn" ? "offense_deterministic" : saved;
    });
    const robotSetups = behaviorProbe
      ? ["offense_deterministic", "none", "none", "none", "none", "none"]
      : selectedSetups;
    if (!behaviorProbe) robotSetups.forEach((setup, robot) => {
      const control = $("robotMode" + robot);
      if (control) control.value = setup;
    });
    const robotTypes = behaviorProbe ? [$("focusedRobotType")?.value || "dumper", ...Array(5).fill("dumper")] : Array.from({ length: 6 }, (_, robot) =>
      savedTypes[robot] || $("robotType" + robot)?.value || "dumper");
    if (behaviorProbe && $("focusedRobotType")) $("focusedRobotType").value = robotTypes[0];
    if (!behaviorProbe) robotTypes.forEach((type, robot) => {
      const control = $("robotType" + robot);
      if (control) control.value = type;
    });
    const hopperControl = $(behaviorProbe ? "focusedHopperCapacity" : "hopperCapacity");
    const scoringControl = $(behaviorProbe ? "focusedScoringBps" : "scoringBps");
    const hopperCapacity = behaviorProbe
      ? Number(hopperControl?.value ?? (robotTypes[0] === "turret" ? 40 : 60))
      : Number(savedJob?.hopper_capacity ?? hopperControl?.value ?? 60);
    const scoringBps = behaviorProbe
      ? Number(scoringControl?.value ?? (robotTypes[0] === "turret" ? 15 : 25))
      : Number(savedJob?.scoring_bps ?? scoringControl?.value ?? 25);
    const teammateIntentKnowledge = behaviorProbe ? false : Boolean(
      savedJob?.teammate_intent_knowledge ?? $("teammateIntentKnowledge")?.checked ?? true,
    );
    const sweepingEnabled = behaviorProbe ? false : Boolean(
      savedJob?.sweeping_enabled ?? $("sweepingEnabled")?.checked ?? false,
    );
    const intentControl = $("teammateIntentKnowledge");
    if (intentControl) intentControl.checked = teammateIntentKnowledge;
    const sweepingControl = $("sweepingEnabled");
    if (sweepingControl) sweepingControl.checked = sweepingEnabled;
    if (hopperControl) hopperControl.value = String(hopperCapacity);
    if (scoringControl) scoringControl.value = String(scoringBps);
    const matchSetup = $("matchSetup");
    if (matchSetup) matchSetup.textContent = behaviorProbe
      ? `Focused HUB ${probeHubState} · 30 s · One ${robotTypes[0]} · ${hopperCapacity} FUEL · ${scoringBps} FUEL/s`
      : `REBUILT 3v3 · 160 s · Dumper ${hopperCapacity} FUEL / ${scoringBps} FUEL/s · Turret 40 / 15`;
    const query = new URLSearchParams({
      task,
      start,
      goal,
      seed: String(seed),
      simulation_id: simulationId,
      behavior_mode: behaviorMode,
      hopper_capacity: String(hopperCapacity),
      scoring_bps: String(scoringBps),
      random_gamepiece_placement: String(Boolean(behaviorProbe && $("focusedRandomPlacement")?.checked)),
      teammate_intent_knowledge: String(teammateIntentKnowledge),
      sweeping_enabled: String(sweepingEnabled),
      dual_gpu: String(Boolean($("dualGpuMode")?.checked)),
    });
    if (!behaviorProbe) robotTypes.forEach((type, robot) => {
      query.set(`robot_type${robot}`, type);
    });
    robotSetups.forEach((setup, robot) => {
      query.set(`robot${robot}`, setup);
    });
    const job = { simulation_id: simulationId, seed, start, goal, task,
      behavior_mode: behaviorMode,
      control_modes: robotSetups, hopper_capacity: hopperCapacity,
      scoring_bps: scoringBps, robot_types: robotTypes,
      teammate_intent_knowledge: teammateIntentKnowledge,
      sweeping_enabled: sweepingEnabled,
      random_gamepiece_placement: behaviorProbe && Boolean($("focusedRandomPlacement")?.checked),
      dual_gpu: Boolean($("dualGpuMode")?.checked) };
    localStorage.setItem(scenarioJobStorageKey, JSON.stringify(job));
    let record;
    if (savedJob) {
      record = await waitForZonePlayback(simulationId, revision,
        controller.signal, query, true);
    } else {
      try {
        const response = await fetch(`/api/zone-playback?${query}`, {
          cache: "no-store", signal: controller.signal,
        });
        const startRecord = await response.json();
        if (!response.ok && response.status !== 202)
          throw new Error(startRecord.error || `Scenario request failed (${response.status})`);
        record = await waitForZonePlayback(simulationId, revision,
          controller.signal, query, false);
      } catch (error) {
        if (!(error instanceof TypeError) && !/fetch|network/i.test(error.message)) throw error;
        record = await waitForZonePlayback(simulationId, revision,
          controller.signal, query, true);
      }
    }
    if (revision !== requestRevision) return;
    if (Array.isArray(record.robot_types)) {
      record.robot_types.forEach((type, robot) => {
        if (type !== "dumper" && type !== "turret") return;
        robotTypes[robot] = type;
        const control = $("robotType" + robot);
        if (control) control.value = type;
      });
      const saved = JSON.parse(localStorage.getItem(scenarioJobStorageKey) || "{}");
      localStorage.setItem(scenarioJobStorageKey,
        JSON.stringify({ ...saved, robot_types: robotTypes }));
    }
    const scenarioTicks = Number(record.simulation_ticks) || requestedTicks;
    const simulatedSeconds = Number(record.simulated_seconds) || requestedSeconds;
    const playedBehaviorMode = record.behavior_mode || behaviorMode;
    const playedProbe = playedBehaviorMode !== "match";
    const [playedProbeBehavior, playedHubState] = playedProbe
      ? playedBehaviorMode.split("_") : [];
    showScenarioProgress({ status: "completed", tick: scenarioTicks, total_ticks: scenarioTicks,
      simulated_seconds: simulatedSeconds, match_seconds: simulatedSeconds, percent: 100,
      behavior_mode: playedBehaviorMode });
    playbackTask = record.task || task;
    playbackRobotTypes = robotTypes.slice();
    playbackFuelCapacities = record.simulation_constraints?.max_fuel_per_robot ||
      robotTypes.map((type) => type === "turret" ? 40 : hopperCapacity);
    world = record.field || { ...defaultWorld };
    loadedScenarios = record.scenarios || [];
    const scenario = loadedScenarios[0];
    frames = scenario?.frames || record.frames || [];
    // A direct scenario link is used to inspect a specific completed match.
    // Open it on the final score so the result is visible immediately; the
    // user can still scrub or replay from the beginning with the controls.
    const requestedScenario = new URLSearchParams(location.search).get("scenario");
    index = requestedScenario === simulationId ? Math.max(0, frames.length - 1) : 0;
    ensurePlaybackClock();
    playing =
      autoplay &&
      frames.length > 0 &&
      !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    $("scenarioCount").textContent = playedProbe
      ? `${playedProbeBehavior[0].toUpperCase()}${playedProbeBehavior.slice(1)} probe · HUB ${playedHubState} · ${simulatedSeconds.toFixed(0)} simulated seconds · ${frames.length.toLocaleString()} frames`
      : `${labelRole(task)} · randomized REBUILT match · seed ${seed} · ${frames.length.toLocaleString()} frames`;
    $("playbackSourceNote").textContent = playedProbe
      ? `Isolated behavior probe · Red robot 1 (${robotTypes[0]}) · HUB ${playedHubState} · ${playedProbeBehavior}.`
      : `Six-robot match · Red ${robotSetups.slice(0, 3).map((setup, i) => `${robotTypes[i]} ${describeRobotSetup(setup)}`).join(" / ")} · Blue ${robotSetups.slice(3).map((setup, i) => `${robotTypes[i + 3]} ${describeRobotSetup(setup)}`).join(" / ")}.`;
    setNotice(
      frames.length
        ? record.compute_scheduling === "reserved"
          ? playedProbe ? "Behavior probe ready. PPO has resumed." : "Randomized scenario ready. PPO has resumed."
          : playedProbe ? "Behavior probe ready." : "Randomized scenario ready."
        : "Scenario simulation returned no frames.",
      frames.length ? "success" : "error",
    );
    setPlaybackControls();
    drawField();
    refreshScenarioReplays();
  } catch (error) {
    if (error.name !== "AbortError" && revision === requestRevision) {
      const progressState = $("scenarioProgressState");
      if (progressState) progressState.textContent = error.message;
      setNotice(
        `Could not create the scenario: ${error.message}. Choose ${behaviorProbe ? "Generate test" : "New scenario"} to retry.`,
        "error",
      );
    }
  } finally {
    if (revision === requestRevision) {
      zoneLoading = false;
      activeRequest = null;
      setPlaybackControls();
      renderFocusedControls();
    }
  }
}

async function loadFocusedReplay() {
  const mode = $("behaviorMode").value;
  const type = $("focusedRobotType").value;
  const defaultCapacity = type === "turret" ? 40 : 60;
  const defaultRate = type === "turret" ? 15 : 25;
  const matchesPreparedSettings = !$("focusedRandomPlacement")?.checked &&
    Number($("focusedHopperCapacity")?.value) === defaultCapacity &&
    Number($("focusedScoringBps")?.value) === defaultRate;
  const revision = ++requestRevision;
  activeRequest?.abort();
  const controller = new AbortController();
  activeRequest = controller;
  zoneLoading = true;
  playing = false;
  $("scenarioProgress").hidden = true;
  setPlaybackControls();
  setNotice(`Loading the ${type} ${mode.replace("_", " · HUB ")} replay.`);
  const url = `/api/focused-playback?behavior_mode=${encodeURIComponent(mode)}&robot_type=${encodeURIComponent(type)}&dual_gpu=${Boolean($("dualGpuMode")?.checked)}`;
  let record;
  try {
  while (revision === requestRevision) {
    const response = await fetch(url, { cache: "no-store", signal: controller.signal });
    if (response.status === 202) {
      const job = await response.json();
      const progressResponse = await fetch(`/api/zone-playback-progress?simulation_id=${encodeURIComponent(job.simulation_id)}`, { cache: "no-store", signal: controller.signal });
      const progress = await progressResponse.json();
      if (progress.behavior_mode == null) progress.behavior_mode = mode;
      showScenarioProgress(progress);
      if (progress.status === "error") throw new Error(progress.error || "Focused replay generation failed");
      if (progress.status !== "ready") {
        await new Promise((resolve) => setTimeout(resolve, 1000));
        continue;
      }
      const resultResponse = await fetch(`/api/zone-playback-result?simulation_id=${encodeURIComponent(job.simulation_id)}`, { cache: "no-store", signal: controller.signal });
      record = await resultResponse.json();
      if (!resultResponse.ok) throw new Error(record.error || "Focused replay could not be loaded");
      break;
    }
    if (!response.ok) throw new Error(`Prepared replay unavailable (${response.status})`);
    record = await response.json();
    break;
  }
  if (!record || revision !== requestRevision) return;
  if (!matchesPreparedSettings) {
    return loadZonePlayback({ newScenario: true, batchMode: mode,
      skipPreparedCache: true });
  }
  showScenarioProgress({ status: "completed", tick: 1500, total_ticks: 1500,
    simulated_seconds: 30, match_seconds: 30, percent: 100,
    behavior_mode: mode });
  playbackTask = record.task || "3v3";
  playbackRobotTypes = record.robot_types || [type];
  playbackFuelCapacities = record.simulation_constraints?.max_fuel_per_robot || [];
  world = record.field || { ...defaultWorld };
  loadedScenarios = record.scenarios || [];
  frames = loadedScenarios[0]?.frames || record.frames || [];
  index = 0;
  ensurePlaybackClock();
  playing = false;
  $("scenarioCount").textContent = `${mode.replaceAll("_", " ")} · ${type} · prepared replay · ${frames.length.toLocaleString()} frames`;
  $("playbackSourceNote").textContent = `Prepared 30 second one-robot ${type} ${mode.replace("_", " · HUB ")}.`;
  setNotice("Prepared focused replay ready.", "success");
  setPlaybackControls();
  drawField();
  } catch (error) {
    if (error.name !== "AbortError" && revision === requestRevision)
      setNotice(`Could not load focused replay: ${error.message}`, "error");
  } finally {
    if (revision === requestRevision) {
      zoneLoading = false;
      activeRequest = null;
      setPlaybackControls();
    }
  }
}

async function loadEvaluations() {
  const response = await fetch("/api/game-evaluations", { cache: "no-store" });
  if (!response.ok)
    throw new Error(`Evaluation list failed (${response.status})`);
  const entries = await response.json();
  const select = $("gameEvaluation");
  select.replaceChildren(
    new Option("Choose an evaluation", ""),
    ...entries.map((entry) => new Option(entry.label, entry.id)),
  );
}

async function loadGameEvaluation() {
  const name = $("gameEvaluation").value;
  if (!name) return;
  activeRequest?.abort();
  const controller = new AbortController();
  activeRequest = controller;
  const revision = ++requestRevision;
  zoneLoading = true;
  playing = false;
  setPlaybackControls();
  setNotice("Loading the recorded evaluation…");
  try {
    const response = await fetch(
      `/api/game-evaluation-playback?name=${encodeURIComponent(name)}`,
      { cache: "no-store", signal: controller.signal },
    );
    const record = await response.json();
    if (!response.ok)
      throw new Error(
        record.error || `Playback request failed (${response.status})`,
      );
    if (revision !== requestRevision) return;
    playbackTask = record.task || "counter_defense";
    playbackRobotTypes = Array.isArray(record.robot_types)
      ? record.robot_types.map((type) => type === "turret" ? "turret" : "dumper")
      : [];
    playbackFuelCapacities = record.simulation_constraints?.max_fuel_per_robot ||
      playbackRobotTypes.map((type) => type === "turret" ? 40 : 60);
    world = record.field || { ...defaultWorld };
    loadedScenarios = record.scenarios?.length
      ? record.scenarios
      : [
          {
            id: "evaluation",
            label: "Evaluation",
            frames: record.frames || [],
          },
        ];
    frames = loadedScenarios[0]?.frames || [];
    index = 0;
    ensurePlaybackClock();
    playing = false;
    $("scenarioCount").textContent =
      `${name.replaceAll("_", " ")} · ${frames.length.toLocaleString()} frames`;
    $("playbackSourceNote").textContent =
      "Recorded evaluation playback; these metrics are separate from the randomized demo.";
    setNotice("Recorded evaluation ready.", "success");
    setPlaybackControls();
    drawField();
  } catch (error) {
    if (error.name !== "AbortError" && revision === requestRevision)
      setNotice(`Could not load evaluation: ${error.message}`, "error");
  } finally {
    if (revision === requestRevision) {
      zoneLoading = false;
      activeRequest = null;
      setPlaybackControls();
    }
  }
}

function renderFocusedControls() {
  const randomized = $("playbackSource").value === "randomized";
  const focused = randomized && $("behaviorMode").value !== "match";
  $("playbackSourceControl").hidden = focused;
  $("playbackSourceNote").hidden = focused;
  $("matchSetup").hidden = !randomized || focused;
  $("robotControlModes").hidden = !randomized || focused;
  $("scenarioPhysicalLimits").hidden = !randomized || focused;
  $("focusedRobotControls").hidden = !randomized || !focused;
  for (const id of ["teammateIntentKnowledge", "sweepingEnabled", "scenarioSeed"]) {
    $(id).closest("label").hidden = !randomized || focused;
  }
  $("nextScenario").hidden = !randomized;
  $("newScenarioLabel").textContent = focused ? "Generate test" : "New scenario";
  $("auditRobot").hidden = focused;
  document.querySelector('label[for="auditRobot"]').hidden = focused;
  if (focused) $("auditRobot").value = "0";
}

function renderSource() {
  const randomized = $("playbackSource").value === "randomized";
  $("robotControlModes").hidden = !randomized;
  $("scenarioPhysicalLimits").hidden = !randomized;
  $("behaviorModeControl").hidden = !randomized;
  $("matchSetup").hidden = !randomized || $("behaviorMode").value !== "match";
  $("evaluationControls").hidden = randomized;
  $("nextScenario").hidden = !randomized || $("behaviorMode").value !== "match";
  renderFocusedControls();
  if (randomized) setNotice("Choose a saved replay or generate a new scenario.");
  else setNotice("Choose a recorded evaluation to load its playback.");
  setPlaybackControls();
}

$("loadScenarioReplay").addEventListener("click", async () => {
  const replay = scenarioReplays.find((item) => item.simulation_id === $("scenarioReplay").value);
  if (!replay) return;
  const response = await fetch(`/api/zone-playback-result?simulation_id=${encodeURIComponent(replay.simulation_id)}`, { cache: "no-store" });
  const record = await response.json();
  if (!response.ok) { setNotice(record.error || "Replay could not be loaded.", "error"); return; }
  playbackTask = record.task || "3v3";
  playbackRobotTypes = record.robot_types || [];
  playbackFuelCapacities = record.simulation_constraints?.max_fuel_per_robot || [];
  world = record.field || { ...defaultWorld };
  loadedScenarios = record.scenarios || [];
  frames = loadedScenarios[0]?.frames || record.frames || [];
  index = 0; playing = false; ensurePlaybackClock();
  $("scenarioCount").textContent = `${record.behavior_mode || "match"} · seed ${record.seed ?? "—"} · ${frames.length.toLocaleString()} frames`;
  const state = scenarioReplays.find((item) => item.simulation_id === replay.simulation_id);
  $("scenarioReplayStatus").textContent = state?.outdated
    ? `Outdated: created at commit ${state.code_state?.commit || "unknown"}; current code differs.`
    : `Current code: ${state?.code_state?.commit || "unknown"}.`;
  setNotice("Saved replay loaded.", "success"); setPlaybackControls(); drawField();
});
$("regenerateScenarioReplay").addEventListener("click", () => {
  const selected = scenarioReplays.find((item) => item.simulation_id === $("scenarioReplay").value);
  const mode = selected?.behavior_mode || $("behaviorMode").value;
  if (mode !== "match") $("behaviorMode").value = mode;
  if (selected?.robot_types?.[0] && $("focusedRobotType"))
    $("focusedRobotType").value = selected.robot_types[0];
  if (selected?.seed != null && mode === "match") $("scenarioSeed").value = String(selected.seed);
  loadZonePlayback({ newScenario: true, batchMode: mode, skipPreparedCache: true });
});

function renderProgressFrame() {
  if (playing && frames.length && playbackTimes.length) {
    const now = performance.now();
    if (!renderProgressFrame.lastTime) renderProgressFrame.lastTime = now;
    const delta = ((now - renderProgressFrame.lastTime) / 1000) * playbackSpeed;
    renderProgressFrame.lastTime = now;
    if (playbackSimTime + delta >= playbackTimes.at(-1)) {
      setPlaybackTime(playbackTimes.at(-1));
      playing = false;
      renderProgressFrame.lastTime = 0;
      setPlaybackControls();
    } else setPlaybackTime(playbackSimTime + delta);
    drawField();
  } else {
    renderProgressFrame.lastTime = 0;
  }
  requestAnimationFrame(renderProgressFrame);
}

async function pollTraining() {
  try {
    const response = await fetch("/api/training-status", { cache: "no-store" });
    if (!response.ok)
      throw new Error(`Status request failed (${response.status})`);
    const status = await response.json();
    renderTraining(status);
    const live =
      status.status === "running" || status.campaign?.status === "running";
    $("connection").dataset.status =
      status.status === "failed"
        ? "error"
        : live
          ? "running"
          : status.status || "waiting";
    const generation = status.current_generation ?? status.generation;
    if (generation != null) lastScenarioGeneration = generation;
    pollDelay = 1500;
  } catch (error) {
    $("connection").dataset.status = "error";
    $("state").textContent = "Status unavailable · retrying";
    $("runPhase").textContent = "Connection issue";
    $("runPhase").dataset.status = "error";
    pollDelay = Math.min(15000, Math.max(3000, pollDelay * 2));
  } finally {
    clearTimeout(pollTimer);
    pollTimer = setTimeout(pollTraining, pollDelay);
  }
}

$("play").addEventListener("click", () => {
  if (!frames.length) return;
  if (!playing && index >= frames.length - 1) {
    index = 0;
    ensurePlaybackClock();
  }
  playing = !playing;
  setPlaybackControls();
});
$("dashboardRefresh").addEventListener("click", () => window.location.reload());
$("dualGpuMode").addEventListener("change", () =>
  localStorage.setItem(dualGpuStorageKey, String($("dualGpuMode").checked)),
);
$("nextScenario").addEventListener("click", () => {
  if ($("behaviorMode").value !== "match") {
    saveFocusedTestSettings();
    setNotice("Focused settings are ready. Choose Regenerate replay when ready.");
  } else {
    setNotice("Choose Regenerate replay when you are ready to simulate.");
  }
});
$("focusedRobotType").addEventListener("change", (event) => {
  const turret = event.target.value === "turret";
  $("focusedHopperCapacity").value = turret ? "40" : "60";
  $("focusedScoringBps").value = turret ? "15" : "25";
  saveFocusedTestSettings();
  if ($("behaviorMode").value !== "match" && $("playbackSource").value === "randomized")
    setNotice("Robot type changed. Choose a saved replay or regenerate when ready.");
});
$("focusedRandomPlacement").addEventListener("change", () => {
  saveFocusedTestSettings();
  if ($("behaviorMode").value !== "match")
    setNotice("Focused settings are ready. Choose Regenerate replay when ready.");
});
$("focusedHopperCapacity").addEventListener("change", saveFocusedTestSettings);
$("focusedScoringBps").addEventListener("change", saveFocusedTestSettings);
$("scenarioSeed").addEventListener("input", (event) => {
  event.target.setCustomValidity("");
});
function robotModeChanged() {
  frames = [];
  loadedScenarios = [];
  index = 0;
  playing = false;
  ensurePlaybackClock();
  $("scenarioCount").textContent = "Scenario settings changed · choose New scenario to simulate them.";
  setNotice("Scenario settings are ready. Choose New scenario to apply them.");
  setPlaybackControls();
  drawField();
}
for (let robot = 0; robot < 6; robot++) {
  $("robotMode" + robot).addEventListener("change", robotModeChanged);
  $("robotType" + robot).addEventListener("change", robotModeChanged);
}
$("sweepingEnabled").addEventListener("change", robotModeChanged);
$("behaviorMode").addEventListener("change", () => {
  robotModeChanged();
  saveFocusedTestSettings();
  renderFocusedControls();
  setNotice("Scenario settings are ready. Choose Generate test or New scenario when ready.");
});
$("playbackSource").addEventListener("change", renderSource);
$("gameEvaluation").addEventListener("change", loadGameEvaluation);
$("playbackSpeed").addEventListener("change", (event) => {
  playbackSpeed = Number(event.target.value) || 1;
});
$("frame").addEventListener("input", (event) => {
  playing = false;
  setPlaybackTime(playbackTimes[Number(event.target.value)] || 0);
  setPlaybackControls();
  drawField();
});
$("auditRobot").addEventListener("change", drawField);
$("trendMetric").addEventListener("change", drawTrainingChart);
$("trendTask").addEventListener("change", () => {
  trendTaskChosen = true;
  if (latestTraining) renderTraining(latestTraining);
});
for (const id of ["showFuel", "showClusterBoundary", "showGrid", "showGhosts"])
  $(id).addEventListener("change", drawField);
window.addEventListener("resize", drawField);

async function initializeDashboard() {
  const initialParams = new URLSearchParams(location.search);
  const requestedBehavior = initialParams.get("behavior_mode");
  let initialSavedJob = null;
  try { initialSavedJob = JSON.parse(localStorage.getItem(scenarioJobStorageKey) || "null"); }
  catch (_) { localStorage.removeItem(scenarioJobStorageKey); }
  let focusedSettings = null;
  try { focusedSettings = JSON.parse(localStorage.getItem(focusedSettingsStorageKey) || "null"); }
  catch (_) { localStorage.removeItem(focusedSettingsStorageKey); }
  const savedDualGpu = localStorage.getItem(dualGpuStorageKey);
  if (savedDualGpu !== null) $("dualGpuMode").checked = savedDualGpu === "true";
  if (["collect_active", "collect_inactive"].includes(requestedBehavior)) {
    $("behaviorMode").value = requestedBehavior;
    const requestedType = initialParams.get("robot_type");
    if (["dumper", "turret"].includes(requestedType))
      $("focusedRobotType").value = requestedType;
  } else if (focusedSettings?.behavior_mode || initialSavedJob?.behavior_mode) {
    const savedFocused = focusedSettings || initialSavedJob;
    $("behaviorMode").value = savedFocused.behavior_mode;
    const savedType = savedFocused.robot_type || savedFocused.robot_types?.[0];
    if (["dumper", "turret"].includes(savedType))
      $("focusedRobotType").value = savedType;
    if (Number.isFinite(Number(savedFocused.hopper_capacity)))
      $("focusedHopperCapacity").value = String(savedFocused.hopper_capacity);
    if (Number.isFinite(Number(savedFocused.scoring_bps)))
      $("focusedScoringBps").value = String(savedFocused.scoring_bps);
    $("focusedRandomPlacement").checked = Boolean(savedFocused.random_gamepiece_placement);
  }
  if (!initialParams.has("scenario")) {
    // A reload restores user preferences, never a generation request.
    initialSavedJob = null;
  }
  renderFocusedControls();
  drawField();
  drawTrainingChart();
  setPlaybackControls();
  await refreshScenarioReplays();
  pollTraining();
  try {
    const response = await fetch("/api/field-layout", { cache: "no-store" });
    if (!response.ok)
      throw new Error(`Field layout request failed (${response.status})`);
    world = await response.json();
    drawField();
  } catch (error) {
    console.error("Could not load the REBUILT field layout", error);
  }
  try {
    await Promise.all([
      loadEvaluations(),
      fetch("/api/ablations", { cache: "no-store" })
        .then((response) =>
          response.ok
            ? response.json()
            : Promise.reject(
                new Error(`Comparison request failed (${response.status})`),
              ),
        )
        .then(renderComparisonRecords),
    ]);
  } catch (error) {
    $("ablations").querySelector("tbody").innerHTML =
      `<tr><td colspan="7" class="empty-note">Could not load comparisons: ${error.message}</td></tr>`;
  }
  if (initialRun) $("runIdentity").textContent = `Requested run: ${initialRun}`;
  if ($("playbackSource").value === "randomized") {
    let savedJob = null;
    try { savedJob = JSON.parse(localStorage.getItem(scenarioJobStorageKey) || "null"); }
    catch (_) { localStorage.removeItem(scenarioJobStorageKey); }
    const requestedScenario = new URLSearchParams(location.search).get("scenario");
    if (requestedScenario && /^[a-zA-Z0-9_-]{1,80}$/.test(requestedScenario)) {
      // A deep link names one exact match. Never let a stale localStorage job
      // or the dashboard's generic active job silently replace it.
      if (savedJob?.simulation_id !== requestedScenario) savedJob = null;
      try {
        const response = await fetch("/api/zone-playback-active", { cache: "no-store" });
        const active = response.ok ? await response.json() : null;
        const matchingJob = active?.job?.simulation_id === requestedScenario
          ? active.job : null;
        if (matchingJob) savedJob = matchingJob;
      } catch (_) {
        // Keep the locally saved job if the requested scenario lookup fails.
      }
    }
    if (savedJob?.simulation_id && !savedJob.behavior_mode) {
      try {
        const response = await fetch(
          `/api/zone-playback-progress?simulation_id=${encodeURIComponent(savedJob.simulation_id)}`,
          { cache: "no-store" },
        );
        const progress = response.ok ? await response.json() : null;
        if (progress && ["cancelled", "error"].includes(progress.status)) {
          localStorage.removeItem(scenarioJobStorageKey);
          savedJob = null;
        }
      } catch (_) {
        // Keep the saved ID and reconnect through the regular retry loop.
      }
    }
    if (requestedScenario && savedJob?.simulation_id === requestedScenario) {
      localStorage.setItem(scenarioJobStorageKey, JSON.stringify(savedJob));
      loadZonePlayback({ autoplay: true, resumeJob: savedJob });
    } else if (requestedScenario) {
      if (scenarioReplays.some((item) => item.simulation_id === requestedScenario)) {
        $("scenarioReplay").value = requestedScenario;
        $("loadScenarioReplay").click();
      }
    } else {
      setNotice("Choose a saved replay to load it, or generate a new one when ready.", "success");
    }
  }
  requestAnimationFrame(renderProgressFrame);
}

initializeDashboard();
