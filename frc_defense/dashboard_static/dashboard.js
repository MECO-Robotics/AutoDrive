import { renderComparisons, renderTrainingChart } from "./training_rendering.js";
import { renderField } from "./field_rendering.js";
const $ = (id) => document.getElementById(id);
const field = $("field");
const fx = field.getContext("2d");
const initialRun = new URLSearchParams(location.search).get("run");
const scenarioJobStorageKey = "autodrive.activeScenarioJob";
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
  const duration = Number(progress.match_seconds) || 160;
  const progressTime = $("scenarioProgressTime");
  if (progressTime) progressTime.textContent =
    `${simulated.toFixed(1)} / ${duration.toFixed(1)} simulated seconds`;
  const status = progress.status || "waiting";
  const progressTitle = $("scenarioProgressTitle");
  if (progressTitle) progressTitle.textContent = ["completed", "ready"].includes(status)
    ? "Randomized match ready"
    : status === "error"
      ? "Scenario simulation failed"
      : status === "waiting"
        ? "Match queued for simulator"
        : "Simulating randomized match";
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
      matchClock: $("matchClock"),
      showFuel: $("showFuel"),
      gameState: $("gameState"),
      showGhosts: $("showGhosts"),
      scenario: $("scenario"),
      frame: $("frame"),
      frameCount: $("frameCount"),
    },
    integer,
  });
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

async function loadZonePlayback({ autoplay = true, resumeJob = null,
                                  newScenario = false } = {}) {
  activeRequest?.abort();
  const controller = new AbortController();
  activeRequest = controller;
  const revision = ++requestRevision;
  let savedJob = resumeJob;
  if (!savedJob && !newScenario) {
    try { savedJob = JSON.parse(localStorage.getItem(scenarioJobStorageKey) || "null"); }
    catch (_) { savedJob = null; }
  }
  const start = savedJob?.start || "red";
  const goal = savedJob?.goal || "blue";
  const task = savedJob?.task || "3v3";
  const seed = savedJob?.seed || Math.floor(Math.random() * 2147483647);
  const simulationId = savedJob?.simulation_id || `${seed}-${Date.now()}`;
  zoneLoading = true;
  playing = false;
  setPlaybackControls();
  setNotice(
    "Requesting GPU time for a randomized six-robot match. PPO will yield while it simulates.",
  );
  showScenarioProgress({ status: "waiting", tick: 0, total_ticks: 8000,
    simulated_seconds: 0, match_seconds: 160, percent: 0 });
  try {
    const savedSetups = Array.isArray(savedJob?.control_modes)
      ? savedJob.control_modes : [];
    const savedTypes = Array.isArray(savedJob?.robot_types)
      ? savedJob.robot_types : [];
    const robotSetups = Array.from({ length: 6 }, (_, robot) => {
      const saved = savedSetups[robot] || $("robotMode" + robot)?.value ||
        (robot < 3 ? "offense_deterministic" : "defense_deterministic");
      return saved === "offense_nn" ? "offense_deterministic" : saved;
    });
    robotSetups.forEach((setup, robot) => {
      const control = $("robotMode" + robot);
      if (control) control.value = setup;
    });
    const robotTypes = Array.from({ length: 6 }, (_, robot) =>
      savedTypes[robot] || $("robotType" + robot)?.value || "dumper");
    robotTypes.forEach((type, robot) => {
      const control = $("robotType" + robot);
      if (control) control.value = type;
    });
    const hopperControl = $("hopperCapacity");
    const scoringControl = $("scoringBps");
    const hopperCapacity = Number(savedJob?.hopper_capacity ?? hopperControl?.value ?? 60);
    const scoringBps = Number(savedJob?.scoring_bps ?? scoringControl?.value ?? 25);
    const teammateIntentKnowledge = Boolean(
      savedJob?.teammate_intent_knowledge ?? $("teammateIntentKnowledge")?.checked ?? true,
    );
    const intentControl = $("teammateIntentKnowledge");
    if (intentControl) intentControl.checked = teammateIntentKnowledge;
    if (hopperControl) hopperControl.value = String(hopperCapacity);
    if (scoringControl) scoringControl.value = String(scoringBps);
    const matchSetup = $("matchSetup");
    if (matchSetup) matchSetup.textContent =
      `REBUILT 3v3 · 160 s · Dumper ${hopperCapacity} FUEL / ${scoringBps} FUEL/s · Turret 40 / 15`;
    const query = new URLSearchParams({
      task,
      start,
      goal,
      seed: String(seed),
      simulation_id: simulationId,
      hopper_capacity: String(hopperCapacity),
      scoring_bps: String(scoringBps),
      teammate_intent_knowledge: String(teammateIntentKnowledge),
    });
    robotTypes.forEach((type, robot) => {
      query.set(`robot_type${robot}`, type);
    });
    robotSetups.forEach((setup, robot) => {
      query.set(`robot${robot}`, setup);
    });
    const job = { simulation_id: simulationId, seed, start, goal, task,
      control_modes: robotSetups, hopper_capacity: hopperCapacity,
      scoring_bps: scoringBps, robot_types: robotTypes,
      teammate_intent_knowledge: teammateIntentKnowledge };
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
    showScenarioProgress({ status: "completed", tick: 8000, total_ticks: 8000,
      simulated_seconds: 160, match_seconds: 160, percent: 100 });
    playbackTask = record.task || task;
    playbackRobotTypes = robotTypes.slice();
    playbackFuelCapacities = record.simulation_constraints?.max_fuel_per_robot ||
      robotTypes.map((type) => type === "turret" ? 40 : hopperCapacity);
    world = record.field || { ...defaultWorld };
    loadedScenarios = record.scenarios || [];
    const scenario = loadedScenarios[0];
    frames = scenario?.frames || record.frames || [];
    index = 0;
    ensurePlaybackClock();
    playing =
      autoplay &&
      frames.length > 0 &&
      !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    $("scenarioCount").textContent =
      `${labelRole(task)} · randomized REBUILT match · seed ${seed} · ${frames.length.toLocaleString()} frames`;
    $("playbackSourceNote").textContent =
      `Six-robot match · Red ${robotSetups.slice(0, 3).map((setup, i) => `${robotTypes[i]} ${describeRobotSetup(setup)}`).join(" / ")} · Blue ${robotSetups.slice(3).map((setup, i) => `${robotTypes[i + 3]} ${describeRobotSetup(setup)}`).join(" / ")}.`;
    setNotice(
      frames.length
        ? record.compute_scheduling === "reserved"
          ? "Randomized scenario ready. PPO has resumed."
          : "Randomized scenario ready."
        : "Scenario simulation returned no frames.",
      frames.length ? "success" : "error",
    );
    setPlaybackControls();
    drawField();
  } catch (error) {
    if (error.name !== "AbortError" && revision === requestRevision) {
      const progressState = $("scenarioProgressState");
      if (progressState) progressState.textContent = error.message;
      setNotice(
        `Could not create the scenario: ${error.message}. Choose New scenario to retry.`,
        "error",
      );
    }
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

function renderSource() {
  const randomized = $("playbackSource").value === "randomized";
  $("robotControlModes").hidden = !randomized;
  $("scenarioPhysicalLimits").hidden = !randomized;
  $("matchSetup").hidden = !randomized;
  $("evaluationControls").hidden = randomized;
  $("nextScenario").hidden = !randomized;
  if (randomized) loadZonePlayback();
  else setNotice("Choose a recorded evaluation to load its playback.");
  setPlaybackControls();
}

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
    if (
      generation != null &&
      lastScenarioGeneration != null &&
      generation !== lastScenarioGeneration &&
      $("playbackSource").value === "randomized"
    ) {
      loadZonePlayback();
    }
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
  if (!frames.length) {
    if ($("playbackSource").value === "randomized" && !zoneLoading) {
      loadZonePlayback({ autoplay: true });
    }
    return;
  }
  if (!playing && index >= frames.length - 1) {
    index = 0;
    ensurePlaybackClock();
  }
  playing = !playing;
  setPlaybackControls();
});
$("nextScenario").addEventListener("click", () =>
  loadZonePlayback({ newScenario: true }),
);
function robotModeChanged() {
  frames = [];
  loadedScenarios = [];
  index = 0;
  playing = false;
  ensurePlaybackClock();
  $("scenarioCount").textContent = "Robot setup changed · choose New scenario to simulate it.";
  setNotice("Robot types, roles and controllers are ready. Choose New scenario to apply all six selections.");
  setPlaybackControls();
  drawField();
}
for (let robot = 0; robot < 6; robot++) {
  $("robotMode" + robot).addEventListener("change", robotModeChanged);
  $("robotType" + robot).addEventListener("change", robotModeChanged);
}
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
$("trendMetric").addEventListener("change", drawTrainingChart);
$("trendTask").addEventListener("change", () => {
  trendTaskChosen = true;
  if (latestTraining) renderTraining(latestTraining);
});
for (const id of ["showFuel", "showGrid", "showGhosts"])
  $(id).addEventListener("change", drawField);
window.addEventListener("resize", drawField);

async function initializeDashboard() {
  drawField();
  drawTrainingChart();
  setPlaybackControls();
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
    if (savedJob?.simulation_id) {
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
    if (!savedJob) {
      try {
        const response = await fetch("/api/zone-playback-active", { cache: "no-store" });
        if (response.ok) savedJob = (await response.json()).job || null;
      } catch (_) {
        // A missing connection should not create a second match automatically.
      }
    }
    if (savedJob) {
      localStorage.setItem(scenarioJobStorageKey, JSON.stringify(savedJob));
      loadZonePlayback({ autoplay: true, resumeJob: savedJob });
    } else {
      setNotice(
        "REBUILT field ready. Choose Generate & play to simulate a randomized match.",
        "success",
      );
    }
  }
  requestAnimationFrame(renderProgressFrame);
}

initializeDashboard();
