// Field and playback rendering. Runtime state and DOM elements are supplied by
// the dashboard entrypoint, keeping rendering independent of its module globals.
function robotAllianceColor(frame, robot, fallbackAttacker = 0) {
  const team = Array.isArray(frame?.robot_teams)
    ? frame.robot_teams[robot]
    : robot === fallbackAttacker
      ? 0
      : 1;
  return team === 0 ? "#ED1C24" : "#0066B3";
}

function drawAdstarRoutes(context, frame, attackerIndex, opacity, ox, scale, py) {
  if (!frame?.robots) return;
  const paths =
    Array.isArray(frame.adstar_paths)
      ? frame.adstar_paths
      : Array.from({ length: frame.robots.length }, () => []);
  for (let robot = 0; robot < Math.min(paths.length, frame.robots.length); robot++) {
    const route = paths[robot];
    if (!Array.isArray(route) || route.length < 2) continue;
    context.save();
    context.globalAlpha = opacity;
    context.beginPath();
    route.forEach((point, i) => {
      const x = ox + point[0] * scale;
      const y = py(point[1]);
      i ? context.lineTo(x, y) : context.moveTo(x, y);
    });
    context.setLineDash([2, 7]);
    context.lineCap = "round";
    context.strokeStyle = robotAllianceColor(frame, robot, attackerIndex);
    context.lineWidth = 3;
    context.stroke();
    context.restore();
  }
}

export function renderField({
  canvas, context: fx, world, frames, currentFrame, index, playbackTask,
  loadedScenarios, playbackRobotTypes, playbackFuelCapacities,
  playbackTimes, playbackSimTime, elements, integer,
}) {
  const w = canvas.width,
    h = canvas.height,
    pad = 32,
    s = Math.min((w - 2 * pad) / world.length, (h - 2 * pad) / world.width),
    ox = (w - world.length * s) / 2,
    oy = (h - world.width * s) / 2;
  const py = (y) => oy + (world.width - y) * s,
    az = world.alliance_zone_depth || 4.028;
  fx.clearRect(0, 0, w, h);
  fx.fillStyle = "#252e32";
  fx.fillRect(ox, oy, world.length * s, world.width * s);
  fx.fillStyle = "rgba(198,93,77,.18)";
  fx.fillRect(ox, oy, az * s, world.width * s);
  fx.fillStyle = "rgba(76,126,180,.18)";
  fx.fillRect(ox + (world.length - az) * s, oy, az * s, world.width * s);
  fx.strokeStyle = "#e1e6e3";
  fx.lineWidth = 2;
  fx.strokeRect(ox, oy, world.length * s, world.width * s);
  // FIRST field markings: alliance-zone boundaries and field centerline.
  fx.setLineDash([7, 6]);
  fx.lineWidth = 1.5;
  fx.strokeStyle = "#d8dedc";
  for (const x of [az, world.length / 2, world.length - az]) {
    fx.beginPath();
    fx.moveTo(ox + x * s, oy);
    fx.lineTo(ox + x * s, oy + world.width * s);
    fx.stroke();
  }
  fx.setLineDash([]);
  if (elements.showGrid.checked) {
    const cell = 0.25;
    fx.save();
    fx.beginPath();
    fx.rect(ox, oy, world.length * s, world.width * s);
    fx.clip();
    fx.strokeStyle = "rgba(177,202,216,.20)";
    fx.lineWidth = 1;
    for (let x = 0; x <= world.length + 1e-6; x += cell) {
      const px = ox + x * s;
      fx.beginPath();
      fx.moveTo(px, oy);
      fx.lineTo(px, oy + world.width * s);
      fx.stroke();
    }
    for (let y = 0; y <= world.width + 1e-6; y += cell) {
      const pyCell = oy + (world.width - y) * s;
      fx.beginPath();
      fx.moveTo(ox, pyCell);
      fx.lineTo(ox + world.length * s, pyCell);
      fx.stroke();
    }
    fx.restore();
  }
  fx.fillStyle = "#ED1C24";
  fx.font = "bold 12px system-ui";
  fx.fillText("RED ALLIANCE", ox + 10, oy + 17);
  fx.fillStyle = "#0066B3";
  fx.fillText("BLUE ALLIANCE", ox + world.length * s - 111, oy + 17);
  fx.fillStyle = "rgba(238,244,247,.72)";
  fx.font = "bold 10px system-ui";
  fx.textAlign = "center";
  fx.fillText("ALLIANCE ZONE", ox + (az * s) / 2, oy + world.width * s - 10);
  fx.fillText(
    "NEUTRAL ZONE",
    ox + (world.length * s) / 2,
    oy + world.width * s - 10,
  );
  fx.fillText(
    "ALLIANCE ZONE",
    ox + (world.length - az / 2) * s,
    oy + world.width * s - 10,
  );
  fx.textAlign = "start";
  fx.fillStyle = "#b4c1c9";
  fx.font = "bold 10px system-ui";
  fx.fillText("2026 FRC · REBUILT", ox, oy - 10);
  for (const b of world.elements || world.colliders || []) {
    const x = ox + (b.x - b.length / 2) * s,
      y = py(b.y + b.width / 2),
      bw = b.length * s,
      bh = b.width * s;
    // Show the tower footprint as an open frame. Its broad box is visual only;
    // the separate upright boxes remain the only ground-level colliders.
    if (b.name === "red_tower" || b.name === "blue_tower") {
      const red = b.name === "red_tower";
      fx.fillStyle = red ? "rgba(237,28,36,.08)" : "rgba(0,102,179,.08)";
      fx.strokeStyle = red ? "rgba(237,28,36,.9)" : "rgba(0,102,179,.9)";
      fx.lineWidth = 2;
      fx.setLineDash([6, 4]);
      fx.fillRect(x, y, bw, bh);
      fx.strokeRect(x, y, bw, bh);
      fx.setLineDash([]);
      fx.fillStyle = "#f2f5f5";
      fx.font = "bold 9px system-ui";
      fx.textAlign = "center";
      fx.textBaseline = "middle";
      fx.fillText("TOWER", x + bw / 2, y + bh / 2);
      fx.textAlign = "start";
      fx.textBaseline = "alphabetic";
      continue;
    }
    if (b.name?.includes("_trench_") && !b.name.includes("_support_")) {
      fx.fillStyle = "rgba(113,135,122,.10)";
      fx.fillRect(x, y, bw, bh);
      fx.save();
      fx.setLineDash([5, 5]);
      fx.strokeStyle = "#a9c3b2";
      fx.lineWidth = 1.5;
      fx.strokeRect(x, y, bw, bh);
      fx.restore();
      fx.strokeStyle = "#a9c3b2";
      fx.lineWidth = 3;
      fx.beginPath();
      fx.moveTo(x + bw / 2, y);
      fx.lineTo(x + bw / 2, y + bh);
      fx.stroke();
      fx.fillStyle = "#d8e2dc";
      fx.font = "bold 9px system-ui";
      fx.textAlign = "center";
      fx.textBaseline = "middle";
      fx.fillText("TRENCH", x + bw / 2, y + bh / 2);
      fx.textAlign = "start";
      fx.textBaseline = "alphabetic";
      continue;
    }
    fx.fillStyle = b.color?.includes("support")
      ? "#b09a72"
      : b.color?.includes("hub")
        ? "#707a7e"
        : b.color?.includes("bump")
          ? b.color.includes("red")
            ? "#ED1C24"
            : "#0066B3"
          : b.color?.includes("tower")
            ? b.color.includes("red")
              ? "#ED1C24"
              : "#0066B3"
            : b.color?.includes("depot")
              ? "#c1a96f"
              : "#59675d";
    fx.fillRect(x, y, bw, bh);
    fx.strokeStyle = "#e1e6e3";
    fx.lineWidth = b.color?.includes("support") ? 1.7 : 1.2;
    fx.strokeRect(x, y, bw, bh);
    if (b.name?.includes("hub")) {
      const cx = x + bw / 2,
        cy = y + bh / 2,
        r = Math.min(bw, bh) * 0.443;
      fx.beginPath();
      for (let i = 0; i < 6; i++) {
        const a = -Math.PI / 2 + (i * Math.PI) / 3,
          px = cx + r * Math.cos(a),
          pY = cy + r * Math.sin(a);
        i ? fx.lineTo(px, pY) : fx.moveTo(px, pY);
      }
      fx.closePath();
      fx.fillStyle = "#273238";
      fx.fill();
      fx.strokeStyle = "#f1d98d";
      fx.lineWidth = 2;
      fx.stroke();
      fx.fillStyle = "#f3e7bc";
      fx.font = "bold 10px system-ui";
      fx.textAlign = "center";
      fx.textBaseline = "middle";
      fx.fillText("72″", cx, cy);
      fx.textAlign = "start";
      fx.textBaseline = "alphabetic";
    } else if (/bump|depot/.test(b.name || "")) {
      const label = b.name.includes("bump")
        ? "BUMP"
        : "DEPOT";
      fx.fillStyle = "#f4f5ec";
      fx.font = "bold 9px system-ui";
      fx.textAlign = "center";
      fx.textBaseline = "middle";
      fx.fillText(label, x + bw / 2, y + bh / 2);
      fx.textAlign = "start";
      fx.textBaseline = "alphabetic";
    }
  }
  if (!frames.length) return;
  const f = currentFrame(),
    defensePlayback =
      playbackTask === "defense" || playbackTask === "adstar_attacker_defense",
    attackerIndex = defensePlayback ? 1 : 0,
    defenderIndex = 1 - attackerIndex;
  drawAdstarRoutes(fx, f, attackerIndex, 1, ox, s, py);

  if (Array.isArray(f.hub_centers)) {
    for (let i = 0; i < f.hub_centers.length; i++) {
      const hub = f.hub_centers[i],
        active = f.hub_active?.[i] !== false;
      fx.beginPath();
      fx.arc(
        ox + hub[0] * s,
        py(hub[1]),
        Math.max(4, 0.58 * s),
        0,
        Math.PI * 2,
      );
      fx.strokeStyle = active ? "#f3e7bc" : "#77818a";
      fx.lineWidth = 2;
      fx.stroke();
    }
  }
  const pieces = Array.isArray(f.fuel_pieces) ? f.fuel_pieces : [];
  // Transfers are instantaneous in the simulator, so infer release vectors
  // from ownership changes between adjacent playback frames.
  const releaseFrame = frames[Math.max(0, index - 1)];
  if (index > 0 && releaseFrame?.fuel_pieces?.length) {
    const before = releaseFrame.fuel_pieces;
    const scoreDelta = (f.fuel_score_count || []).map((score, team) =>
      Number(score) - Number(releaseFrame.fuel_score_count?.[team] || 0));
    let scoresRemaining = scoreDelta.reduce((sum, count) => sum + Math.max(0, count), 0);
    for (let id = 0; id < Math.min(before.length, pieces.length); id++) {
      const oldOwner = Number(before[id]?.[2]), newOwner = Number(pieces[id]?.[2]);
      if (!(oldOwner >= 0 && newOwner < 0)) continue;
      const robot = releaseFrame.robots?.[oldOwner];
      if (!robot) continue;
      const team = oldOwner < 3 ? 0 : 1;
      const scored = scoresRemaining > 0 && scoreDelta[team] > 0;
      const hub = f.hub_centers?.[team];
      const destination = scored && hub ? hub : pieces[id].slice(0, 2);
      const x1 = ox + robot[0] * s, y1 = py(robot[1]);
      const x2 = ox + destination[0] * s, y2 = py(destination[1]);
      const angle = Math.atan2(y2 - y1, x2 - x1);
      fx.save();
      fx.strokeStyle = scored ? "#ffd84d" : "#7de3ff";
      fx.fillStyle = fx.strokeStyle;
      fx.lineWidth = 3;
      fx.beginPath();
      fx.moveTo(x1, y1);
      fx.lineTo(x2, y2);
      fx.stroke();
      fx.translate(x2, y2);
      fx.rotate(angle);
      fx.beginPath();
      fx.moveTo(0, 0);
      fx.lineTo(-10, -5);
      fx.lineTo(-10, 5);
      fx.closePath();
      fx.fill();
      fx.restore();
      if (scored) {
        scoreDelta[team]--;
        scoresRemaining--;
      }
    }
  }
  const hopperLoads = Array(f.robots.length).fill(0);
  for (const piece of pieces) {
    const owner = piece?.[2];
    if (Number.isInteger(owner) && owner >= 0 && owner < hopperLoads.length)
      hopperLoads[owner]++;
  }
  const scores = Array.isArray(f.fuel_score_count) ? f.fuel_score_count : [];
  elements.redScore.textContent = Number.isFinite(Number(scores[0]))
    ? integer(scores[0]) : "—";
  elements.blueScore.textContent = Number.isFinite(Number(scores[1]))
    ? integer(scores[1]) : "—";
  const elapsed = Number(f.match_elapsed);
  if (Number.isFinite(elapsed)) {
    const seconds = Math.max(0, Math.floor(elapsed));
    elements.matchClock.textContent = `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`;
  } else {
    elements.matchClock.textContent = "—:—";
  }
  let loose = 0,
    heldRed = 0,
    heldBlue = 0;
  for (const piece of pieces) {
    if (!piece || piece.length < 3) continue;
    const owner = piece[2],
      held = owner >= 0;
    if (owner === -1) loose++;
    else if (owner < 3) heldRed++;
    else heldBlue++;
    if (!elements.showFuel.checked) continue;
    fx.beginPath();
    fx.arc(ox + piece[0] * s, py(piece[1]), held ? 5.2 : 4.4, 0, Math.PI * 2);
    fx.fillStyle = owner < 0 ? "#ffd84d" : robotAllianceColor(f, owner);
    fx.fill();
    fx.strokeStyle = "#18222a";
    fx.lineWidth = 1.5;
    fx.stroke();
  }
  elements.gameState.textContent =
    `FUEL in frame: ${pieces.length} active · loose ${loose} · held Red ${heldRed} / Blue ${heldBlue}${f.match_remaining != null ? ` · ${Number(f.match_remaining).toFixed(1)} s remaining` : ""}`;
  if (f.predicted_intercept) {
    const x = ox + f.predicted_intercept[0] * s,
      y = py(f.predicted_intercept[1]);
    fx.save();
    fx.translate(x, y);
    fx.rotate(Math.PI / 4);
    fx.fillStyle = "#ffe071";
    fx.strokeStyle = "#302a16";
    fx.lineWidth = 1.5;
    fx.fillRect(-7, -7, 14, 14);
    fx.strokeRect(-7, -7, 14, 14);
    fx.restore();
    fx.fillStyle = "#ffe071";
    fx.font = "12px system-ui";
    fx.fillText(
      `intercept ${Number(f.predicted_intercept_time || 0).toFixed(1)}s`,
      x + 10,
      y - 9,
    );
  }
  const trailFrames = frames.slice(Math.max(0, index - 109), index + 1);
  if (elements.showGhosts.checked) {
    const selectedId = elements.scenario.value,
      progress = frames.length > 1 ? index / (frames.length - 1) : 0;
    for (const scenario of loadedScenarios) {
      if (scenario.id === selectedId || !scenario.frames?.length) continue;
      const ghostFrames = scenario.frames,
        ghostIndex = Math.min(
          ghostFrames.length - 1,
          Math.round(progress * Math.max(0, ghostFrames.length - 1)),
        ),
        ghost = ghostFrames[ghostIndex];
      if (!ghost?.robots) continue;
      // Each ghost keeps its own motion history and planner route, synchronized
      // to the selected run by normalized playback progress.
      const ghostTrail = ghostFrames.slice(
        Math.max(0, ghostIndex - 109),
        ghostIndex + 1,
      );
      for (let k = 0; k < ghost.robots.length; k++) {
        const color = robotAllianceColor(ghost, k, attackerIndex);
        fx.save();
        fx.globalAlpha = 0.45;
        fx.strokeStyle = color;
        fx.lineWidth = 2;
        fx.beginPath();
        let started = false;
        for (const frame of ghostTrail) {
          const point = frame.robots?.[k];
          if (!point) continue;
          const x = ox + point[0] * s,
            y = py(point[1]);
          if (!started) {
            fx.moveTo(x, y);
            started = true;
          } else fx.lineTo(x, y);
        }
        fx.stroke();
        fx.restore();
      }
      drawAdstarRoutes(fx, ghost, attackerIndex, 0.68, ox, s, py);
      for (let k = 0; k < ghost.robots.length; k++) {
        const r = ghost.robots[k],
          [l, b] =
            ghost.sizes?.length >= k * 2 + 2
              ? ghost.sizes.slice(k * 2, k * 2 + 2)
              : f.sizes.slice(k * 2, k * 2 + 2),
          color = robotAllianceColor(ghost, k, attackerIndex);
        fx.save();
        fx.globalAlpha = 0.52;
        fx.translate(ox + r[0] * s, py(r[1]));
        fx.rotate(-r[2]);
        fx.fillStyle = color + "35";
        fx.strokeStyle = color;
        fx.lineWidth = 2;
        fx.setLineDash([5, 4]);
        fx.fillRect((-l * s) / 2, (-b * s) / 2, l * s, b * s);
        fx.strokeRect((-l * s) / 2, (-b * s) / 2, l * s, b * s);
        fx.setLineDash([]);
        fx.beginPath();
        fx.moveTo(0, 0);
        fx.lineTo(l * s * 0.48, 0);
        fx.stroke();
        fx.restore();
      }
    }
  }
  for (let k = 0; k < f.robots.length; k++) {
    const color = robotAllianceColor(f, k, attackerIndex);
    let segment = [];
    const drawSegment = () => {
      if (segment.length > 1) {
        fx.beginPath();
        segment.forEach((p, i) => {
          const x = ox + p[0] * s,
            y = py(p[1]);
          i ? fx.lineTo(x, y) : fx.moveTo(x, y);
        });
        fx.stroke();
      }
    };
    fx.strokeStyle = color + "88";
    fx.lineWidth = 2;
    for (const frame of trailFrames) {
      const point = frame.robots?.[k]?.slice(0, 2);
      if (!point || point.length < 2) continue;
      const previous = segment[segment.length - 1];
      if (
        previous &&
        Math.hypot(point[0] - previous[0], point[1] - previous[1]) > 0.8
      ) {
        drawSegment();
        segment = [];
      }
      segment.push(point);
    }
    drawSegment();
  }
  f.robots.forEach((r, k) => {
    let vectors = f.robot_effort_vectors;
    let v = Array.isArray(vectors?.[k])
      ? vectors[k]
      : k === defenderIndex
        ? f.chassis_effort_vector
        : null;
    if (!Array.isArray(v) || v.length < 2 || !v.every(Number.isFinite)) {
      const before = frames[Math.max(0, index - 1)]?.robots?.[k],
        after = frames[Math.min(frames.length - 1, index + 1)]?.robots?.[k];
      v = before && after ? [after[0] - before[0], after[1] - before[1]] : [0, 0];
      v = v.map((x) => x * 3);
    }
    const mag = Math.hypot(v[0], v[1]);
    if (mag > 0.03) {
      const len = Math.max(0.32, Math.min(1.15, mag * 0.28)) * s,
        dx = (v[0] / mag) * len,
        dy = (-v[1] / mag) * len,
        // Move the effort arrow beside the robot so its short shaft and tip
        // remain visible instead of being hidden by the robot body.
        offset = 0.28 * s,
        x = ox + r[0] * s - (dy / len) * offset,
        y = py(r[1]) + (dx / len) * offset,
        ang = Math.atan2(dy, dx);
      fx.save();
      fx.strokeStyle = "#FFFFFF";
      fx.fillStyle = "#FFFFFF";
      fx.lineWidth = 4;
      fx.lineCap = "round";
      fx.beginPath();
      fx.moveTo(x, y);
      fx.lineTo(x + dx, y + dy);
      fx.stroke();
      fx.translate(x + dx, y + dy);
      fx.rotate(ang);
      fx.beginPath();
      fx.moveTo(0, 0);
      fx.lineTo(-11, -6);
      fx.lineTo(-11, 6);
      fx.closePath();
      fx.fill();
      fx.restore();
    }
  });
  f.robots.forEach((r, k) => {
    let [x, y, t] = r,
      [l, b] = f.sizes.slice(k * 2, k * 2 + 2);
    fx.save();
    fx.translate(ox + x * s, py(y));
    fx.rotate(-t);
    fx.fillStyle = robotAllianceColor(f, k, attackerIndex);
    fx.fillRect((-l * s) / 2, (-b * s) / 2, l * s, b * s);
    fx.lineJoin = "round";
    fx.strokeStyle = "#10151b";
    fx.lineWidth = 7;
    fx.strokeRect((-l * s) / 2, (-b * s) / 2, l * s, b * s);
    fx.strokeStyle = "#f4f7f6";
    fx.lineWidth = 2.5;
    fx.strokeRect((-l * s) / 2, (-b * s) / 2, l * s, b * s);
    const robotType = playbackRobotTypes[k] === "turret" ? "T" : "D";
    fx.fillStyle = "#10151b";
    fx.beginPath();
    fx.arc(0, 0, Math.min(10, b * s * 0.3), 0, Math.PI * 2);
    fx.fill();
    fx.fillStyle = "#ffffff";
    fx.font = "bold 10px system-ui";
    fx.textAlign = "center";
    fx.textBaseline = "middle";
    fx.fillText(robotType, 0, 0);
    const capacity = Math.max(1, Number(playbackFuelCapacities[k]) ||
      (playbackRobotTypes[k] === "turret" ? 40 : 60));
    const load = Math.min(1, hopperLoads[k] / capacity);
    const pieX = -l * s * 0.31, pieY = -b * s * 0.31, pieR = 6;
    fx.beginPath();
    fx.arc(pieX, pieY, pieR, 0, Math.PI * 2);
    fx.fillStyle = "#18222a";
    fx.fill();
    if (load > 0) {
      fx.beginPath();
      fx.moveTo(pieX, pieY);
      fx.arc(pieX, pieY, pieR - 1, -Math.PI / 2,
        -Math.PI / 2 + load * Math.PI * 2);
      fx.closePath();
      fx.fillStyle = "#ffd84d";
      fx.fill();
    }
    fx.strokeStyle = "#ffffff";
    fx.lineWidth = 1;
    fx.beginPath();
    fx.arc(pieX, pieY, pieR, 0, Math.PI * 2);
    fx.stroke();
    const action = Number(f.robot_actions?.[k]);
    let behavior = "";
    if (Number.isInteger(action)) {
      if (action < 4) behavior = "C";
      else if (action === 4) behavior = "S";
      else if (action === 5) {
        behavior = f.robot_roles?.[k] === "defense"
          ? (f.robot_opponent_visible?.[k] ? "I" : "G") : "D";
      }
      else if (action === 6) {
        const deterministicOffense = f.robot_control_modes?.[k] === "deterministic" &&
          f.robot_roles?.[k] === "offense";
        behavior = hopperLoads[k] > 0 || !deterministicOffense ? "F" : "E";
      }
    }
    if (behavior) {
      const behaviorX = l * s * 0.31, behaviorY = b * s * 0.31;
      fx.beginPath();
      fx.arc(behaviorX, behaviorY, 6, 0, Math.PI * 2);
      fx.fillStyle = "#18222a";
      fx.fill();
      fx.strokeStyle = "#ffffff";
      fx.lineWidth = 1;
      fx.stroke();
      fx.fillStyle = "#ffffff";
      fx.font = "bold 8px system-ui";
      fx.textAlign = "center";
      fx.textBaseline = "middle";
      fx.fillText(behavior, behaviorX, behaviorY);
    }
    fx.beginPath();
    fx.moveTo(l * s * 0.43, -b * s * 0.32);
    fx.lineTo(l * s * 0.43, b * s * 0.32);
    fx.strokeStyle = "#161b1e";
    fx.lineWidth = 7;
    fx.stroke();
    fx.strokeStyle = "#ffd84d";
    fx.lineWidth = 4;
    fx.stroke();
    fx.restore();
  });
  elements.frame.max = Math.max(0, frames.length - 1);
  elements.frame.value = index;
  elements.frameCount.textContent = frames.length
    ? `${index + 1} / ${frames.length} · ${playbackSimTime.toFixed(1)} / ${playbackTimes[playbackTimes.length - 1]?.toFixed(1) || "0.0"} s`
    : "Waiting for rollout frames";
}
