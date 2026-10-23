(() => {
  const esc = value => String(value ?? "—").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const pretty = value => esc(JSON.stringify(value ?? {}, null, 2));
  const updated = document.querySelector("#updated");
  async function refresh() {
    const response = await fetch(`/api/meco-status?t=${Date.now()}`, {cache:"no-store"});
    if (!response.ok) throw new Error(`Status endpoint returned ${response.status}`);
    const data = await response.json();
    updated.textContent = `Updated ${data.generated_utc || "unknown"}`;
    updated.dataset.stale = "false";
    const activeLanes = Object.entries(data.queue?.lanes || {})
      .filter(([, lane]) => lane.pid).map(([track, lane]) => `${track} pid ${lane.pid}`);
    const queueEval = data.queue?.evaluation || {};
    document.querySelector("#queue").textContent =
      `Queue: ${data.queue?.state || "unavailable"} · ${data.queue?.phase || "no phase"} · cycle ${data.queue?.cycle ?? "—"}`+
      `${activeLanes.length ? ` · workers ${activeLanes.join(", ")}` : ""}`+
      ` · evaluator ${queueEval.status || "idle"}${queueEval.pid ? ` pid ${queueEval.pid}` : ""} (queue is separate from worker liveness)`;
    document.querySelector("#policies").innerHTML = (data.policies || []).map(p => {
      const run = p.run || {}, evidence = p.latest_evidence || {};
      return `<tr><td><strong>${esc(p.policy_id)}</strong><small>${esc(p.kind)}</small></td>`+
        `<td><span class="state" data-state="${esc(p.training_state)}">${esc(p.training_state)}</span>`+
        `<small>${esc(run.completed_updates ?? "")} updates · heartbeat ${esc(run.heartbeat?.state || "n/a")}</small></td>`+
        `<td><span class="state">${esc(p.evaluation_state)}</span></td>`+
        `<td>${esc(run.run_id || "—")}<small>Evidence ${esc(evidence.evaluation_id || "—")}</small></td>`+
        `<td>${esc(run.latest_checkpoint?.path || "—")}<small>${esc(run.latest_checkpoint?.bytes ?? "")} bytes</small></td>`+
        `<td><small>Source ${esc(run.source_identity_sha256 || evidence.source_hash || "—")}</small>`+
        `<small>Policy ${esc(evidence.policy_manifest?.policy_id || "—")}</small>`+
        `<small>Eval weights ${esc(evidence.policy_manifest?.checkpoint_sha256 || "deterministic")}</small>`+
        `<small>Protocol ${esc(evidence.protocol_id || "—")}</small></td>`+
        `<td><small>${pretty(evidence.records || run.latest_training_metrics || {})}</small></td></tr>`;
    }).join("");
    const groups = data.comparison_groups || [];
    const evaluatorRuns = data.evaluation_runs || [];
    document.querySelector("#evaluators").innerHTML = evaluatorRuns.length ? evaluatorRuns.map(run =>
      `<article class="group"><h3>${esc(run.status)} · ${esc(run.evaluation_id)}</h3>`+
      `<p>${esc(run.progress?.completed_records ?? 0)} / ${esc(run.progress?.total_records ?? "?")} records</p>`+
      `<small>Heartbeat ${esc(run.heartbeat_state)} · ${esc(run.heartbeat_age_seconds ?? "?")}s · process ${run.process_alive ? "alive" : "not confirmed"}</small>`+
      `<p>Protocol ${esc(run.protocol_id)} · split ${esc(run.split)} · output ${esc(run.output_file)}</p></article>`).join("") : "<p class='muted'>No evaluator status manifest is available.</p>";
    document.querySelector("#groups").innerHTML = groups.length ? groups.map(g =>
      `<article class="group"><h3>${esc(g.scenario_id)} · ${esc(g.matched_status)}</h3>`+
      `<p>${esc(g.evaluation_ids?.join(", "))}</p><small>Protocol ${esc(g.protocol_id)} · ${esc(g.record_count)} records per policy</small>`+
      `<pre>${pretty(g.policy_metrics)}</pre></article>`).join("") : "<p class='muted'>No fully matched, validated comparison is available.</p>";
    const errors = data.errors || [];
    document.querySelector("#errors").innerHTML = errors.length ? errors.map(e => `<li>${esc(e.source || "status")}: ${esc(e.error)}</li>`).join("") : "<li>None reported.</li>";
  }
  refresh().catch(error => { updated.textContent = `Status stale: ${error.message}`; updated.dataset.stale = "true"; });
  window.setInterval(() => refresh().catch(error => {
    updated.textContent = `Status stale: ${error.message}`;
    updated.dataset.stale = "true";
  }), 15000);
})();
