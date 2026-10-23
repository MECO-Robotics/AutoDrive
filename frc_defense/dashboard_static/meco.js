(() => {
  const esc = value => String(value ?? "—").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const pretty = value => esc(JSON.stringify(value ?? {}, null, 2));
  async function refresh() {
    const response = await fetch(`/api/meco-status?t=${Date.now()}`, {cache:"no-store"});
    if (!response.ok) throw new Error(`Status endpoint returned ${response.status}`);
    const data = await response.json();
    document.querySelector("#updated").textContent = `Updated ${data.generated_utc || "unknown"}`;
    document.querySelector("#queue").textContent = `Queue manifest: ${data.queue?.state || "unavailable"} (separate from worker liveness)`;
    document.querySelector("#policies").innerHTML = (data.policies || []).map(p => {
      const run = p.run || {}, evidence = p.latest_evidence || {};
      return `<tr><td><strong>${esc(p.policy_id)}</strong><small>${esc(p.kind)}</small></td>`+
        `<td><span class="state" data-state="${esc(p.training_state)}">${esc(p.training_state)}</span>`+
        `<small>${esc(run.completed_updates ?? "")} updates · heartbeat ${esc(run.heartbeat?.state || "n/a")}</small></td>`+
        `<td><span class="state">${esc(p.evaluation_state)}</span></td>`+
        `<td>${esc(run.run_id || "—")}<small>Evidence ${esc(evidence.evaluation_id || "—")}</small></td>`+
        `<td>${esc(run.latest_checkpoint?.path || "—")}<small>${esc(run.latest_checkpoint?.bytes ?? "")} bytes</small></td>`+
        `<td><small>Source ${esc(run.source_identity_sha256 || evidence.source_hash || "—")}</small>`+
        `<small>Protocol ${esc(evidence.protocol_id || "—")}</small></td>`+
        `<td><small>${pretty(evidence.records?.[0] || run.latest_training_metrics || {})}</small></td></tr>`;
    }).join("");
    const groups = data.comparison_groups || [];
    document.querySelector("#groups").innerHTML = groups.length ? groups.map(g =>
      `<article class="group"><h3>${esc(g.scenario_id)} · ${g.decision_grade ? "decision grade" : "diagnostic"}</h3>`+
      `<p>${esc(g.evaluation_ids?.join(", "))}</p><small>Protocol ${esc(g.protocol_id)} · ${esc(g.record_count)} records per policy</small>`+
      `<pre>${pretty(g.policy_metrics)}</pre></article>`).join("") : "<p class='muted'>No fully matched, validated comparison is available.</p>";
    const errors = data.errors || [];
    document.querySelector("#errors").innerHTML = errors.length ? errors.map(e => `<li>${esc(e.source || "status")}: ${esc(e.error)}</li>`).join("") : "<li>None reported.</li>";
  }
  refresh().catch(error => { document.querySelector("#updated").textContent = `Status unavailable: ${error.message}`; });
  window.setInterval(() => refresh().catch(() => {}), 15000);
})();
