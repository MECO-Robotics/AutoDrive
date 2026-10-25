(() => {
  const esc = value => String(value ?? "—").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
  const pretty = value => esc(JSON.stringify(value ?? {}, null, 2));
  const updated = document.querySelector("#updated");
  async function refresh() {
    const response = await fetch(`/api/meco-status?t=${Date.now()}`, {cache:"no-store"});
    if (!response.ok) throw new Error(`Status endpoint returned ${response.status}`);
    const data = await response.json();
    let provenance;
    try {
      const evidenceResponse = await fetch(`/api/curriculum-status?t=${Date.now()}`, {cache:"no-store"});
      if (!evidenceResponse.ok) throw new Error(`Evidence endpoint returned ${evidenceResponse.status}`);
      provenance = (await evidenceResponse.json()).three_pass_provenance;
    } catch (error) {
      provenance = {status:"untrusted", tranches:[], untrusted_reason:error.message};
    }
    updated.textContent = `Updated ${data.generated_utc || "unknown"}`;
    updated.dataset.stale = "false";
    document.querySelector("#sources").textContent =
      `Live data ${data.data_root || "unavailable"} · checkpoints ${data.checkpoint_root || "unavailable"} · evaluations ${data.evaluation_root || "unavailable"} · three-pass ledger ${data.campaign_ledger_path || "unavailable"}`;
    const activeLanes = Object.entries(data.queue?.lanes || {})
      .filter(([, lane]) => lane.pid).map(([track, lane]) =>
        `${track} pid ${lane.pid} ${lane.process_confirmed ? "confirmed" : "unconfirmed"}`+
        `${lane.process_state ? ` OS ${lane.process_state}` : " OS unknown"}`);
    const queuePhase = data.queue?.phase;
    const cycle = Number(data.queue?.cycle);
    const isAdvancingCycle = data.queue?.state === "fresh" &&
      ["training", "catching_up"].includes(queuePhase) &&
      Number.isInteger(cycle) && cycle > 0;
    const laneSummary = Object.entries(data.queue?.lanes || {}).map(([track, lane]) => {
      const updates = lane.completed_updates;
      const hasKnownUpdates = typeof updates === "number" &&
        Number.isFinite(updates) && Number.isInteger(updates) && updates >= 0;
      const chunk = Number(lane.chunk_updates);
      const target = isAdvancingCycle && Number.isFinite(chunk) && chunk > 0
        ? cycle * chunk : null;
      const status = data.queue?.state !== "fresh" ? "unconfirmed" :
        lane.status === "complete" && hasKnownUpdates && target !== null && updates < target
          ? "queued" : lane.worker_state || lane.status || "unknown";
      return `${track} ${status} ${hasKnownUpdates ? updates : "—"}${target === null || !hasKnownUpdates ? "" : `/${target}`}`;
    });
    const queueEval = data.queue?.evaluation || {};
    const evalSchedule = data.queue?.evaluation_schedule || {};
    document.querySelector("#queue").textContent =
      `Queue file: ${data.queue?.state || "unavailable"} · workers ${data.queue?.worker_state || "unknown"} · ${queuePhase || "no phase"} · cycle ${data.queue?.cycle ?? "—"}`+
      `${activeLanes.length ? ` · workers ${activeLanes.join(", ")}` : ""}`+
      ` · evaluator ${queueEval.status || "idle"}${queueEval.pid ? ` pid ${queueEval.pid}` : ""}`+
      `${Number.isInteger(evalSchedule.schedule_count) ? ` · scheduled evals ${evalSchedule.completed_count ?? 0}/${evalSchedule.schedule_count} complete, ${evalSchedule.pending_count ?? 0} pending, ${evalSchedule.failed_count ?? 0} failed attempts` : ""}`+
      ` (queue is separate from worker liveness)`;
    document.querySelector("#lane-queue").textContent = laneSummary.length
      ? `Tracks: ${laneSummary.join(" · ")}` : "No track queue data";
    const monitor = data.queue?.diagnostic_monitor;
    const monitorLine = document.querySelector("#diagnostic-monitor");
    if (!monitor || monitor.state === "not_configured" || !monitor.configured) {
      monitorLine.textContent = "Not configured. Diagnostic monitor output is diagnostic-only and never qualification eligible.";
    } else if (monitor.state !== "fresh" && monitor.state !== "stale") {
      monitorLine.textContent = `Configured, but its saved summary is untrusted. Qualification eligibility is never granted by this monitor. Error: ${monitor.error || "summary could not be validated"}`;
    } else {
      const confirmed = data.queue?.state === "fresh" && monitor.state === "fresh" && monitor.fresh === true;
      const pending = confirmed ? monitor.pending_frontier : null;
      const pendingText = pending
        ? `pending ${pending.frontier_id} · cycle ${pending.cycle} · ${pending.status}${Number.isInteger(pending.attempt) ? ` · attempt ${pending.attempt}` : ""}`
        : confirmed ? "no pending frontier" : "pending frontier unconfirmed";
      const freshnessText = confirmed ? "Current queue snapshot" : "Last saved snapshot is stale; progress is unconfirmed";
      const errorText = monitor.error ? ` · Error: ${monitor.error}` : "";
      monitorLine.textContent = `${monitor.configured ? "Configured" : "Not configured"} · ${freshnessText} · diagnostic-only · never qualification eligible · completed ${monitor.completed_count ?? "unknown"} · ${pendingText} · skipped ${monitor.skipped_count ?? 0} · missed ${monitor.missed_count ?? 0}${errorText}`;
    }
    document.querySelector("#policies").innerHTML = (data.policies || []).map(p => {
      const run = p.run || {}, evidence = p.latest_evidence || {};
      return `<tr><td><strong>${esc(p.policy_id)}</strong><small>${esc(p.kind)}</small></td>`+
        `<td><span class="state" data-state="${esc(p.training_state)}">${esc(p.training_state)}</span>`+
        `<small>${esc(run.completed_updates ?? "")} updates · heartbeat ${esc(run.heartbeat?.state || "n/a")} · `+
        `age ${esc(run.heartbeat?.age_seconds ?? "?")}s · OS ${esc(run.heartbeat?.process_state || "unknown")}</small></td>`+
        `<td><span class="state">${esc(p.evaluation_state)}</span></td>`+
        `<td>${esc(run.run_id || "—")}<small>Evidence ${esc(evidence.evaluation_id || "—")}</small></td>`+
        `<td>${esc(run.latest_checkpoint?.path || "—")}<small>${esc(run.latest_checkpoint?.bytes ?? "")} bytes</small></td>`+
        `<td><small>Source ${esc(run.source_identity_sha256 || evidence.source_hash || "—")}</small>`+
      `<small>Policy ${esc(evidence.policy_manifest?.policy_id || "—")}</small>`+
      `<small>Eval weights ${esc(evidence.policy_manifest?.checkpoint_sha256 || "deterministic")}</small>`+
      `<small>Protocol ${esc(evidence.protocol_id || "—")}</small>`+
      `<small>Candidate contract ${pretty({pool: run.candidate_boundary?.selectable_pool,
        oracle: run.candidate_boundary?.reference_pool,
        commitment: run.candidate_boundary?.continue_gap})}</small></td>`+
        `<td><small>Route P provenance ${pretty(evidence.route_cost_provenance || {})}</small>`+
        `<small>${pretty(evidence.records || run.latest_training_metrics || {})}</small></td></tr>`;
    }).join("");
    const groups = data.comparison_groups || [];
    const evaluatorRuns = data.evaluation_runs || [];
    document.querySelector("#evaluators").innerHTML = evaluatorRuns.length ? evaluatorRuns.map(run => {
      const lifecycle = run.execution_status || run.status || "unconfirmed/invalid";
      const evaluatorLabel = run.diagnostic === true && lifecycle === "status-reported complete"
        ? "status-reported complete · diagnostic · report unvalidated"
        : lifecycle;
      const evidenceError = run.evidence_error || run.error;
      return `<article class="group"><h3>${esc(evaluatorLabel)}</h3>`+
      `<small>Status-file evaluation ${esc(run.evaluation_id)}</small>`+
      `<p>${esc(run.progress?.completed_records ?? 0)} / ${esc(run.progress?.total_records ?? "?")} records</p>`+
      `<small>Heartbeat ${esc(run.heartbeat_state)} · ${esc(run.heartbeat_age_seconds ?? "?")}s · process ${run.process_alive ? "alive" : "not confirmed"}</small>`+
      `<p>Protocol ${esc(run.protocol_id)} · split ${esc(run.split)} · output ${esc(run.output_file)}</p>`+
      `<small>Report validation ${esc(run.report_validation || "unconfirmed")}</small>`+
      `${evidenceError ? `<p>Evidence error: ${esc(evidenceError)}</p>` : ""}</article>`;
    }).join("") : "<p class='muted'>No evaluator status manifest is available.</p>";
    const workflow = data.implementation_passes;
    const passStatus = workflow?.passes?.length ? workflow.passes.map(pass =>
      `<article class="group"><h3>${esc(pass.pass)} · ${esc(pass.status)}</h3>`+
      `<small>Workflow status only; this does not report agent liveness. Provenance: ${esc((pass.provenance || []).join(" · "))}</small></article>`).join("") :
      "<p class='muted'>No campaign pass-status records are available.</p>";
    const roleEvidence = (name, role) => {
      if (!role) return `<li>${esc(name)}: missing</li>`;
      const artifact = role.artifact || {};
      const commit = name === "implementer" ?
        Object.entries(role).filter(([key]) => key === "commit" || key.endsWith("_commit"))
          .map(([key, value]) => `${key} ${value}`).join(" · ") :
        name === "reviewer" ? `approved ${role.approved_commit || "—"}` : "";
      const paths = name === "implementer" ? (role.paths || []).join(", ") : "";
      return `<li><strong>${esc(name)}</strong> · ${esc(role.status || "unknown")}`+
        `${commit ? `<small>${esc(commit)}</small>` : ""}`+
        `${artifact.path ? `<small>${esc(artifact.kind)} · ${esc(artifact.path)} · SHA-256 ${esc(artifact.sha256)}</small>` : ""}`+
        `${paths ? `<small>Changed paths ${esc(paths)}</small>` : ""}`+
        `${role.verdict_detail ? `<small>Review ${esc(role.verdict)}: ${esc(role.verdict_detail)}</small>` : ""}</li>`;
    };
    const provenanceHtml = provenance?.validation_status === "validated_summary_only"
      ? `<p class="muted">Validated summary-only provenance. This is not whole-plan approval or agent-liveness evidence.</p>`+
        `${(provenance.tranches || []).map(tranche =>
          `<article class="group"><h3>${esc(tranche.title)} · ${esc(tranche.requirement_id)}</h3>`+
          `<small>Evidence class ${esc(provenance.evidence_class)} · validation ${esc(provenance.validation_status)}</small>`+
          `<ul>${roleEvidence("planner", tranche.planner)}${roleEvidence("implementer", tranche.implementer)}${roleEvidence("reviewer", tranche.reviewer)}</ul>`+
          `<p>Limits: ${esc(tranche.limits)}</p></article>`).join("")}`+
        `<p class="muted">${(provenance.limits || []).map(esc).join(" · ")}</p>`
      : `<p class="muted">Three-pass provenance is untrusted: ${esc(provenance?.untrusted_reason || "no validated evidence payload")}</p>`;
    document.querySelector("#passes").innerHTML = `<h3>Campaign pass status</h3>${passStatus}`+
      `<h3>Scoped implementation evidence</h3>${provenanceHtml}`;
    document.querySelector("#phase4-factors").innerHTML = workflow?.phase4_factors?.length
      ? workflow.phase4_factors.map(factor =>
        `<article class="group"><h3>${esc(factor.factor_id)} · ${esc(factor.status)}</h3>`+
        `<p>${esc(factor.summary)}</p><small>Planner ${esc(factor.planner_status)} · `+
        `implementation ${esc(factor.implementation_status || factor.status)} · `+
        `review ${esc(factor.review_status)} · evaluation ${esc(factor.evaluation_status)}</small></article>`).join("")
      : "<p class='muted'>No Phase 4 factor plan is recorded.</p>";
    const qualification = data.qualification;
    document.querySelector("#qualification").innerHTML = qualification ?
      `<article class="group"><h3>${esc(qualification.scenario_id)} · ${esc(qualification.split)} · ${esc(qualification.failure ? "round failed / streak reset" : qualification.qualified ? (qualification.gate_calibration_status === "calibrated_baseline_frozen" ? "qualified under calibrated gates" : "qualified under provisional gates") : qualification.promotion?.passed ? "streaking" : "not qualified")}</h3>`+
      `<p>Policy ${esc(qualification.candidate_policy)} · streak ${esc(qualification.streak?.count ?? "—")} / 3 · ${esc(qualification.evaluation_id)}</p>`+
      `<small>Protocol ${esc(qualification.protocol_id)} · bundle ${esc(qualification.bundle_id)}</small>`+
      `<small>Checkpoint ${esc(qualification.checkpoint_sha256 || "—")} · final-test isolated ${qualification.final_test_isolated ? "yes" : "no"}</small>`+
      `<small>Gate calibration ${esc(qualification.gate_calibration_status || "unknown")}</small>`+
      `<small>Protocol ${esc(qualification.protocol_version || "unknown")} · calibration ${esc(qualification.calibration_id || "not frozen")}</small>`+
      `<small>Calibration reports ${pretty(qualification.calibration_ids || {})}</small>`+
      `<small>Retention scenarios ${esc((qualification.retention_scenario_ids || []).join(", ") || "none")}</small>`+
      `<small>Safety gate ${esc(qualification.safety_gate_status || "unknown")}</small>`+
      `<small>Gate ${pretty(qualification.gate_definition || {})}</small>`+
      `<pre>${pretty({failure: qualification.failure, promotion: qualification.promotion, paired_clustered_confidence_intervals: qualification.confidence_intervals})}</pre></article>` :
      "<p class='muted'>No formal qualification round is recorded.</p>";
    document.querySelector("#groups").innerHTML = groups.length ? groups.map(g => {
      const boutRows = Object.entries(g.policy_metrics || {}).flatMap(([policy, categories]) =>
        Object.entries(categories || {}).map(([category, metrics]) => {
          const bouts = metrics?.collection_action_bouts;
          const proportion = bouts?.acquired_proportion;
          const shown = typeof proportion === "number" && Number.isFinite(proportion)
            ? `${(proportion * 100).toFixed(1)}%`
            : bouts?.count === 0 ? "N/A (zero bouts)" : "unavailable";
          const unsuccessfulCoverage = `${bouts?.unsuccessful_collection_action_bouts_available_records ?? 0}/${bouts?.unsuccessful_collection_action_bouts_total_records ?? bouts?.available_records ?? 0} records`;
          const unsuccessfulShown = bouts?.unsuccessful_collection_action_bouts == null
            ? `unavailable (${unsuccessfulCoverage})`
            : `${bouts.unsuccessful_collection_action_bouts} (${unsuccessfulCoverage})`;
          return `<tr><th scope="row">${esc(policy)}</th><td>${esc(category)}</td>`+
            `<td>${esc(bouts?.available_records ?? "—")}</td>`+
            `<td>${esc(bouts?.acquired_count ?? "—")} / ${esc(bouts?.count ?? "—")}</td>`+
            `<td>${esc(unsuccessfulShown)}</td>`+
            `<td>${esc(shown)}</td></tr>`;
        })).join("");
      return `<article class="group"><h3>${esc(g.scenario_id)} · ${esc(g.matched_status)}</h3>`+
      `<p>${esc(g.evaluation_ids?.join(", "))}</p><small>Protocol ${esc(g.protocol_id)} · ${esc(g.record_count)} records per policy</small>`+
      `<small>Candidate contract ${pretty({pool: g.candidate_boundary?.selectable_pool,
        oracle: g.candidate_boundary?.reference_pool,
        commitment: g.candidate_boundary?.continue_gap})}</small>`+
      `<h4>Collection action bouts — descriptive bout outcomes</h4>`+
      `<p class="muted">Counts include available episode records only. Unsuccessful bouts are finalized bouts whose authoritative acquired quantity is zero; this is a bout outcome, not causal intake failure.</p>`+
      `<div class="table-wrap"><table><thead><tr><th>Policy</th><th>Category</th><th>Available records</th><th>Acquired / bouts</th><th>Unsuccessful collection action bouts</th><th>Acquired proportion</th></tr></thead><tbody>${boutRows}</tbody></table></div>`+
      `<pre>${pretty(g.policy_metrics)}</pre></article>`;
    }).join("") : "<p class='muted'>No fully matched, validated comparison is available.</p>";
    const errors = data.errors || [];
    document.querySelector("#errors").innerHTML = errors.length ? errors.map(e => `<li>${esc(e.source || "status")}: ${esc(e.error)}</li>`).join("") : "<li>None reported.</li>";
  }
  refresh().catch(error => { updated.textContent = `Status stale: ${error.message}`; updated.dataset.stale = "true"; });
  window.setInterval(() => refresh().catch(error => {
    updated.textContent = `Status stale: ${error.message}`;
    updated.dataset.stale = "true";
  }), 15000);
})();
