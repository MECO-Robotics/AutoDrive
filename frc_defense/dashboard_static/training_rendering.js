// Rendering helpers for training metrics and controller comparisons.
// DOM elements and formatting are supplied by the dashboard so this module
// stays independent of dashboard state and global lookups.
export function renderTrainingChart({ canvas, rows, selectedMetric }) {
  const context = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const metricValueFor = (row) => {
    if (selectedMetric === "transitions_per_second")
      return row.strategic_transitions_per_second ?? row.transitions_per_second;
    return row[selectedMetric];
  };
  const chartRows = rows.filter((row) => Number.isFinite(Number(metricValueFor(row))));
  context.clearRect(0, 0, width, height);
  context.strokeStyle = "#52616b";
  context.lineWidth = 1;
  context.beginPath();
  context.moveTo(42, 14);
  context.lineTo(42, height - 30);
  context.lineTo(width - 12, height - 30);
  context.stroke();
  if (!chartRows.length) {
    context.fillStyle = "#b4c1c9";
    context.font = "14px system-ui";
    context.fillText("Generation metrics will appear here", 56, 40);
    return;
  }
  const values = chartRows.map((row) => Number(metricValueFor(row)));
  const min = Math.min(...values, 0);
  const max = Math.max(...values, 1);
  const span = max - min || 1;
  const x = (i) => 50 + (i * (width - 70)) / Math.max(1, chartRows.length - 1);
  const y = (value) => height - 32 - ((value - min) / span) * (height - 54);
  context.strokeStyle = "#79ddb0";
  context.lineWidth = 2.5;
  context.beginPath();
  values.forEach((value, i) =>
    i ? context.lineTo(x(i), y(value)) : context.moveTo(x(i), y(value)),
  );
  context.stroke();
  context.fillStyle = "#eef4f7";
  context.font = "12px system-ui";
  chartRows.forEach((row, i) => {
    context.beginPath();
    context.arc(x(i), y(values[i]), 4, 0, Math.PI * 2);
    context.fill();
    context.fillText(
      `G${row.generation ?? i + 1}`,
      Math.max(44, x(i) - 10),
      height - 10,
    );
  });
}

export function renderComparisons({ table, records, labelRole, metricValue, createElement }) {
  const body = table.querySelector("tbody");
  const rows = Array.isArray(records) ? records : [];
  if (!rows.length) {
    body.innerHTML =
      '<tr><td colspan="7" class="empty-note">No controller evaluations are available for this run.</td></tr>';
    return;
  }
  body.replaceChildren(
    ...rows.map((record) => {
      const row = createElement("tr");
      const role = labelRole(record.task);
      const cycleOrDelay =
        record.task === "defense"
          ? `Delay ${metricValue(record.mean_defensive_delay)} s`
          : `Cycle ${metricValue(record.mean_cycle_time)} s`;
      const cells = [
        record.name || record.opponent || record.architecture || "Strategy",
        role,
        record.episodes ?? "—",
        metricValue(record.mean_acquisitions),
        metricValue(record.mean_scores),
        cycleOrDelay,
        record.scenario_comparability ||
          (record.comparable === true
            ? "Matched"
            : record.comparable === false
              ? "Unpaired"
              : "—"),
      ];
      for (const text of cells) {
        const cell = createElement("td");
        cell.textContent = String(text);
        row.append(cell);
      }
      return row;
    }),
  );
}
