const PLAN_COLOR = {
  ON_DEMAND: "#1d4ed8",
  SPOT: "#c2410c",
  RESERVATION: "#047857",
  CUSTOM: "#6d28d9",
};

const LINE_COLOR = ["#1d4ed8", "#c2410c", "#047857", "#6d28d9", "#b45309", "#0f766e", "#be123c", "#4338ca"];

const state = {
  meta: null,
  catalog: [],
  selected: new Set(),
  plans: new Set(),
  query: "",
};

let plotted = [];
let tableSeries = [];
let staticHistory = null;
let chartSize = { width: 0, height: 0 };
const tableView = { showPrice: false, newestFirst: true };

const search = document.querySelector("#search");
const options = document.querySelector("#options");
const chips = document.querySelector("#chips");
const chart = document.querySelector("#chart");
const table = document.querySelector("#table");
const matchCount = document.querySelector("#match-count");

for (const button of document.querySelectorAll(".panel-toggle")) {
  button.addEventListener("click", () => {
    const panel = button.closest(".panel");
    const collapsed = panel.classList.toggle("collapsed");
    button.setAttribute("aria-expanded", String(!collapsed));
  });
}

for (const head of document.querySelectorAll(".panel-head")) {
  head.addEventListener("click", (event) => {
    if (event.target.closest(".panel-tools") || event.target.closest(".panel-toggle")) return;
    head.querySelector(".panel-toggle").click();
  });
}

document.querySelector("#show-price").addEventListener("change", (event) => {
  tableView.showPrice = event.target.checked;
  if (tableSeries.length) drawTable(tableSeries);
});

document.querySelector("#date-sort").addEventListener("click", () => {
  tableView.newestFirst = !tableView.newestFirst;
  const button = document.querySelector("#date-sort");
  button.textContent = tableView.newestFirst ? "Newest first" : "Oldest first";
  button.setAttribute("aria-pressed", String(tableView.newestFirst));
  if (tableSeries.length) drawTable(tableSeries);
});

document.querySelector("#export-csv").addEventListener("click", downloadTableCsv);

new ResizeObserver(() => {
  if (plotted.length) drawChart(plotted);
}).observe(chart);

search.addEventListener("input", () => {
  state.query = search.value.trim().toLowerCase();
  renderOptions();
});

document.querySelector("#select-visible").addEventListener("click", () => {
  for (const row of visibleSeries()) state.selected.add(row.id);
  renderOptions();
  renderChips();
  loadHistory();
});

document.querySelector("#clear-selected").addEventListener("click", () => {
  state.selected.clear();
  renderOptions();
  renderChips();
  loadHistory();
});

boot();

async function readJson(paths) {
  for (const path of paths) {
    try {
      const response = await fetch(path);
      const type = response.headers.get("content-type") || "";
      if (response.ok && type.includes("json")) return response.json();
    } catch (error) {
      // Try the next location. GitHub Pages has no API server.
    }
  }
  throw new Error("Could not load dashboard data");
}

async function boot() {
  const payload = await readJson(["/api/catalog", "data/catalog.json"]);
  state.meta = payload.meta;
  state.catalog = payload.series;
  state.plans = new Set(payload.meta.plans.map((plan) => plan.id));
  if (payload.meta.baseline_id) state.selected.add(payload.meta.baseline_id);
  document.querySelector("#title").textContent = `${payload.meta.baseline_name} · ${payload.meta.baseline_plan}`;
  document.querySelector("#formula").textContent =
    "Index × selected price = this baseline price. The baseline series itself is its price divided by the price on the first continuous day.";
  renderPlanFilters(payload.meta.plans);
  renderOptions();
  renderChips();
  loadHistory();
}

function renderPlanFilters(plans) {
  const host = document.querySelector("#plan-filters");
  host.replaceChildren();
  for (const plan of plans) {
    const button = document.createElement("button");
    button.type = "button";
    button.dataset.plan = plan.id;
    button.textContent = plan.label;
    button.setAttribute("aria-pressed", "true");
    button.addEventListener("click", () => {
      if (state.plans.has(plan.id)) state.plans.delete(plan.id);
      else state.plans.add(plan.id);
      button.setAttribute("aria-pressed", String(state.plans.has(plan.id)));
      renderOptions();
    });
    host.appendChild(button);
  }
}

function visibleSeries() {
  return state.catalog.filter((row) => {
    if (!state.plans.has(row.billing)) return false;
    if (!state.query) return true;
    const haystack = `${row.name} ${row.gpu_id} ${row.plan} ${row.billing}`.toLowerCase();
    return haystack.includes(state.query);
  });
}

function renderOptions() {
  const rows = visibleSeries();
  matchCount.textContent = `${rows.length} shown · ${state.selected.size} selected`;
  options.replaceChildren();
  for (const row of rows) {
    const item = document.createElement("li");
    if (state.selected.has(row.id)) item.classList.add("selected");
    const box = document.createElement("input");
    box.type = "checkbox";
    box.checked = state.selected.has(row.id);
    box.setAttribute("aria-label", `${row.name} ${row.plan}`);
    box.addEventListener("click", (event) => event.stopPropagation());
    box.addEventListener("change", () => toggle(row.id));
    const label = document.createElement("span");
    const name = document.createElement("span");
    name.className = "name";
    name.textContent = row.baseline ? `${row.name} (baseline)` : row.name;
    const id = document.createElement("span");
    id.className = "id";
    id.textContent = row.gpu_id;
    label.append(name, id);
    item.append(box, label, planBadge(row));
    item.addEventListener("click", () => toggle(row.id));
    options.appendChild(item);
  }
}

function toggle(id) {
  if (state.selected.has(id)) state.selected.delete(id);
  else state.selected.add(id);
  renderOptions();
  renderChips();
  loadHistory();
}

function renderChips() {
  chips.replaceChildren();
  const chosen = state.catalog.filter((row) => state.selected.has(row.id));
  if (!chosen.length) {
    const note = document.createElement("p");
    note.className = "empty";
    note.textContent = "Select one or more cards.";
    chips.appendChild(note);
    return;
  }
  chosen.forEach((row, index) => {
    const chip = document.createElement("span");
    chip.className = "chip";
    chip.style.borderLeftColor = LINE_COLOR[index % LINE_COLOR.length];
    const text = document.createElement("span");
    text.textContent = row.name;
    const remove = document.createElement("button");
    remove.type = "button";
    remove.textContent = "×";
    remove.setAttribute("aria-label", `Remove ${row.name} ${row.plan}`);
    remove.addEventListener("click", () => toggle(row.id));
    chip.append(text, planBadge(row), remove);
    chips.appendChild(chip);
  });
}

function planBadge(row) {
  const badge = document.createElement("span");
  badge.className = `plan plan-${row.billing}`;
  badge.textContent = row.plan;
  return badge;
}

async function loadHistory() {
  const ids = [...state.selected];
  if (!ids.length) {
    plotted = [];
    tableSeries = [];
    chart.replaceChildren(empty("Select a card to plot its index."));
    table.replaceChildren(empty("Historical index values will appear here."));
    return;
  }
  let payload = null;
  try {
    const response = await fetch(`/api/history?series=${encodeURIComponent(ids.join(","))}`);
    const type = response.headers.get("content-type") || "";
    if (response.ok && type.includes("json")) payload = await response.json();
  } catch (error) {
    payload = null;
  }
  if (!payload) {
    if (!staticHistory) staticHistory = await readJson(["data/history.json"]);
    payload = staticHistory;
  }
  const ordered = ids.map((id) => payload.series.find((series) => series.id === id)).filter(Boolean);
  drawChart(ordered);
  drawTable(ordered);
}

function drawChart(seriesList) {
  const dates = [...new Set(seriesList.flatMap((series) => series.points.map((point) => point.date)))].sort();
  const width = Math.round(chart.clientWidth);
  const height = Math.round(chart.clientHeight);
  const sameSeries = plotted === seriesList;
  plotted = seriesList;
  if (!dates.length) {
    chart.replaceChildren(empty("No index values for this selection."));
    return;
  }
  if (width < 80 || height < 80) return;
  if (
    sameSeries
    && Math.abs(width - chartSize.width) < 2
    && Math.abs(height - chartSize.height) < 2
    && chart.querySelector("svg")
  ) return;
  chartSize = { width, height };
  const margin = { top: 16, right: 16, bottom: 36, left: 52 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;
  const values = seriesList.flatMap((series) => series.points.map((point) => point.index));
  let min = Math.min(...values, 1);
  let max = Math.max(...values, 1);
  if (min === max) {
    min -= 0.05;
    max += 0.05;
  }
  const pad = (max - min) * 0.08;
  min -= pad;
  max += pad;
  const x = (date) => margin.left + (dates.indexOf(date) / Math.max(dates.length - 1, 1)) * innerWidth;
  const y = (value) => margin.top + ((max - value) / (max - min)) * innerHeight;

  const svg = svgEl("svg", { viewBox: `0 0 ${width} ${height}`, role: "img" });
  svg.appendChild(svgEl("title", {}, "Inverse index over time"));
  for (let tick = 0; tick < 5; tick += 1) {
    const value = min + ((max - min) * tick) / 4;
    const yPos = y(value);
    svg.appendChild(svgEl("line", { class: "grid", x1: margin.left, x2: width - margin.right, y1: yPos, y2: yPos }));
    svg.appendChild(svgEl("text", { class: "axis", x: margin.left - 8, y: yPos + 4, "text-anchor": "end" }, value.toFixed(2)));
  }
  const labelEvery = Math.ceil(dates.length / 6);
  dates.forEach((date, index) => {
    if (index % labelEvery !== 0 && index !== dates.length - 1) return;
    svg.appendChild(svgEl("text", {
      class: "axis",
      x: x(date),
      y: height - 12,
      "text-anchor": "middle",
    }, date.slice(5)));
  });

  const yOne = y(1);
  svg.appendChild(svgEl("line", {
    class: "reference-line",
    x1: margin.left,
    x2: width - margin.right,
    y1: yOne,
    y2: yOne,
  }));
  svg.appendChild(svgEl("text", {
    class: "reference-label",
    x: width - margin.right - 4,
    y: yOne - 6,
    "text-anchor": "end",
  }, "1.000"));

  seriesList.forEach((series, index) => {
    const color = LINE_COLOR[index % LINE_COLOR.length];
    const byDate = new Map(series.points.map((point) => [point.date, point.index]));
    let commands = "";
    let previous = null;
    for (const date of dates) {
      const value = byDate.get(date);
      if (value == null) {
        previous = null;
        continue;
      }
      const gap = previous && (Date.parse(date) - Date.parse(previous)) / 86400000 > 1;
      commands += `${previous == null || gap ? "M" : "L"} ${x(date).toFixed(1)} ${y(value).toFixed(1)} `;
      previous = date;
    }
    const dash = { ON_DEMAND: "", SPOT: "7 4", RESERVATION: "2 3", CUSTOM: "9 3 2 3" }[series.billing] || "";
    svg.appendChild(svgEl("path", {
      class: "chart-line",
      d: commands.trim(),
      stroke: color,
      "stroke-dasharray": dash,
    }));
  });

  const hover = svgEl("line", { class: "hover-line", x1: 0, x2: 0, y1: margin.top, y2: height - margin.bottom, visibility: "hidden" });
  svg.appendChild(hover);
  const overlay = svgEl("rect", {
    x: margin.left,
    y: margin.top,
    width: innerWidth,
    height: innerHeight,
    fill: "transparent",
  });
  svg.appendChild(overlay);

  const box = document.createElement("div");
  box.className = "chart-box";
  const tip = document.createElement("div");
  tip.className = "tip";
  tip.hidden = true;
  box.append(svg, tip);
  chart.replaceChildren(box);

  overlay.addEventListener("mousemove", (event) => {
    const bounds = svg.getBoundingClientRect();
    const relative = ((event.clientX - bounds.left) / bounds.width) * width;
    let nearest = 0;
    let best = Infinity;
    dates.forEach((date, index) => {
      const distance = Math.abs(x(date) - relative);
      if (distance < best) {
        best = distance;
        nearest = index;
      }
    });
    const date = dates[nearest];
    hover.setAttribute("x1", x(date));
    hover.setAttribute("x2", x(date));
    hover.setAttribute("visibility", "visible");
    tip.hidden = false;
    tip.replaceChildren();
    const heading = document.createElement("div");
    heading.textContent = date;
    tip.appendChild(heading);
    seriesList.forEach((series) => {
      const point = series.points.find((item) => item.date === date);
      const line = document.createElement("div");
      line.textContent = point ? `${series.name} ${series.plan}: ${point.index.toFixed(4)}` : `${series.name}: —`;
      tip.appendChild(line);
    });
    const left = event.clientX - bounds.left + 12;
    tip.style.left = `${Math.min(left, bounds.width - 180)}px`;
    tip.style.top = `${event.clientY - bounds.top + 12}px`;
  });
  overlay.addEventListener("mouseleave", () => {
    hover.setAttribute("visibility", "hidden");
    tip.hidden = true;
  });
}

function drawTable(seriesList) {
  tableSeries = seriesList;
  const dates = [...new Set(seriesList.flatMap((series) => series.points.map((point) => point.date)))].sort();
  if (tableView.newestFirst) dates.reverse();
  const tableEl = document.createElement("table");
  const head = document.createElement("tr");
  const dateHeader = document.createElement("th");
  dateHeader.textContent = "Date";
  head.appendChild(dateHeader);
  for (const series of seriesList) {
    const header = document.createElement("th");
    header.append(series.name, planBadge(series));
    head.appendChild(header);
    if (!tableView.showPrice) continue;
    const priceHeader = document.createElement("th");
    priceHeader.textContent = "Price";
    head.appendChild(priceHeader);
  }
  const thead = document.createElement("thead");
  thead.appendChild(head);
  const body = document.createElement("tbody");
  for (const date of dates) {
    const row = document.createElement("tr");
    const dateCell = document.createElement("td");
    dateCell.textContent = date;
    row.appendChild(dateCell);
    for (const series of seriesList) {
      const point = series.points.find((item) => item.date === date);
      const cell = document.createElement("td");
      cell.textContent = point ? point.index.toFixed(4) : "—";
      row.appendChild(cell);
      if (!tableView.showPrice) continue;
      const priceCell = document.createElement("td");
      priceCell.textContent = point && point.price != null ? Number(point.price).toFixed(4) : "—";
      row.appendChild(priceCell);
    }
    body.appendChild(row);
  }
  tableEl.append(thead, body);
  table.replaceChildren(tableEl);
}

function csvEscape(value) {
  const text = String(value);
  if (/[",\n]/.test(text)) return `"${text.replaceAll('"', '""')}"`;
  return text;
}

function currentTableCsv() {
  const dates = [...new Set(tableSeries.flatMap((series) => series.points.map((point) => point.date)))].sort();
  if (tableView.newestFirst) dates.reverse();
  const header = ["Date"];
  for (const series of tableSeries) {
    header.push(`${series.name} ${series.plan}`);
    if (tableView.showPrice) header.push(`${series.name} ${series.plan} price`);
  }
  const lines = [header.map(csvEscape).join(",")];
  for (const date of dates) {
    const cells = [date];
    for (const series of tableSeries) {
      const point = series.points.find((item) => item.date === date);
      cells.push(point ? point.index.toFixed(4) : "");
      if (tableView.showPrice) cells.push(point && point.price != null ? Number(point.price).toFixed(4) : "");
    }
    lines.push(cells.map(csvEscape).join(","));
  }
  return `${lines.join("\n")}\n`;
}

function downloadTableCsv() {
  if (!tableSeries.length) return;
  const blob = new Blob([currentTableCsv()], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "index-history.csv";
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function empty(message) {
  const note = document.createElement("p");
  note.className = "empty";
  note.textContent = message;
  return note;
}

function svgEl(name, attributes, text) {
  const element = document.createElementNS("http://www.w3.org/2000/svg", name);
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
  if (text != null) element.textContent = text;
  return element;
}
