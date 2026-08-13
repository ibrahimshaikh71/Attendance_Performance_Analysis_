const NUMERIC_COLUMNS = [
  "Attendance_Percentage",
  "Maths_Marks",
  "Science_Marks",
  "English_Marks",
  "Assignment_Score",
  "Study_Hours_Per_Day",
  "Previous_Marks",
  "Average_Marks",
];

const CORR_COLUMNS = [
  "Attendance_Percentage",
  "Assignment_Score",
  "Study_Hours_Per_Day",
  "Previous_Marks",
  "Average_Marks",
];

const CATEGORY_ORDER = ["Low", "Average", "Good", "Excellent"];
const CATEGORY_COLORS = {
  Low: "#440154",
  Average: "#3b528b",
  Good: "#21918c",
  Excellent: "#5ec962",
};

const chartInstances = {};

function splitCSVLine(line) {
  const result = [];
  let cur = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (inQuotes) {
      if (c === '"') {
        if (line[i + 1] === '"') {
          cur += '"';
          i++;
        } else {
          inQuotes = false;
        }
      } else {
        cur += c;
      }
    } else if (c === '"') {
      inQuotes = true;
    } else if (c === ",") {
      result.push(cur);
      cur = "";
    } else {
      cur += c;
    }
  }
  result.push(cur);
  return result;
}

function parseCSV(text) {
  const lines = text.replace(/\r\n/g, "\n").split("\n").filter((l) => l.length > 0);
  if (lines.length === 0) return [];
  const headers = splitCSVLine(lines[0]).map((h) => h.trim());
  return lines.slice(1).map((line) => {
    const values = splitCSVLine(line);
    const obj = {};
    headers.forEach((h, i) => {
      obj[h] = values[i] !== undefined ? values[i] : "";
    });
    return obj;
  });
}

function categorize(pct) {
  if (!Number.isFinite(pct)) return null;
  if (pct <= 69.99) return "Low";
  if (pct <= 79.99) return "Average";
  if (pct <= 89.99) return "Good";
  return "Excellent";
}

function cleanData(rows) {
  const seen = new Set();
  const deduped = [];
  for (const r of rows) {
    const key = JSON.stringify(r);
    if (seen.has(key)) continue;
    seen.add(key);
    deduped.push(r);
  }

  const coerced = deduped.map((r) => {
    const nr = { ...r };
    for (const col of NUMERIC_COLUMNS) {
      const v = parseFloat(r[col]);
      nr[col] = Number.isFinite(v) ? v : NaN;
    }
    return nr;
  });

  const valid = coerced.filter((r) => NUMERIC_COLUMNS.every((col) => Number.isFinite(r[col])));
  valid.forEach((r) => {
    r.Attendance_Category = categorize(r.Attendance_Percentage);
  });
  return valid;
}

function mean(arr) {
  if (arr.length === 0) return NaN;
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

function median(arr) {
  if (arr.length === 0) return NaN;
  const s = [...arr].sort((a, b) => a - b);
  const n = s.length;
  return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
}

function pearson(xs, ys) {
  const n = xs.length;
  const mx = mean(xs);
  const my = mean(ys);
  let num = 0;
  let dx2 = 0;
  let dy2 = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx;
    const dy = ys[i] - my;
    num += dx * dy;
    dx2 += dx * dx;
    dy2 += dy * dy;
  }
  const denom = Math.sqrt(dx2 * dy2);
  return denom === 0 ? 0 : num / denom;
}

function linreg(xs, ys) {
  const mx = mean(xs);
  const my = mean(ys);
  let num = 0;
  let den = 0;
  for (let i = 0; i < xs.length; i++) {
    num += (xs[i] - mx) * (ys[i] - my);
    den += (xs[i] - mx) ** 2;
  }
  const slope = den === 0 ? 0 : num / den;
  const intercept = my - slope * mx;
  return { slope, intercept };
}

function groupBy(rows, keyFn) {
  const map = new Map();
  rows.forEach((r) => {
    const key = keyFn(r);
    if (!map.has(key)) map.set(key, []);
    map.get(key).push(r);
  });
  return map;
}

function formatNum(v) {
  if (typeof v !== "number" || !Number.isFinite(v)) return String(v);
  return Number.isInteger(v) ? String(v) : v.toFixed(2);
}

function renderTable(tableEl, columns, rows) {
  tableEl.innerHTML = "";
  const thead = document.createElement("thead");
  const trh = document.createElement("tr");
  columns.forEach((c) => {
    const th = document.createElement("th");
    th.textContent = c;
    trh.appendChild(th);
  });
  thead.appendChild(trh);
  tableEl.appendChild(thead);

  const tbody = document.createElement("tbody");
  rows.forEach((r) => {
    const tr = document.createElement("tr");
    columns.forEach((c) => {
      const td = document.createElement("td");
      td.textContent = formatNum(r[c]);
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  tableEl.appendChild(tbody);
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function corrColor(v) {
  const clamped = Math.max(-1, Math.min(1, v));
  const neg = [33, 102, 172];
  const mid = [247, 247, 247];
  const pos = [178, 24, 43];
  let c;
  if (clamped <= 0) {
    const t = clamped + 1;
    c = neg.map((n, i) => lerp(n, mid[i], t));
  } else {
    c = mid.map((n, i) => lerp(n, pos[i], clamped));
  }
  return `rgb(${c.map((x) => Math.round(x)).join(",")})`;
}

function destroyChart(id) {
  if (chartInstances[id]) {
    chartInstances[id].destroy();
    delete chartInstances[id];
  }
}

function setStatus(message, isError) {
  const el = document.getElementById("sidebarStatus");
  el.textContent = message;
  el.style.color = isError ? "#b3261e" : "";
}

function toCSV(rows, columns) {
  const escape = (v) => {
    const s = v === undefined || v === null ? "" : String(v);
    if (/[",\n]/.test(s)) return '"' + s.replace(/"/g, '""') + '"';
    return s;
  };
  const lines = [columns.map(escape).join(",")];
  rows.forEach((r) => lines.push(columns.map((c) => escape(r[c])).join(",")));
  return lines.join("\n");
}

function render(rows) {
  const columns = Object.keys(rows[0]);

  renderTable(document.getElementById("previewTable"), columns, rows.slice(0, 10));

  document.getElementById("metricRows").textContent = rows.length;
  document.getElementById("metricCols").textContent = columns.length;

  const atRisk = rows.filter((r) => r.Attendance_Percentage < 70 || r.Average_Marks < 50);
  document.getElementById("metricAtRisk").textContent = atRisk.length;

  const catGroups = groupBy(rows, (r) => r.Attendance_Category);
  const categoryRows = CATEGORY_ORDER.filter((c) => catGroups.has(c)).map((c) => {
    const group = catGroups.get(c);
    const marks = group.map((r) => r.Average_Marks);
    return {
      Category: c,
      Count: group.length,
      Mean: mean(marks),
      Median: median(marks),
    };
  });
  renderTable(document.getElementById("categoryTable"), ["Category", "Count", "Mean", "Median"], categoryRows);

  const classTableEl = document.getElementById("classTable");
  if (columns.includes("Class")) {
    const classGroups = groupBy(rows, (r) => r.Class);
    const classRows = [...classGroups.keys()].sort().map((cls) => {
      const group = classGroups.get(cls);
      return {
        Class: cls,
        Attendance_Percentage: mean(group.map((r) => r.Attendance_Percentage)),
        Average_Marks: mean(group.map((r) => r.Average_Marks)),
      };
    });
    renderTable(classTableEl, ["Class", "Attendance_Percentage", "Average_Marks"], classRows);
  } else {
    classTableEl.innerHTML = "<tr><td>No \"Class\" column found in this dataset.</td></tr>";
  }

  const subjectRows = [
    { Subject: "Maths_Marks", Average: mean(rows.map((r) => r.Maths_Marks)) },
    { Subject: "Science_Marks", Average: mean(rows.map((r) => r.Science_Marks)) },
    { Subject: "English_Marks", Average: mean(rows.map((r) => r.English_Marks)) },
  ];
  renderTable(document.getElementById("subjectTable"), ["Subject", "Average"], subjectRows);

  const correlation = pearson(rows.map((r) => r.Attendance_Percentage), rows.map((r) => r.Average_Marks));
  document.getElementById("correlationInfo").textContent =
    `Attendance vs Average Marks correlation: ${correlation.toFixed(3)}`;

  renderCharts(rows, categoryRows, subjectRows, columns);
  renderHeatmap(rows);

  renderTable(document.getElementById("atRiskTable"), columns, atRisk);

  const downloadBtn = document.getElementById("downloadBtn");
  downloadBtn.onclick = () => {
    const csv = toCSV(atRisk, columns);
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "at_risk_students.csv";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };
}

function renderCharts(rows, categoryRows, subjectRows, columns) {
  destroyChart("scatter");
  destroyChart("categoryBar");
  destroyChart("subjectBar");
  destroyChart("classBar");

  const scatterDatasets = CATEGORY_ORDER.filter((c) => rows.some((r) => r.Attendance_Category === c)).map((c) => ({
    label: c,
    data: rows.filter((r) => r.Attendance_Category === c).map((r) => ({ x: r.Attendance_Percentage, y: r.Average_Marks })),
    backgroundColor: CATEGORY_COLORS[c],
    pointRadius: 3,
  }));

  const xs = rows.map((r) => r.Attendance_Percentage);
  const ys = rows.map((r) => r.Average_Marks);
  const { slope, intercept } = linreg(xs, ys);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  scatterDatasets.push({
    type: "line",
    label: "Trend",
    data: [
      { x: minX, y: slope * minX + intercept },
      { x: maxX, y: slope * maxX + intercept },
    ],
    borderColor: "#1f2933",
    borderWidth: 2,
    pointRadius: 0,
    showLine: true,
  });

  chartInstances.scatter = new Chart(document.getElementById("chartScatter"), {
    type: "scatter",
    data: { datasets: scatterDatasets },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { title: { display: true, text: "Attendance vs Academic Performance" } },
      scales: {
        x: { title: { display: true, text: "Attendance (%)" } },
        y: { title: { display: true, text: "Average Marks" } },
      },
    },
  });

  chartInstances.categoryBar = new Chart(document.getElementById("chartCategoryBar"), {
    type: "bar",
    data: {
      labels: categoryRows.map((r) => r.Category),
      datasets: [{ label: "Average Marks", data: categoryRows.map((r) => r.Mean), backgroundColor: "#4c72b0" }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { title: { display: true, text: "Average Marks by Attendance Category" }, legend: { display: false } },
    },
  });

  chartInstances.subjectBar = new Chart(document.getElementById("chartSubjectBar"), {
    type: "bar",
    data: {
      labels: subjectRows.map((r) => r.Subject),
      datasets: [
        {
          label: "Average Marks",
          data: subjectRows.map((r) => r.Average),
          backgroundColor: ["#ff7f0e", "#2ca02c", "#d62728"],
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { title: { display: true, text: "Average Subject Performance" }, legend: { display: false } },
    },
  });

  if (columns.includes("Class")) {
    const classGroups = groupBy(rows, (r) => r.Class);
    const classLabels = [...classGroups.keys()].sort();
    const classMeans = classLabels.map((cls) => mean(classGroups.get(cls).map((r) => r.Attendance_Percentage)));
    chartInstances.classBar = new Chart(document.getElementById("chartClassBar"), {
      type: "bar",
      data: {
        labels: classLabels,
        datasets: [{ label: "Attendance (%)", data: classMeans, backgroundColor: "#9467bd" }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { title: { display: true, text: "Average Attendance by Class" }, legend: { display: false } },
      },
    });
  }
}

function renderHeatmap(rows) {
  const table = document.getElementById("heatmapTable");
  table.innerHTML = "";
  const values = {};
  CORR_COLUMNS.forEach((c) => {
    values[c] = rows.map((r) => r[c]);
  });

  const thead = document.createElement("thead");
  const trh = document.createElement("tr");
  trh.appendChild(document.createElement("th"));
  CORR_COLUMNS.forEach((c) => {
    const th = document.createElement("th");
    th.textContent = c;
    trh.appendChild(th);
  });
  thead.appendChild(trh);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  CORR_COLUMNS.forEach((rowCol) => {
    const tr = document.createElement("tr");
    const th = document.createElement("th");
    th.textContent = rowCol;
    tr.appendChild(th);
    CORR_COLUMNS.forEach((colCol) => {
      const v = pearson(values[rowCol], values[colCol]);
      const td = document.createElement("td");
      td.textContent = v.toFixed(2);
      td.style.backgroundColor = corrColor(v);
      td.style.color = Math.abs(v) > 0.55 ? "#ffffff" : "#1f2933";
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
}

function processAndRender(rawRows, successMsg) {
  try {
    const cleaned = cleanData(rawRows);
    if (cleaned.length === 0) {
      setStatus(
        "No valid rows found after cleaning. Make sure the file has the expected numeric columns (Attendance_Percentage, Maths_Marks, Science_Marks, English_Marks, Assignment_Score, Study_Hours_Per_Day, Previous_Marks, Average_Marks).",
        true
      );
      return;
    }
    document.getElementById("emptyState").classList.add("hidden");
    document.getElementById("results").classList.remove("hidden");
    document.getElementById("statusBanner").textContent = successMsg;
    setStatus("");
    render(cleaned);
  } catch (err) {
    setStatus("Failed to process file: " + err.message, true);
  }
}

async function loadSample() {
  setStatus("Loading sample dataset...");
  try {
    const res = await fetch("data/sample.csv");
    const text = await res.text();
    const rows = parseCSV(text);
    processAndRender(rows, "Sample dataset created.");
  } catch (err) {
    setStatus("Failed to load sample dataset: " + err.message, true);
  }
}

function handleFile(file) {
  const name = file.name.toLowerCase();
  if (name.endsWith(".csv")) {
    const reader = new FileReader();
    reader.onload = (e) => {
      const rows = parseCSV(e.target.result);
      processAndRender(rows, "Data uploaded and loaded successfully.");
    };
    reader.onerror = () => setStatus("Failed to read uploaded file.", true);
    reader.readAsText(file);
  } else if (name.endsWith(".xls") || name.endsWith(".xlsx")) {
    const reader = new FileReader();
    reader.onload = (e) => {
      const data = new Uint8Array(e.target.result);
      const wb = XLSX.read(data, { type: "array" });
      const sheet = wb.Sheets[wb.SheetNames[0]];
      const rows = XLSX.utils.sheet_to_json(sheet, { defval: "" });
      processAndRender(rows, "Data uploaded and loaded successfully.");
    };
    reader.onerror = () => setStatus("Failed to read uploaded file.", true);
    reader.readAsArrayBuffer(file);
  } else {
    setStatus("Unsupported file format. Please upload a CSV or Excel file.", true);
  }
}

document.getElementById("sampleBtn").addEventListener("click", loadSample);
document.getElementById("fileInput").addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (file) handleFile(file);
});
