const API_PREDICT = "/api/v1/predict";
const API_CLASSIFY = "/api/v1/classify";
const HISTORY_KEY = "prompt-security-history-v1";
const MAX_HISTORY = 30;

const form = document.getElementById("predict-form");
const promptInput = document.getElementById("prompt-input");
const runBtn = document.getElementById("run-btn");
const clearBtn = document.getElementById("clear-btn");
const clearHistoryBtn = document.getElementById("clear-history");
const errorEl = document.getElementById("error");
const resultEl = document.getElementById("result");
const decisionBadge = document.getElementById("decision-badge");
const classificationBadge = document.getElementById("classification-badge");
const promptDisplay = document.getElementById("prompt-display");
const scoreEl = document.getElementById("score");
const thresholdEl = document.getElementById("threshold");
const featureCountEl = document.getElementById("feature-count");
const classificationConfidenceEl = document.getElementById("classification-confidence");
const classificationThresholdEl = document.getElementById("classification-threshold");
const classificationRuleEl = document.getElementById("classification-rule");
const classificationUncertainEl = document.getElementById("classification-uncertain");
const historyList = document.getElementById("history-list");

function showError(msg) {
  errorEl.textContent = msg;
  errorEl.classList.remove("hidden");
}

function clearError() {
  errorEl.textContent = "";
  errorEl.classList.add("hidden");
}

function saveHistory(item) {
  const existing = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
  existing.unshift(item);
  const trimmed = existing.slice(0, MAX_HISTORY);
  localStorage.setItem(HISTORY_KEY, JSON.stringify(trimmed));
}

function loadHistory() {
  return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
}

function renderHistory() {
  const history = loadHistory();
  historyList.innerHTML = "";
  if (!history.length) {
    historyList.innerHTML = "<li class='history-meta'>No prompts run yet.</li>";
    return;
  }

  history.forEach((item) => {
    const li = document.createElement("li");
    li.className = `history-item ${item.is_anomalous ? "bad" : "ok"}`;
    const label = item.predicted_label || "n/a";
    const confidence = Number(item.confidence || 0).toFixed(4);
    li.innerHTML = `
      <div>${item.prompt}</div>
      <div class="history-meta">
        ${new Date(item.ts).toLocaleString()} | anomaly=${item.decision_label} (${item.anomaly_score.toFixed(4)}/${item.threshold.toFixed(4)}) | class=${label} (${confidence})
      </div>
    `;
    historyList.appendChild(li);
  });
}

function badgeToneForClassification(result) {
  const label = String(result.predicted_label || "").toLowerCase();
  if (result.is_uncertain) {
    return "warn";
  }
  if (label.includes("mal") || label.includes("attack") || label.includes("inject") || label.includes("unsafe")) {
    return "bad";
  }
  if (label.includes("benign") || label.includes("safe") || label.includes("normal")) {
    return "ok";
  }
  return "neutral";
}

function renderResult(anomalyResult, classificationResult) {
  resultEl.classList.remove("hidden");
  const isBad = Boolean(anomalyResult.is_anomalous);
  const classifierTone = badgeToneForClassification(classificationResult);

  decisionBadge.textContent = anomalyResult.decision_label;
  decisionBadge.className = `badge ${isBad ? "bad" : "ok"}`;

  promptDisplay.textContent = anomalyResult.prompt_normalized;
  promptDisplay.className = isBad ? "prompt-bad" : "prompt-ok";

  scoreEl.textContent = Number(anomalyResult.anomaly_score).toFixed(6);
  thresholdEl.textContent = Number(anomalyResult.threshold).toFixed(6);
  featureCountEl.textContent = String(anomalyResult.feature_count);

  classificationBadge.textContent = classificationResult.predicted_label;
  classificationBadge.className = `badge ${classifierTone}`;
  classificationConfidenceEl.textContent = Number(classificationResult.confidence).toFixed(6);
  classificationThresholdEl.textContent = Number(classificationResult.min_confidence).toFixed(6);
  classificationRuleEl.textContent = classificationResult.decision_rule;
  classificationUncertainEl.textContent = classificationResult.is_uncertain ? "Yes" : "No";
}

async function runPrediction(prompt) {
  const response = await fetch(API_PREDICT, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt }),
  });
  const body = await response.json();
  if (!response.ok) {
    throw new Error(body.detail || "Request failed");
  }
  return body;
}

async function runClassification(prompt) {
  const response = await fetch(API_CLASSIFY, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt }),
  });
  const body = await response.json();
  if (!response.ok) {
    throw new Error(body.detail || "Classification failed");
  }
  return body;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  clearError();

  const prompt = promptInput.value.trim();
  if (!prompt) {
    showError("Prompt is required.");
    return;
  }

  runBtn.disabled = true;
  runBtn.textContent = "Running...";
  try {
    const [anomalyResult, classificationResult] = await Promise.all([
      runPrediction(prompt),
      runClassification(prompt),
    ]);
    renderResult(anomalyResult, classificationResult);
    saveHistory({
      ...anomalyResult,
      ...classificationResult,
      ts: new Date().toISOString(),
    });
    renderHistory();
  } catch (err) {
    showError(err.message || "Analysis failed.");
  } finally {
    runBtn.disabled = false;
    runBtn.textContent = "Run Analysis";
  }
});

clearBtn.addEventListener("click", () => {
  promptInput.value = "";
  clearError();
});

clearHistoryBtn.addEventListener("click", () => {
  localStorage.removeItem(HISTORY_KEY);
  renderHistory();
});

renderHistory();
