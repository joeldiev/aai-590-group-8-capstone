const API_PREDICT = "/api/v1/predict";
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
const promptDisplay = document.getElementById("prompt-display");
const scoreEl = document.getElementById("score");
const thresholdEl = document.getElementById("threshold");
const featureCountEl = document.getElementById("feature-count");
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
    li.innerHTML = `
      <div>${item.prompt}</div>
      <div class="history-meta">
        ${new Date(item.ts).toLocaleString()} | score=${item.anomaly_score.toFixed(4)} | threshold=${item.threshold.toFixed(4)} | ${item.decision_label}
      </div>
    `;
    historyList.appendChild(li);
  });
}

function renderResult(result) {
  resultEl.classList.remove("hidden");
  const isBad = Boolean(result.is_anomalous);

  decisionBadge.textContent = result.decision_label;
  decisionBadge.className = `badge ${isBad ? "bad" : "ok"}`;

  promptDisplay.textContent = result.prompt_normalized;
  promptDisplay.className = isBad ? "prompt-bad" : "prompt-ok";

  scoreEl.textContent = Number(result.anomaly_score).toFixed(6);
  thresholdEl.textContent = Number(result.threshold).toFixed(6);
  featureCountEl.textContent = String(result.feature_count);
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
    const result = await runPrediction(prompt);
    renderResult(result);
    saveHistory({ ...result, ts: new Date().toISOString() });
    renderHistory();
  } catch (err) {
    showError(err.message || "Prediction failed.");
  } finally {
    runBtn.disabled = false;
    runBtn.textContent = "Run Prediction";
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
