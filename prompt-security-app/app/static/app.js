const API_PROMPT = "/api/v1/prompt";
const HISTORY_KEY = "prompt-security-history-v1";
const MAX_HISTORY = 30;

const form = document.getElementById("predict-form");
const promptInput = document.getElementById("prompt-input");
const runBtn = document.getElementById("run-btn");
const clearBtn = document.getElementById("clear-btn");
const clearHistoryBtn = document.getElementById("clear-history");
const errorEl = document.getElementById("error");
const resultEl = document.getElementById("result");
const finalDecisionBadge = document.getElementById("final-decision-badge");
const finalDecisionFlag = document.getElementById("final-decision-flag");
const decisionReasonsList = document.getElementById("decision-reasons-list");
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
const severityPanel = document.getElementById("severity-panel");
const severityTierBadge = document.getElementById("severity-tier-badge");
const severityThreatType = document.getElementById("severity-threat-type");
const severityScoreFill = document.getElementById("severity-score-fill");
const severityScoreEl = document.getElementById("severity-score");
const severityThreatConfidence = document.getElementById("severity-threat-confidence");
const severityLatency = document.getElementById("severity-latency");
const severityBreakdown = document.getElementById("severity-breakdown");
const severityIntel = document.getElementById("severity-intel");

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
    li.className = `history-item ${item.is_malicious ? "bad" : "ok"}`;
    const label = item.classification?.predicted_label || "n/a";
    const confidence = Number(item.classification?.confidence || 0).toFixed(4);
    const anomalyScore = Number(item.anomaly?.anomaly_score || 0).toFixed(4);
    const anomalyThreshold = Number(item.anomaly?.threshold || 0).toFixed(4);
    li.innerHTML = `
      <div>${item.anomaly?.prompt || item.classification?.prompt || "Unknown prompt"}</div>
      <div class="history-meta">
        ${new Date(item.ts).toLocaleString()} | final=${item.final_label} | anomaly=${item.anomaly?.decision_label || "n/a"} (${anomalyScore}/${anomalyThreshold}) | class=${label} (${confidence})
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

function finalDecisionTone(result) {
  if (result.is_malicious) {
    return "bad";
  }

  const hasUncertainSignal =
    Boolean(result.classification?.is_uncertain) ||
    String(result.final_label || "").toLowerCase().includes("review") ||
    (Array.isArray(result.reasons) && result.reasons.some((reason) => reason.includes("uncertain")));

  if (hasUncertainSignal) {
    return "warn";
  }

  return "ok";
}

function renderResult(result) {
  resultEl.classList.remove("hidden");
  const anomalyResult = result.anomaly;
  const classificationResult = result.classification;
  const finalTone = finalDecisionTone(result);
  const finalIsMalicious = finalTone === "bad";
  const isBad = Boolean(anomalyResult.is_anomalous);
  const classifierTone = badgeToneForClassification(classificationResult);
  const reasonItems = Array.isArray(result.reasons) ? result.reasons : [];

  finalDecisionBadge.textContent = result.final_label;
  finalDecisionBadge.className = `badge ${finalTone}`;
  finalDecisionFlag.textContent = result.is_malicious ? "Yes" : "No";
  decisionReasonsList.innerHTML = "";
  if (!reasonItems.length) {
    decisionReasonsList.innerHTML = "<li class='history-meta'>No decision reasons returned.</li>";
  } else {
    reasonItems.forEach((reason) => {
      const li = document.createElement("li");
      li.textContent = reason;
      decisionReasonsList.appendChild(li);
    });
  }

  decisionBadge.textContent = anomalyResult.decision_label;
  decisionBadge.className = `badge ${isBad ? "bad" : "ok"}`;

  promptDisplay.textContent = anomalyResult.prompt_normalized;
  promptDisplay.className = `prompt-${finalTone}`;

  scoreEl.textContent = Number(anomalyResult.anomaly_score).toFixed(6);
  thresholdEl.textContent = Number(anomalyResult.threshold).toFixed(6);
  featureCountEl.textContent = String(anomalyResult.feature_count);

  classificationBadge.textContent = classificationResult.predicted_label;
  classificationBadge.className = `badge ${classifierTone}`;
  classificationConfidenceEl.textContent = Number(classificationResult.confidence).toFixed(6);
  classificationThresholdEl.textContent = Number(classificationResult.min_confidence).toFixed(6);
  classificationRuleEl.textContent = classificationResult.decision_rule;
  classificationUncertainEl.textContent = classificationResult.is_uncertain ? "Yes" : "No";

  // Severity panel — only shown when malicious
  if (result.severity) {
    renderSeverity(result.severity);
  } else {
    severityPanel.classList.add("hidden");
  }
}

const TIER_COLORS = {
  critical: "#dc2626",
  high: "#ea580c",
  medium: "#ca8a04",
  low: "#65a30d",
};

const THREAT_TYPE_LABELS = {
  injection: "Prompt Injection",
  jailbreak: "Jailbreak",
  exfiltration: "Data Exfiltration",
  unknown_malicious: "Unknown Threat",
};

function renderSeverity(severity) {
  severityPanel.classList.remove("hidden");

  const tier = severity.severity_tier;
  const color = TIER_COLORS[tier] || "#71717a";

  severityTierBadge.textContent = tier.toUpperCase();
  severityTierBadge.style.background = color;

  severityThreatType.textContent = THREAT_TYPE_LABELS[severity.threat_type] || severity.threat_type;

  // Score bar (0-10 scale)
  const pct = Math.min((severity.severity_score / 10) * 100, 100);
  severityScoreFill.style.width = pct + "%";
  severityScoreFill.style.background = color;

  severityScoreEl.textContent = severity.severity_score + " / 10";
  severityThreatConfidence.textContent = Number(severity.threat_confidence).toFixed(4);
  severityLatency.textContent = Number(severity.latency_ms).toFixed(1) + " ms";

  // Scoring breakdown
  const breakdown = severity.scoring_breakdown || {};
  const breakdownKeys = Object.keys(breakdown);
  if (breakdownKeys.length) {
    severityBreakdown.innerHTML = "<h4>Scoring Breakdown</h4>" +
      breakdownKeys.map(k => {
        const label = k.replace(/_/g, " ");
        return `<div class="row"><span class="k">${label}</span><span>+${breakdown[k]}</span></div>`;
      }).join("");
  } else {
    severityBreakdown.innerHTML = "";
  }

  // Threat intelligence
  const intel = severity.threat_intel || {};
  if (intel.mitre_atlas || intel.owasp) {
    let html = "<h4>Threat Intelligence</h4>";
    if (intel.mitre_atlas) {
      const m = intel.mitre_atlas;
      html += `<div class="intel-source">
        <div class="intel-label">MITRE ATLAS</div>
        <div class="row"><span class="k">Technique</span><span>${m.technique_id} — ${m.name}</span></div>
        <div class="row"><span class="k">Tactics</span><span>${(m.tactics || []).join(", ")}</span></div>
        <div class="row"><span class="k">Link</span><span><a href="${m.url}" target="_blank" rel="noopener">View on ATLAS</a></span></div>
      </div>`;
    }
    if (intel.owasp) {
      const o = intel.owasp;
      html += `<div class="intel-source">
        <div class="intel-label">OWASP LLM Top 10</div>
        <div class="row"><span class="k">Risk</span><span>${o.id} — ${o.name}</span></div>
        <div class="row"><span class="k">Link</span><span><a href="${o.url}" target="_blank" rel="noopener">View on OWASP</a></span></div>
      </div>`;
    }
    if (intel.recommended_actions) {
      html += `<div class="intel-actions"><h4>Recommended Actions</h4><ul>` +
        intel.recommended_actions.map(a => `<li>${a}</li>`).join("") +
        `</ul></div>`;
    }
    severityIntel.innerHTML = html;
  } else {
    severityIntel.innerHTML = "";
  }
}

async function runPromptAnalysis(prompt) {
  const response = await fetch(API_PROMPT, {
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
    const result = await runPromptAnalysis(prompt);
    renderResult(result);
    saveHistory({
      ...result,
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
