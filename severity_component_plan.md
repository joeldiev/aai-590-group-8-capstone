# Severity & Threat Intelligence Component — Implementation Plan

## Overview

A third pipeline stage that activates **only when both classifiers agree a prompt is malicious**. It does two things: (1) classifies the **threat type** using a lightweight ML model, (2) scores **severity** via rules, and (3) fetches **threat intelligence** from structured sources. No LLM involved.

---

## Architecture

```
User Prompt
    │
    ├──► DAE Anomaly Detector ──► PredictionResponse
    │
    ├──► RoBERTa Classifier ────► ClassificationResponse
    │
    └──► decide_prompt_risk()
              │
              ├─ if is_malicious == True:
              │       │
              │       ▼
              │   SeverityService
              │       ├── ThreatTypeClassifier (trained model)
              │       │     └── Classifies into: injection, jailbreak,
              │       │         exfiltration, system_prompt_extraction
              │       │
              │       ├── SeverityScorer (rule-based)
              │       │     └── Combines: classifier confidence, anomaly margin,
              │       │         threat type weights, text heuristics
              │       │     └── Output: Low / Medium / High / Critical
              │       │
              │       └── ThreatIntelLookup (online + cache)
              │             └── Maps threat type → MITRE ATLAS + OWASP LLM Top 10
              │             └── Returns: technique ID, description, mitigations
              │
              ▼
         DecisionResponse (extended)
```

---

## Component 1: Threat Type Classifier (Trained Model)

**What:** A lightweight scikit-learn model that classifies malicious prompts into subtypes.

**Model:** TF-IDF + Gradient Boosted Trees (XGBoost or LightGBM)
- Fast inference (~1-5ms)
- No GPU required
- Proven strong on text classification tasks at this scale

**Classes:**
| ID | Label | Description |
|----|-------|-------------|
| 0 | injection | Attempts to override system instructions |
| 1 | jailbreak | Attempts to bypass safety/content filters |
| 2 | exfiltration | Attempts to extract data or system prompts |
| 3 | unknown_malicious | Doesn't match known patterns (OOD fallback) |

**Training data:** Reuse the existing multi-source datasets (deepset, JailbreakV-28K, Lakera/mosaic, hackaprompt, synthetic exfiltration) — these already have class labels before they were collapsed to binary.

**Artifacts to produce:**
- `models/severity/threat_classifier/tfidf_vectorizer.joblib`
- `models/severity/threat_classifier/model.joblib`
- `models/severity/threat_classifier/label_mapping.json`

**Academic justification:** Ensemble-of-specialists pattern — binary detector for speed/recall, subtype classifier for interpretability. Cite: "Hierarchical classification reduces error propagation by isolating detection from categorization."

---

## Component 2: Severity Scorer (Rule-Based)

**What:** Deterministic scoring that combines signals from all three components into a categorical tier.

**Input signals:**
| Signal | Source | Weight |
|--------|--------|--------|
| `classifier_confidence` | RoBERTa softmax | High confidence → higher severity |
| `anomaly_margin` | DAE score - threshold | Larger margin → higher severity |
| `cross_agreement` | Both flagged malicious | Boolean amplifier |
| `threat_type` | Threat classifier | Exfiltration/injection weighted higher than jailbreak |
| `text_risk_features` | Heuristics | URL presence, base64 patterns, code blocks, length |

**Scoring logic:**
```
base_score = 0

# Classifier confidence (0-3 points)
if confidence >= 0.95: base_score += 3
elif confidence >= 0.85: base_score += 2
elif confidence >= 0.70: base_score += 1

# Anomaly margin (0-3 points)
margin = anomaly_score - threshold
if margin >= 0.50: base_score += 3
elif margin >= 0.25: base_score += 2
elif margin >= 0.10: base_score += 1

# Threat type weight (0-2 points)
if threat_type == "exfiltration": base_score += 2
elif threat_type == "injection": base_score += 2
elif threat_type == "jailbreak": base_score += 1
elif threat_type == "unknown_malicious": base_score += 1

# Text heuristics (0-2 points)
if contains_url: base_score += 1
if contains_base64_or_encoding: base_score += 1

# Map to tiers
Critical: base_score >= 8
High:     base_score >= 5
Medium:   base_score >= 3
Low:      base_score < 3
```

**Academic justification:** Interpretable, auditable decision logic — important for security tooling where explainability matters. Cite weighted scoring matrices from CVSS (Common Vulnerability Scoring System) as precedent.

---

## Component 3: Threat Intelligence Lookup

**What:** Maps the classified threat type to structured intelligence from established security frameworks.

**Sources (recommended):**

1. **MITRE ATLAS** (Adversarial Threat Landscape for AI Systems)
   - Purpose-built for AI/ML threats
   - Structured technique IDs (e.g., AML.T0051 — LLM Prompt Injection)
   - URL: https://atlas.mitre.org
   - Has a public JSON/STIX export we can cache locally

2. **OWASP Top 10 for LLM Applications (2025)**
   - Industry standard for LLM security
   - Maps directly to our threat types (LLM01: Prompt Injection, LLM06: Excessive Agency, etc.)
   - URL: https://genai.owasp.org

**Mapping table (pre-built, cached locally as JSON):**
```json
{
  "injection": {
    "mitre_atlas": {
      "technique_id": "AML.T0051",
      "name": "LLM Prompt Injection",
      "url": "https://atlas.mitre.org/techniques/AML.T0051",
      "tactics": ["ML Attack Staging"],
      "mitigations": ["AML.M0016 — Adversarial Input Detection"]
    },
    "owasp": {
      "id": "LLM01",
      "name": "Prompt Injection",
      "url": "https://genai.owasp.org/llmrisk/llm01-prompt-injection/",
      "description": "...",
      "prevention": ["..."]
    }
  },
  "jailbreak": { ... },
  "exfiltration": { ... }
}
```

**Runtime behavior:**
1. Look up threat type in local cache (JSON file bundled with app)
2. Attempt online fetch for latest data (MITRE ATLAS STIX feed)
3. If online fails, use cached version
4. Return structured intel to frontend

**Artifacts:**
- `models/severity/threat_intel/threat_mapping.json` (pre-built cache)
- `prompt-security-app/app/ml/threat_intel.py` (lookup service)

---

## File Changes Summary

### New files to create:
```
prompt-security-app/app/ml/severity.py          # SeverityService (orchestrates all 3)
prompt-security-app/app/ml/threat_classifier.py  # TF-IDF + XGBoost threat type model
prompt-security-app/app/ml/threat_intel.py        # MITRE/OWASP lookup + caching
prompt-security-app/app/schemas/severity.py       # SeverityResponse schema
models/severity/threat_classifier/                # Model artifacts (after training)
models/severity/threat_intel/threat_mapping.json   # Pre-built intel cache
notebooks/train_threat_classifier.ipynb            # Training notebook
src/data/build_threat_dataset.py                   # Extract multi-class labels from raw data
```

### Files to modify:
```
prompt-security-app/app/api/routes.py      # Add severity to /prompt endpoint
prompt-security-app/app/schemas/decision.py # Extend DecisionResponse with severity fields
prompt-security-app/app/ml/decision.py      # Pass severity into decision logic
prompt-security-app/app/main.py             # Initialize SeverityService on startup
prompt-security-app/app/static/index.html   # Add severity card to UI
prompt-security-app/app/static/app.js       # Render severity + threat intel results
prompt-security-app/app/static/styles.css   # Severity tier color coding
prompt-security-app/requirements.txt        # Add xgboost/lightgbm
```

---

## Execution Order

### Phase 1: Data & Training
1. Build multi-class threat dataset from existing raw sources
2. Train TF-IDF + XGBoost threat type classifier
3. Export model artifacts
4. Validate on held-out test set

### Phase 2: Backend Integration
5. Build `threat_mapping.json` from MITRE ATLAS + OWASP
6. Implement `SeverityService` (threat classifier + severity scorer + threat intel lookup)
7. Create `SeverityResponse` schema
8. Extend `DecisionResponse` schema
9. Update `/prompt` endpoint to call severity when malicious
10. Initialize `SeverityService` in `main.py`

### Phase 3: Frontend
11. Add severity card to `index.html`
12. Update `app.js` to render severity tier, threat type, and intel links
13. Add tier-based color coding to CSS
14. Update history display

### Phase 4: Validation
15. End-to-end test with known malicious prompts
16. Verify benign prompts skip severity (no latency penalty)
17. Test offline fallback (kill network, verify cached intel works)

---

## Schema Preview

```python
class SeverityResponse(BaseModel):
    severity_tier: str          # "low" | "medium" | "high" | "critical"
    severity_score: int         # Raw point score (0-10)
    threat_type: str            # "injection" | "jailbreak" | "exfiltration" | "unknown_malicious"
    threat_confidence: float    # Classifier confidence for threat type
    threat_intel: dict          # MITRE + OWASP references
    scoring_breakdown: dict     # Explainable point-by-point breakdown
```

Extended `DecisionResponse`:
```python
class DecisionResponse(BaseModel):
    anomaly: PredictionResponse
    classification: ClassificationResponse
    severity: SeverityResponse | None    # None when benign (not evaluated)
    final_label: str
    is_malicious: bool
    reasons: list[str]
```
