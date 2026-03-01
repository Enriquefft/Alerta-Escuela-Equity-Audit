# Requirements: Alerta Escuela Equity Audit

**Defined:** 2026-03-01
**Core Value:** The fairness audit is the product. Models exist to be audited, not to achieve SOTA.

## v4.0 Requirements

Requirements for model experimentation milestone. Each maps to roadmap phases.

### Feature Engineering

- [ ] **FEAT-01**: Construct overage-for-grade feature from age and enrollment level (Module 300)
- [x] **FEAT-02**: Assess ENAHO panel linkage feasibility (% of sample linkable across waves)
- [x] **FEAT-03**: If panel linkage >40%, build trajectory features (income change, sibling dropout, work transitions)
- [ ] **FEAT-04**: Engineer interaction features (age x working, age x poverty, rural x parental_education, secondary_age x income)

### Model Training

- [ ] **MODEL-01**: Retrain LightGBM with new features, re-optimize with Optuna
- [ ] **MODEL-02**: Retrain all 5 model families (LR, LightGBM, XGBoost, RF, MLP) with new features
- [ ] **MODEL-03**: Re-calibrate primary LightGBM (Platt scaling) and re-export ONNX
- [ ] **MODEL-04**: Compare PR-AUC before/after for each model family

### Fairness Analysis

- [ ] **FAIR-01**: Re-run full fairness pipeline with updated primary model
- [ ] **FAIR-02**: Assess whether FNR disparity persists, narrows, or disappears
- [ ] **FAIR-03**: Check cross-architecture consistency with all 5 updated models

### Interpretation & Paper

- [ ] **PAPER-01**: Interpret results per outcome scenario (persist/narrow/disappear/ceiling)
- [ ] **PAPER-02**: Update paper tables, figures, and text with new results
- [ ] **PAPER-03**: Document panel linkage outcome (positive or negative) in Limitations

## Future Requirements

### Media & Outreach

- **MEDIA-01**: Press materials (summaries, blog posts, pitch emails in ES/EN)
- **MEDIA-02**: Pitch media outlets

## Out of Scope

| Feature | Reason |
|---------|--------|
| Bias mitigation / debiasing | Audit scope only — not building corrective models |
| Deep learning beyond MLP | Tabular data, diminishing returns past tree models + MLP |
| SIAGIE data integration | No access — documented as limitation |
| New fairness dimensions | Existing 7 dims + 3 intersections sufficient |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| FEAT-01 | Phase 26 | Pending |
| FEAT-02 | Phase 26 | Complete |
| FEAT-03 | Phase 26 | Complete |
| FEAT-04 | Phase 26 | Pending |
| MODEL-01 | Phase 27 | Pending |
| MODEL-02 | Phase 27 | Pending |
| MODEL-03 | Phase 27 | Pending |
| MODEL-04 | Phase 27 | Pending |
| FAIR-01 | Phase 28 | Pending |
| FAIR-02 | Phase 28 | Pending |
| FAIR-03 | Phase 28 | Pending |
| PAPER-01 | Phase 29 | Pending |
| PAPER-02 | Phase 29 | Pending |
| PAPER-03 | Phase 29 | Pending |

**Coverage:**
- v4.0 requirements: 14 total
- Mapped to phases: 14
- Unmapped: 0

---
*Requirements defined: 2026-03-01*
*Last updated: 2026-03-01 after roadmap creation*
