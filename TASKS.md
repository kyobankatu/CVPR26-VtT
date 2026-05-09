# TASKS.md

## Objective
Improve the current VtT-based model on EuroSAT 1-shot without hurting the main CLIP+LoRA branch.

Current reference:

- Baseline `logs/EuroSAT_1.log`: `85.1417 | 80.9933 | 64.3067 | 85.13`
- Current model `logs/VtT.79340`: `85.5417 | 36.5517 | 66.4733 | 83.89`

Interpretation:

- `fine_acc` is slightly better than baseline.
- The support-derived VtT branch is severely degraded.
- The final combined score is lower because the auxiliary branch is unstable.

---

## Working Hypothesis
The main regression is not in the CLIP+LoRA branch itself. It is likely caused by:

1. Train/test mismatch in the Mamba input path.
2. An attribute-alignment loss defined across incompatible representation spaces.
3. Weak class-level consistency constraints for absorber-token outputs.
4. Over-strong auxiliary weighting (`beta`, `lambda_attr`) before the auxiliary branch is stable.

---

## Phase 1: Fix the Highest-Risk Evaluation Mismatch
Goal: make training-time and test-time Mamba inputs follow the same feature definition.

- [ ] Inspect the Mamba input construction in `VtT.py`.
- [ ] Align the first token used in `cat_input` between train and test.
  - Train currently uses `image_features_raw`.
  - Test currently uses normalized `supp_image_features`.
- [ ] Decide one consistent choice and apply it to both paths.
  - Default choice: use raw features in both paths.
- [ ] Verify tensor shapes and dtype consistency after the change.
- [ ] Run EuroSAT 1-shot and compare:
  - `fine_acc`
  - support-derived branch accuracy
  - final combined accuracy

Files:

- `VtT.py`

Success criterion:

- The support-derived branch should recover substantially from `36.55%`.

---

## Phase 2: Rework the Attribute Alignment Loss
Goal: stop forcing absorber tokens to match a semantically different feature space.

- [ ] Review the current auxiliary loss in `VtT.py`.
  - Current design compares `absorber_tokens` directly against `clip_model.encode_text(attr_tokens)`.
- [ ] Replace the current loss with one of the following safer variants:
  - Variant A: compare injected-text outputs against attribute-text outputs in the final CLIP embedding space.
  - Variant B: compare absorber tokens against token-embedding-space attribute anchors instead of final text embeddings.
- [ ] Implement Variant A first because it is closer to the current inference path.
- [ ] Keep the old implementation behind a small switch or comment block for ablation.
- [ ] Re-run EuroSAT 1-shot and record whether:
  - support-derived branch improves
  - `fine_acc` remains stable

Files:

- `VtT.py`
- possibly `clip/model.py` if token-level utilities are needed

Success criterion:

- Auxiliary branch improves without reducing `fine_acc` below the current `85.54%`.

---

## Phase 3: Add Same-Class Consistency for Absorber Outputs
Goal: make support examples from the same class produce compatible restored text representations.

- [ ] Add a class-consistency loss on top of the current VtT path.
- [ ] Test one simple version first:
  - average absorber-derived text features within each class
  - pull each sample toward its class prototype
- [ ] If needed, try a supervised-contrastive variant that treats same-class samples as positives.
- [ ] Keep this loss lightweight and numerically stable.
  - Use `float()` when needed around similarity/log-softmax code paths.
- [ ] Measure impact on:
  - support-derived branch
  - combined branch
  - `fine_acc`

Files:

- `VtT.py`

Success criterion:

- Same-class support features cluster better and the auxiliary branch becomes less noisy across episodes.

---

## Phase 4: Retune Auxiliary Weights After the Branch is Stable
Goal: avoid letting an unstable auxiliary loss damage the main branch.

- [ ] Sweep `beta` after Phases 1 to 3 are in place.
  - Try at least `1`, `2`, `4`, `7`.
- [ ] Sweep `lambda_attr`.
  - Try at least `0.0`, `0.02`, `0.05`, `0.1`.
- [ ] Keep `gamma` fixed initially unless orthogonal loss appears ineffective.
- [ ] Select the setting using combined accuracy first, with `fine_acc` as a hard constraint.

Files:

- `run_utils.py` for exposed arguments if additional controls are needed
- experiment logs

Success criterion:

- Combined score exceeds the current `83.89%` and ideally approaches or surpasses the baseline `85.13%`.

---

## Phase 5: Improve Diagnostics and Logging
Goal: make ablations readable and reduce wasted experiment cycles.

- [ ] Restore real intermediate evaluation instead of writing fixed zero placeholders.
- [ ] Log per-episode metrics with stable names.
- [ ] Print active hyperparameters at launch:
  - `beta`
  - `gamma`
  - `lambda_attr`
  - `num_attrs`
  - encoder mode
- [ ] Log the final summary in a format that is easy to compare across runs.
- [ ] If useful, save a compact CSV or TXT summary per run.

Files:

- `VtT.py`
- optionally `main.py`

Success criterion:

- Each run is directly comparable without manual decoding of the log format.

---

## Phase 6: Refresh EuroSAT Attribute Prompts if Needed
Goal: improve the quality of class attributes used by the VtT branch.

- [ ] Review the current EuroSAT entries in `data/semantic_attributes.json`.
- [ ] If the branch is still weak after Phases 1 to 4, regenerate EuroSAT attributes with a more dataset-specific prompt.
- [ ] Prefer prompt wording tied to satellite imagery, for example:
  - land-cover texture
  - spatial pattern
  - color distribution
  - man-made vs natural structure
- [ ] Re-run with old vs new attributes as a clean ablation.

Files:

- `scripts/generate_attributes.py`
- `data/semantic_attributes.json`

Success criterion:

- New attributes improve the auxiliary branch without changing the rest of the pipeline.

---

## Execution Order
Implement and test in this order:

1. Phase 1
2. Phase 2
3. Phase 3
4. Phase 4
5. Phase 5
6. Phase 6 only if needed

---

## Experiment Notes Template
Use this template for each run:

- Run ID:
- Code change:
- Dataset / shot:
- Key args:
- `fine_acc`:
- support-derived branch:
- calibration branch:
- combined branch:
- Main observation:
- Keep / revert:
