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

---

## Original VtT Improvement Track
Target: improve the original model in `CVPR-VtT-main` without relying on semantic attributes.

### Phase O1: Prompt Redesign for EuroSAT
Goal: replace the generic natural-image prompt with a satellite-imagery-aware prompt and measure whether the text side becomes more compatible with EuroSAT.

- [ ] Locate every place where the original model builds class prompts.
- [ ] Replace the default prompt
  - from: `a photo of a {}.`
  - to: `a satellite image of {}.`
- [ ] Keep the prompt string configurable from the command line instead of hard-coding a single template.
- [ ] Add a new argument such as `--prompt_template`.
- [ ] Use the new argument in both:
  - support-time text feature construction
  - evaluation-time text feature construction
- [ ] Run an ablation on EuroSAT 1-shot with:
  - baseline prompt: `a photo of a {}.`
  - satellite prompt: `a satellite image of {}.`
- [ ] If the satellite prompt helps, test one stronger variant:
  - `a satellite image of {} land cover.`

---

## Current Non-Attribute Track: Failed Align/Variance Regularization

Goal: document the recent non-attribute attempt and avoid re-testing the same weak direction.

### Result Summary

The `image_mae_encode` visual-alignment and variance-regularization approach was tested with:

- `lambda_align`
- `lambda_var`

The best EuroSAT setting observed was around:

- `lambda_align=0.01`
- `lambda_var=0.002`

However, after comparing across datasets, this direction degraded performance on all datasets. Treat this as a failed direction unless a substantially different formulation is introduced.

Likely reason:

- The paper's strongest inference setting removes the VtT auxiliary branch at test time.
- Directly constraining `image_mae_encode` can still distort LoRA/CLIP adaptation during training.
- Strong alignment can help early episodes but later restricts useful task adaptation.

Decision:

- [x] Do not continue broad sweeps of `lambda_align` / `lambda_var`.
- [ ] Remove or disable this branch before the next main experiment unless it is needed for ablation records.

Files:

- `VtT.py`
- `run_utils.py`
- `execute.sh`

---

## Next Improvement Direction: Control the VtT Auxiliary Loss Instead of Adding New Feature Losses

Goal: make VtT helpful during training without letting its auxiliary objective damage the main CLIP+LoRA branch.

Rationale:

- Since VtT is removed at inference in the strongest reported setting, the main goal is not to make the VtT branch itself stronger.
- The safer target is to prevent `mae_loss` from pulling the trainable LoRA parameters away from the classification objective.
- The current code already computes gradient similarity between `ce_loss` and `mae_loss`, but the beta update is coarse and can be made more stable.

### Phase N1: Beta Warmup

Add a schedule for the auxiliary weight:

```text
effective_beta = beta * min(1, step / beta_warmup_steps)
```

Tasks:

- [ ] Add `--beta_warmup_steps` to `run_utils.py`.
- [ ] Use `effective_beta` instead of raw `Used_beta` inside `VtT.py`.
- [ ] Start with:
  - `beta_warmup_steps=50`
  - `beta=7`
- [ ] Compare against baseline on:
  - EuroSAT
  - CropDisease
  - ISIC
  - ChestX

Expected benefit:

- Avoids applying a strong VtT auxiliary signal before the Mamba branch is stable.

### Phase N2: Conflict-Aware Beta Gating

Use gradient agreement between `ce_loss` and `mae_loss` to decide whether the auxiliary objective should be active.

Proposed rule:

```text
if mean_sim < beta_conflict_threshold:
    effective_beta = beta * beta_min_scale
else:
    effective_beta = beta
```

Initial settings:

- `beta_conflict_threshold=0.0`
- `beta_min_scale=0.0`

Alternative softer setting:

- `beta_conflict_threshold=0.1`
- `beta_min_scale=0.2`

Tasks:

- [ ] Add `--beta_conflict_threshold`.
- [ ] Add `--beta_min_scale`.
- [ ] Replace the current `get_grad_beta_updatae()` behavior with a bounded scale instead of returning `-1`.
- [ ] Log the average effective beta per episode.

Expected benefit:

- Suppresses VtT training signal only when it conflicts with classification.
- Preserves VtT when it aligns with the main task.

### Phase N3: Class-Prototype Consistency for VtT Outputs

If N1/N2 are stable, add a lightweight class-consistency term for the VtT-produced text features.

Design:

- Build class prototypes from `mae_text_features` inside each support episode.
- Pull same-class VtT outputs toward their class prototype.
- Keep the coefficient small.

Initial setting:

- `lambda_proto=0.01`

Tasks:

- [ ] Add `--lambda_proto`.
- [ ] Implement prototype consistency on normalized `mae_text_features`.
- [ ] Run only after beta warmup/gating is validated.

Expected benefit:

- Makes the auxiliary branch less noisy without forcing it into image-feature space.

---

## Next Structural Direction: Text Residual Absorb Token

Goal: improve the absorb token construction while preserving the paper's finding that all text layers should be used.

Decision:

- Do not pursue layer selection or layer gating because the paper reports that using all text layers is best.
- Instead, keep the all-layer Mamba input unchanged and change only the way the final absorb token is formed.

### Motivation

The failed variants pushed `image_mae_encode` toward the wrong target:

- `adv` made it too text-token-like.
- `align/var` tied it too directly to image features.
- `beta_warmup` weakened the original auxiliary signal and reduced performance.

The new hypothesis is that Mamba should not synthesize the full absorb token from scratch. It should predict a visual residual on top of the original class text token.

### Proposed Form

Instead of:

```text
absorb_token = mamba_output
```

use:

```text
absorb_token = class_text_token_at_position_5 + residual_scale * mamba_output
```

Initial settings:

- `residual_scale=0.05`
- `residual_scale=0.1`
- `residual_scale=0.2`

Tasks:

- [x] Add `--residual_scale`; use a negative value to preserve the original behavior.
- [x] Apply the residual absorb token in the training path.
- [x] Apply the same residual absorb token in `fsl_test`.
- [ ] Run EuroSAT 1-shot with:
  - baseline: `--residual_scale -1`
  - residual: `--residual_scale 0.1`
- [ ] If EuroSAT improves, test CropDisease, ISIC, and ChestX.

Expected benefit:

- The class text token preserves semantic identity.
- Mamba only needs to learn visual correction.
- The all-layer VtT input path remains intact.

---

## Recommended Next Commands

First validate the baseline and the text-residual variant with the same seed.

Baseline:

```bash
ybatch execute.sh --dataset EuroSAT --seed 1 --residual_scale -1
```

Residual absorb token:

```bash
ybatch execute.sh --dataset EuroSAT --seed 1 --residual_scale 0.1
```

Scale sweep:

```bash
ybatch execute.sh --dataset EuroSAT --seed 1 --residual_scale 0.05
ybatch execute.sh --dataset EuroSAT --seed 1 --residual_scale 0.2
```

Files:

- `main.py`
- `VtT.py`
- `run_utils.py`

Success criterion:

- `fine_acc` improves over the original EuroSAT baseline without introducing instability in the auxiliary branches.

### Phase O2: Token-Space Adversarial Alignment
Goal: make `image_mae_encode` look like a valid text token before it enters the CLIP text encoder, while keeping the existing image-text similarity loss.

- [ ] Keep the current prompt-template support from Phase O1.
- [ ] Do not change the final fusion rule yet.
  - Keep the current fixed `0.5 / 0.5` combination as a control.
- [ ] Add a small discriminator that distinguishes:
  - real text-token-like vectors
  - fake vectors produced by `Mamba_Net`
- [ ] Use token-space adversarial training, not final text-embedding adversarial training.
  - Reason: final-embedding adversarial loss is more likely to compete directly with `mae_loss`.
- [ ] Define the fake samples as:
  - `image_mae_encode`
- [ ] Define the real samples as one of:
  - averaged class-name token embeddings
  - class-token-region average from the prompt input embedding sequence
- [ ] Start with the simpler real-target definition:
  - averaged class-name token embeddings
- [ ] Add a lightweight discriminator, for example:
  - `Linear(512, 256) -> ReLU -> Linear(256, 1)`
- [ ] Train the discriminator to classify:
  - real token-space samples as `1`
  - fake token-space samples as `0`
- [ ] Train `Mamba_Net` adversarially so its outputs are classified as real.
- [ ] Keep the existing `mae_loss` unchanged.
  - The adversarial loss should be an additional regularizer.
- [ ] Start with a small adversarial weight:
  - `0.01`
  - `0.05`
  - `0.1`
- [ ] Log separately:
  - `ce_loss`
  - `mae_loss`
  - discriminator loss
  - generator adversarial loss
- [ ] Evaluate primarily on:
  - `fine_acc` (leftmost value in `full acc is ...`)

Files:

- `VtT.py`
- `run_utils.py`
- possibly `clip/model.py` if token-level extraction helpers are needed

Success criterion:

- `fine_acc` improves over the original EuroSAT baseline without relying on semantic attributes.

### Phase O2A: Build Real Token Targets
Goal: create stable "real text token" targets for adversarial training.

- [ ] Inspect CLIP tokenization for class prompts.
- [ ] Extract the token-embedding sequence before the text transformer.
- [ ] Identify the class-name token span inside the prompt.
- [ ] Compute a pooled real token target from that span.
- [ ] Verify the pooled target has the same dimensionality as `image_mae_encode`.
- [ ] Sanity-check with a few EuroSAT class names that tokenize into multiple pieces.

Files:

- `clip/model.py`
- `VtT.py`

Success criterion:

- Real token targets are consistent across classes and compatible with the fake token shape.

### Phase O2B: Add the Token Discriminator
Goal: train a discriminator to separate genuine text-token vectors from Mamba-generated vectors.

- [ ] Implement a small discriminator module in `VtT.py`.
- [ ] Add a separate optimizer for discriminator parameters.
- [ ] Keep discriminator training detached from the main generator path where appropriate.
- [ ] Train the discriminator on:
  - real pooled class-token targets
  - fake `image_mae_encode`
- [ ] Track discriminator accuracy to avoid collapse.

Files:

- `VtT.py`

Success criterion:

- Discriminator learns a meaningful boundary without immediately saturating at 100%.

### Phase O2C: Adversarially Train `Mamba_Net`
Goal: force `Mamba_Net` to produce token-space vectors that the discriminator cannot reliably distinguish from real text tokens.

- [ ] Choose one training scheme:
  - Gradient Reversal Layer
  - alternating optimization
- [ ] Start with alternating optimization if implementation clarity is better.
- [ ] Add generator-side adversarial loss on `image_mae_encode`.
- [ ] Combine losses as:
  - existing `ce_loss`
  - existing `beta * mae_loss`
  - small `lambda_adv * adv_loss`
- [ ] Sweep only a few `lambda_adv` values first.
- [ ] Check that `fine_acc` is the main model-selection metric.

Files:

- `VtT.py`

Success criterion:

- `image_mae_encode` becomes more text-like in token space and `fine_acc` improves.

### Recommended Execution Order for the Original Model
Implement and test in this order:

1. Phase O1 with only the prompt change
2. Phase O2A to build real token targets
3. Phase O2B to verify discriminator behavior
4. Phase O2C to enable adversarial training
5. Keep the best prompt + adversarial setting as the new original-model baseline
