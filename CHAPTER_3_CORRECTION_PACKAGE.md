# Chapter 3 Correction Package for Reproducible Chapter 4 Implementation

## 1) Scope and Deliverable Boundary (Locked)

This project delivers a **simulated prototype** for AI-based malaria blood-smear image classification and performance evaluation.  
It does **not** claim real clinical deployment, regulatory approval, or replacement of medical diagnosis.

### Chapter 4 Deliverables (Explicit)
- End-to-end reproducible training and evaluation pipeline.
- Trained CNN model for malaria image classification.
- Baseline model comparison.
- Performance report against predefined acceptance thresholds.
- Prototype inference workflow with class output, confidence score, and heatmap explanation.
- Simulated usability check with intended users/tasks.

---

## 2) Reproducible Implementation Specification (Fixed Blueprint)

## 2.1 Dataset
- **Primary dataset**: NIH Malaria Cell Images Dataset (Labeled parasitized/uninfected cell images).  
- **Source**: https://ceb.nlm.nih.gov/repositories/malaria-datasets/  
- **Frozen acquisition date**: `2026-04-24`  
- **Frozen local dataset ID for this project**: `NIH_Malaria_Cell_Images_2026-04-24`

## 2.2 Inclusion/Exclusion Rules
- Include only RGB cell image files with valid labels in `{Parasitized, Uninfected}`.
- Exclude corrupted, unreadable, duplicate-hash files, and unlabeled items.
- Exclude non-image metadata files from modeling.

## 2.3 Preprocessing Pipeline (Deterministic)
1. Load image and convert to RGB.
2. Resize to `128 x 128`.
3. Pixel normalization to `[0,1]`.
4. Training-time augmentation only:
   - Random horizontal flip (`p=0.5`)
   - Random rotation (`±15°`)
   - Random zoom (`0.9–1.1`)
   - Width/height shift (`±10%`)
5. Validation/test sets receive **no random augmentation**.

## 2.4 Data Split Protocol
- Split ratio: **70% train / 15% validation / 15% test**
- Stratified by class label.
- Fixed random seed: **42**
- Leakage control:
  - No overlap of exact images across splits.
  - Deduplicate by perceptual hash before splitting.
  - If patient/source IDs are available, enforce no patient overlap across splits.
- Save split manifests to files:
  - `train_manifest.csv`
  - `val_manifest.csv`
  - `test_manifest.csv`

## 2.5 CNN Architecture (Final Locked Model)
- Input: `128 x 128 x 3`
- Block 1: Conv2D(32, 3x3, ReLU) → MaxPool(2x2)
- Block 2: Conv2D(64, 3x3, ReLU) → MaxPool(2x2)
- Block 3: Conv2D(128, 3x3, ReLU) → MaxPool(2x2)
- Flatten
- Dense(128, ReLU)
- Dropout(0.5)
- Output Dense(1, Sigmoid)

## 2.6 Training Configuration
- Loss: Binary cross-entropy
- Optimizer: Adam (`learning_rate=0.001`)
- Batch size: 32
- Epochs: max 30
- Early stopping: patience 5, monitor `val_loss`, restore best weights
- Class imbalance handling: class weights computed from training split

## 2.7 Baseline Model
- Baseline: Logistic Regression on flattened, normalized grayscale features.
- Same train/validation/test split as CNN.
- Used for objective performance comparison.

## 2.8 Evaluation Metrics and Benchmarks (Pass/Fail)
- Accuracy
- Sensitivity (Recall for parasitized class)
- Specificity (Recall for uninfected class)
- F1-score
- Inference time per image (ms)

### Minimum Acceptance Thresholds
- Accuracy `>= 0.92`
- Sensitivity `>= 0.90`
- Specificity `>= 0.90`
- F1-score `>= 0.90`
- Mean inference time `<= 300 ms/image` on stated hardware

Project success in Chapter 4 requires:
1. CNN meets all minimum thresholds, and  
2. CNN outperforms baseline on F1-score and sensitivity.

## 2.9 Validation and Statistical Comparison
- Primary strategy: fixed holdout split (70/15/15) with seed 42.
- Stability check: 5 repeated runs with seeds `{42, 52, 62, 72, 82}`.
- Report mean ± standard deviation for all metrics.
- Statistical test for CNN vs baseline F1-score: paired t-test (`alpha = 0.05`).

## 2.10 Software and Hardware Environment
- Python 3.10
- TensorFlow 2.13
- scikit-learn 1.3
- NumPy 1.24
- pandas 2.0
- matplotlib 3.7
- OS: Ubuntu 22.04 (or equivalent)
- Hardware assumption:
  - Minimum: 8 GB RAM, CPU-only supported
  - Recommended: NVIDIA GPU with >= 6 GB VRAM

## 2.11 End-to-End Experiment Workflow
1. Acquire and freeze dataset snapshot.
2. Apply inclusion/exclusion filtering and deduplication.
3. Generate stratified leakage-controlled split manifests.
4. Train baseline model and record metrics.
5. Train CNN with fixed hyperparameters and callbacks.
6. Evaluate on validation set for model selection.
7. Evaluate final locked model on test set.
8. Compute confusion matrix and all target metrics.
9. Measure inference time.
10. Generate interpretability heatmaps for selected test samples.
11. Compare CNN vs baseline and check pass/fail thresholds.
12. Produce final reproducible report table and conclusion.

---

## 3) Clinical-Support Claim Operationalization (Prototype Level)

## 3.1 Intended User Workflow
1. User uploads a blood-smear cell image.
2. System returns predicted class (`Parasitized` or `Uninfected`).
3. System displays confidence score (0–1).
4. System displays interpretability heatmap highlighting influential regions.
5. User reviews output as decision-support, not final diagnosis.

## 3.2 Interpretability Output Requirement
- Use Grad-CAM (or equivalent saliency method) for model explanation.
- Show heatmap overlaid on original image.
- Save explanation artifact for audit/reporting.

## 3.3 Simulated Usability Check Procedure
- Participants: minimum 5 intended users (e.g., lab/science students or clinicians in simulation).
- Tasks:
  - Upload sample image
  - Interpret predicted class and confidence
  - Interpret heatmap
  - Record decision confidence
- Measures:
  - Task completion rate
  - Time to complete task
  - Post-task usability score (SUS or 5-point Likert equivalent)
- Acceptance criterion:
  - >= 80% successful task completion
  - Mean usability score >= 3.5/5 (or SUS >= 68)

---

## 4) Literature-to-Design Traceability Table

| Design Decision | Evidence from Prior Studies | Project Constraint/Reason | Final Choice |
|---|---|---|---|
| CNN for malaria image classification | Prior malaria imaging studies report strong CNN performance over manual feature methods | Need robust automatic feature learning | 3-block CNN classifier |
| Input resizing and normalization | Standard deep learning practice for stable training | Ensure reproducible, memory-feasible training | 128x128 + [0,1] normalization |
| Metrics beyond accuracy | Medical classification requires balanced error awareness | False negatives are critical in screening | Accuracy, Sensitivity, Specificity, F1 |
| Interpretability output | Clinical-support contexts require explainability | User trust and auditability | Grad-CAM heatmaps |
| Prototype-only deployment | Undergraduate project and simulated setting constraints | Avoid over-claiming real-world readiness | Simulated decision-support prototype |

---

## 5) Rewritten Core Sections

## 5.1 Revised Problem Statement
In low-resource settings, malaria diagnosis is often delayed or inaccurate because microscopy depends heavily on skilled personnel and manual interpretation. Existing AI studies are not consistently translated into practical, reproducible diagnostic support prototypes suitable for these constraints. This project addresses that gap by defining and implementing a reproducible AI-based malaria screening and diagnosis support prototype using blood smear images in a simulated environment.

## 5.2 Revised Aim
To design, implement, and evaluate a reproducible AI-driven malaria screening and diagnosis support prototype for blood smear image classification in a simulated low-resource setting.

## 5.3 Revised Objectives
1. Identify and document the exact public blood smear dataset(s), inclusion/exclusion criteria, class balance strategy, and leakage-controlled train/validation/test split protocol.  
2. Develop a fixed CNN-based pipeline by specifying preprocessing steps, model architecture, training hyperparameters, and software/hardware environment.  
3. Implement and run the end-to-end prototype to generate malaria classification outputs and confidence information in a simulated user workflow.  
4. Evaluate the prototype against predefined acceptance thresholds using accuracy, sensitivity, specificity, F1-score, and inference time, and compare results with at least one baseline model.  
5. Analyze results and state whether the prototype meets the defined success criteria for Chapter 4 completion.

---

## 6) Chapter 3-to-4 Readiness Statement

With the above locked specifications, Chapter 4 now has:
- Reproducible inputs,
- Controlled data protocol,
- Objective acceptance criteria,
- Testable clinical-support behavior in prototype scope,
- Clear and realistic implementation boundaries.

Therefore, implementation can proceed in an auditable, step-by-step manner.
