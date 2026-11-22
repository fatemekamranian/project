Project Report: Automated Visual Inspection for Corner Parts
Prepared by: Fateme Kamranian
Date: November 17, 2026
Environment: NVIDIA RTX A3000 Laptop GPU (6 GB VRAM) | PyTorch & Anomalib
Executive Summary
•	Result: PaDiM, the best of six tested models, catches 20 of 20 defective parts and passes 993 of 1,000 good parts (threshold 0.495). Training takes about 12 seconds.
•	Risk: Scores of good and defective parts overlap between 0.495 and 0.540. A single pass/fail threshold is therefore not reliable enough for fully automatic rejection.
•	Recommendation: Do not deploy PaDiM as the only decision-maker yet. First improve preprocessing and part alignment.
1. Project Overview
The goal was to build an automated pipeline that detects surface and structural defects on corner parts (270 × 121 px).
Main challenge: Defective parts are very rare in production (only 20–21 available). A supervised classifier could not be trained. Instead, the data was organized in an MVTec-style structure and unsupervised anomaly detection models were trained on good parts only.
Best result (PaDiM, final test: 1,000 good / 20 bad, threshold 0.495):
•	Confusion matrix: TN = 993 | FP = 7 | FN = 0 | TP = 20
•	99.86% AUROC, 99.31% accuracy, 100% recall (20/20 defects caught)
•	7 false alarms (0.7% of good parts scrapped or rechecked unnecessarily)
•	About 12 seconds training time
What this means for production: No defective part was missed (no defect risk reaching the customer), but the overlapping scores make this result sensitive to lighting and part position.
2. Methodology
No single unsupervised algorithm works for every part shape or defect type. The six tested methods work in different ways:
•	Reconstruction-based: reconstruct images or features and look for differences (DRAEM, Reverse Distillation).
•	Statistics- or memory-based: measure pixel statistics at fixed locations (PaDiM) or store normal patches in a memory bank (PatchCore).
•	Probability- or transformer-based: calculate probability scores across scales (CS-Flow) or use Vision Transformers (Dinomaly).
All six were tested across 164 runs because our part combines hard requirements:
•	A wide rectangular shape (270 × 121 px) in the bottom-right corner of the camera frame
•	Two very different defect types: large edge tilts and tiny internal lines
•	Strict hardware limits: fast runtime on a 6 GB GPU and clear heatmaps for visualization
3. Model Comparison and Lessons Learned
Method	Result	Why it failed or fell short	Lesson learned
DRAEM	>30 min training, >6 GB VRAM	Needed an external texture dataset and ran out of memory on the 6 GB GPU.	Check hardware limits before choosing a model, not after 30 minutes of failed training. Heavy dual-network models are not practical on this hardware.
Dinomaly (DINOv2 ViT)	TP = 12/21, FN = 9	Forced 112 × 112 square center-crop cut off the left and right edges of our wide part.	Model input geometry must match the part geometry. A model that crops the edges cannot find defects on the edges.
PatchCore	100 / 21 split, FP = 3–4, up to 5.5 GB GPU	Flagged empty image borders as defects (FP = 3–4), so its results were not reliable without extra masking. It also used 5.5 of 6 GB GPU memory.	A model that reacts to empty borders cannot be trusted without masking. Its high memory use leaves no room to tune it further on a 6 GB GPU.
CS-Flow	TP = 21/21, ~3.8 GB	Good image-level scores, but heatmaps were blurry green blobs that did not show where the defect is.	A good score is not enough. If operators cannot see where the defect is, they will not trust the system.
Reverse Distillation	TP = 18/20, FP = 1 per 1,000	Smoothed features too much and missed 2 very thin internal lines (FN = 2).	A strong model with only 1 false alarm in 1,000 good parts, but it had the same problem as PaDiM: faint defects look too much like normal parts. We tried to improve it, and PaDiM still gave the better result.
PaDiM (WideResNet-50)	TP = 20/20, FP = 7, ~12 s	Learns at fixed (x, y) pixel blocks, so it cannot separate a faint crack from a normal reflection or a slight part shift. Good and bad scores overlap (0.495–0.540).	Best prototype: fast, repeatable, sharp heatmaps. Its weakness is position and lighting sensitivity, which should be fixed in preprocessing.
Strengths of PaDiM
•	Fast and repeatable: Trains in about 12 seconds using only 2.2 GB GPU memory and gives the exact same result every run.
•	Better heatmaps than other models: Sharp heatmaps show where the defect is on the part, which makes results easy for operators to verify.
•	High detection rate: Catches 20/20 bad parts (FN = 0) and passes 993/1,000 good parts (FP = 7) at threshold 0.495.
Limitations of PaDiM
•	Overlapping scores (0.495–0.540): Faint defects and borderline good parts receive similar scores.
•	Unstable threshold: A small change of the threshold causes either false alarms or missed defects.
•	Sensitive to part movement: PaDiM checks fixed (x, y) positions. If a part shifts slightly in the holder, its edges move and cause false alarms.
4. Model Optimization and Troubleshooting: What Failed and What We Learned
To reduce false alarms (FP) and missed defects (FN), we tested several preprocessing and scoring changes.
4.1 Top and left border false alarms
•	Problem: A visible frame edge along the top and left sides of the image was sometimes flagged as a defect.
•	What failed: Removing the border with resize and center-crop in the preprocessor. Standard cropping cuts all four sides and risks cutting off part of the metal component. Default square resizing (256 × 256) also distorted our rectangular part (270 × 121 px).
•	What worked: We kept the true rectangular shape (IMAGE_SIZE = (128, 256)) and wrote a custom function in helpers.py (apply_roi_mask) that ignores anomalies in the top 40 pixels and left 20 pixels of the heatmap. This removed the border false alarms without cropping any part of the component.
•	Lesson learned: Mask the problem area in the result instead of changing the image. Do not use generic preprocessing that does not respect the part geometry.
4.2 Gaussian blur
•	What we tried: A Gaussian blur filter to smooth out shiny metal surface noise.
•	What happened: Performance got worse. Blurring smoothed over thin, faint internal lines and micro-cracks, so the model missed them.
•	Lesson learned: Generic noise filters can remove the exact signal we are searching for. Lighting problems need edge-preserving methods, not blur.
4.3 Checking pixel errors on FP and FN images
•	What we tried: Inspected the raw pixel values of every misclassified image for missing or dead pixels (value = 0) and wrong image dimensions.
•	What we found: Every FP and FN image had 0 missing pixels.
•	Lesson learned: This ruled out broken camera pixels and corrupted files as the cause. The remaining errors come from the model and its scoring, so effort should go there.
4.4 Improving defect scoring (Top-2% area score + layer1 features)
•	Problem: By default, Anomalib scores the whole image using only the single brightest pixel in the heatmap. A tiny 1-pixel light reflection on a good part caused a false alarm, while a long faint crack was missed because no single pixel was bright enough. Removing layer1 (which keeps fine image details) also reduced AUROC.
•	Solution: We kept layer1 + layer2 + layer3 and changed the score formula to 40% of the brightest pixel + 60% of the average of the top 2% brightest pixels. This ignores 1-pixel glare and helped PaDiM catch all 20 of 20 bad parts.
•	Lesson learned: Scoring on the area of the anomaly is more robust than scoring on a single pixel. Fine-detail features (layer1) are required for thin defects.
5. Final Conclusion and Next Steps
PaDiM is the best of the six tested models and a strong prototype, but we cannot trust it on the production line yet. Unsupervised models only learn from good parts, so they flag any visual difference from a normal image. Even after fixing the image shape (128 × 256), masking the top/left borders (40 px top, 20 px left) and using top-2% area scoring, faint defects on bad parts (0.500–0.535) get the same scores as normal lighting or position changes on good parts (0.495–0.540). Because of this overlap, a single pass/fail threshold is not stable enough for fully automatic rejection.
Recommended next steps (in priority order):
1.	Automatically align the part position (dynamic cropping). Instead of a fixed border mask, detect the outer edges of the part and align every image to the same position before running the model. This should reduce false alarms when the part shifts slightly in the camera frame. It targets the biggest known weakness of PaDiM and needs no new data.
2.	Use better preprocessing (lighting normalization and edge-preserving filters). Gaussian blur and CLAHE hurt performance because they either erased thin cracks or boosted random background noise. Lighting normalization removes bright surface reflections, and edge-preserving filters keep normal surfaces smooth without blurring fine cracks. We still need to test this. If a defect is very faint, preprocessing may not fully solve the overlap.
Limit of the current result: The 20/20 recall and the 0.495 threshold come from only 20 bad parts. Recall on new data may be lower, so the threshold should be re-validated as more bad parts appear in production.
Key takeaway: The six models showed that for this part, the quality of the input (alignment, lighting, and geometry-aware preprocessing) matters more than choosing a more complex model.
