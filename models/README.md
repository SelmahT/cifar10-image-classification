# Models

Trained weights (.pt) are saved on Google Drive, not in GitHub (about 500 MB in total).

| Files | What they are |
|---|---|
| `folds/vit_small_fold{0-4}.pt` | 5 × ViT-Small, one per cross-validation fold |
| `folds/convnext_tiny_fold{0-4}.pt` | 5 × ConvNeXt-Tiny, one per cross-validation fold |
| `baseline_cnn.pt` | Small ResNet trained from scratch (baseline) |
| `optuna/cat2_study.db` | Optuna study with all 20 hyper-parameter trials |

The final prediction is the average of the 10 fold models (with horizontal-flip TTA).

**To reproduce:** run notebooks 03 and 04 on Google Colab (T4 GPU, about 1.5 hours in total).
The weights can also be made available upon request.
