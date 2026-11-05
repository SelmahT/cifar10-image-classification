# Data

The data is **not stored in this repository**. It comes from the lecturer's shared Google Drive folder .
The data can be made available upon request, subject to the lecturer's permission.

| File | Contents |
|---|---|
| `train.zip` | 50,000 colour images, 32×32 PNG, named `<id>.png` (labelled and test images together) |
| `Train.csv` | 4,000 labelled images: `id`, `label` (400 per class, 10 classes) |
| `Test.csv` | 46,000 images to predict: `id`, `label` (empty) |

**Classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

## How to use it
1. In Google Drive, add the lecturer's shared folder to **My Drive** as a shortcut.
2. Set `DATA_SOURCE` at the top of each notebook to that folder (default: `/content/drive/MyDrive/Advanced ML`).
3. The notebooks copy `train.zip` to Colab's local disk and unzip it automatically (~30 s per session).

## Files created by the notebooks (saved on Google Drive under `data/processed/`)
| File | Created in | Purpose |
|---|---|---|
| `train_folds.csv` | 01 | The 4,000 labelled images with their cross-validation fold (5 folds, 80 per class each) |
| `train_mean_std.npy` | 01 | Channel mean/std of the training images (used by the baseline) |
| `best_params.json` | 03 | Best backbone and hyper-parameters from the Optuna search |
| `test_probs_ensemble.npy` | 04 | Final ensemble probabilities for the 46,000 test images |