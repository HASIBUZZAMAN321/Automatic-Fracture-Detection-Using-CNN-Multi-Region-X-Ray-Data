# Reproduction package for the Life 2025 fracture-detection paper

Files
- model.py: proposed SE + CBAM + spatial-attention CNN.
- preprocess.py: paper-described normalization, Gaussian/median filtering,
  flips, +/-20-degree rotation and scaling.
- train.py: Adam, lr=0.001, batch=32, 100 epochs.
- evaluate.py: accuracy, precision, recall, F1, Cohen kappa, confusion matrix.
- plots.py: learning curves.
- predict.py: single-image inference.
- statistics.py: paired t-test + McNemar helpers.
- requirements.txt

Expected dataset:
dataset/
  train/
    fractured/
    non_fractured/
  val/
    fractured/
    non_fractured/
  test/
    fractured/
    non_fractured/

1. Dropout: Results states 0.3; Discussion states 0.5 after fully-connected layers.
   This package uses 0.3 in convolutional dropout and 0.5 after the FC layer.
2. CBAM spatial kernel: Method Eq. (3) states 3x3, while the detailed attention
   description later specifies 7x7. model.py exposes cbam_spatial_kernel.
3. Dense-layer neuron count is not specified.
4. Gaussian and median filter kernel sizes are not specified; preprocess.py uses 3x3
   as a practical choice, not as a claimed paper value.
5. Scaling range/magnitude is not numerically specified; RandomZoom(0.10) is an
   implementation choice.
6. The paper says hyperparameters were tuned by grid search, but does not provide
   the search grid or ablation table values.
7. The reported parameter count is 1.58 million; exact matching cannot be guaranteed
   without the unspecified Dense size and all original implementation details.
