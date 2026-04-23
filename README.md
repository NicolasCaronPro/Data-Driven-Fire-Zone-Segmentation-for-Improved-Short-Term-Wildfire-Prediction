# Data-Driven Fire Zone Segmentation for Improved Short-Term Wildfire Prediction

This repository focuses on **watershed-based fire zone segmentation** from rasterized wildfire risk maps.

![Texte alternatif](images/schema.png)

## Abstract

Wildfire prediction models typically discretize study areas into uniform grids, ignoring the heterogeneous spatial distribution of ignitions. We challenge this paradigm by showing that *how* data is discretized matters more than *which* model is used. We propose an unsupervised fire-zone segmentation algorithm combining watershed detection with K-means clustering to define prediction units directly from historical fire patterns. Experiments across six French departments and six forecasting models show that fire-zone segmentation consistently outperforms grid-based approaches, with mean IoU improvements of +3--6% depending on spatial scale. The method is computationally lightweight (<10s per configuration) and fully parallelizable. Our results demonstrate that optimizing spatial discretization yields significant, reproducible performance gains for short-term wildfire forecasting.

## What this project does

The segmentation pipeline:
1. Loads a risk raster and a valid-pixel mask.
2. Reduces the risk values into discrete classes.
3. Computes edges (Sobel), distance transform, and markers.
4. Runs the **watershed** algorithm.
5. Post-processes watershed regions with size-based merging/splitting.
6. Saves intermediate feature maps and final predictions.

The implementation is centered on `Segmentation.create_geometry_with_watershed` in `segmentation.py`.

## Main script to highlight

The most useful entry point for quick verification is:

- `test_simple_segmentation.py`

This script performs a grid search over:
- `scale` in `[1, 2, 3, 4, 5]`
- `attempt` in `[1, 2, 3, 4, 5]`
- `reduce` in `[2, 3, 4, 5, 6]`

For each combination, it computes IoU and writes:
- a CSV summary: `test_output/grid_search_results.csv`
- a prediction image: `test_output/pred_s{scale}_a{attempt}_r{reduce}.png`

## Key result to inspect

Please review this output in priority:

- `test_output/pred_s4_a2_r3.png`

It is the reference result highlighted for the simple watershed segmentation test.

## Run the simple segmentation test

```bash
python test_simple_segmentation.py
```

## Outputs generated

- Feature visualizations: `features_geometry/*`
- Watershed objects: `features_geometry/watershed_*.pkl`
- Grid-search predictions and metrics: `test_output/*`
