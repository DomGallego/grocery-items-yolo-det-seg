# Grocery Items YOLO Detection and Segmentation

This repository provides:
- **main.py**: Launches a Gradio app for real-time detection/segmentation of grocery items.
- **batch_train.py**: Trains multiple YOLO models (detection/segmentation).
- **batch_metrics.py**: Evaluates model metrics and inference speeds.
- **batch_export.py**: Exports models to TensorRT format.

## Quick Start
1. Go through [data_preprocessing.ipynb](data_preprocessing.ipynb) to prepare your data before running the scripts.

2. Install dependencies:
   ```bash
   pip install ultralytics gradio opencv-python tensorrt
   ```

3. Run the Gradio app:
    ```python
    python main.py
    ```

4. Train models:
    ```python
    python batch_train.py
    ```

5. Evaluate metrics:
    ```python
    python batch_metrics.py
    ```

6. Export models:
    ```python
    python batch_export.py
    ```