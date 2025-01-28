import os
import pandas as pd
from ultralytics import YOLO
import torch
import time
from tqdm import tqdm
from itertools import product
import gc


# Set environment variables
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Use GPU 2


# Paths
base_path = "/data/students/christian/machine_exercises/me6/runs"
validation_paths = {
    "detect": "/data/students/christian/machine_exercises/me6/data_det/images/val",
    "segment": "/data/students/christian/machine_exercises/me6/data_seg/images/val",
}


# Backends for each model
backends = ["PT", "TR"]  # PT for PyTorch, TR for TensorRT

# Function to load metrics from CSV
def get_metrics(task, train_number):
    model_val_path = f"/data/students/christian/machine_exercises/me6/runs/{task}/train{train_number}/weights/best.pt"
    model_val = YOLO(model_val_path)
    val_results = model_val.val()

    metrics = {}
    metrics["Precision (Boxes)"] = val_results.results_dict["metrics/precision(B)"]
    metrics["Recall (Boxes)"] = val_results.results_dict["metrics/recall(B)"]
    metrics["mAP50 (Boxes)"] = val_results.results_dict["metrics/mAP50(B)"]
    metrics["mAP50-95 (Boxes)"] = val_results.results_dict["metrics/mAP50-95(B)"]

    if task == "segment":
        metrics["Precision (Masks)"] = val_results.results_dict["metrics/precision(M)"]
        metrics["Recall (Masks)"] = val_results.results_dict["metrics/recall(M)"]
        metrics["mAP50 (Masks)"] = val_results.results_dict["metrics/mAP50(M)"]
        metrics["mAP50-95 (Masks)"] = val_results.results_dict["metrics/mAP50-95(M)"]
    else:
        metrics["Precision (Masks)"] = ""
        metrics["Recall (Masks)"] = ""
        metrics["mAP50 (Masks)"] = ""
        metrics["mAP50-95 (Masks)"] = ""

    del model_val
    torch.cuda.empty_cache()
    gc.collect()

    return metrics

# Function to calculate inference speed using time module
def get_inference_speed(task, model_path, quantized=False):
    val_path = validation_paths[task]
    model = YOLO(model_path).to('cuda') if not model_path.endswith(".engine") else YOLO(model_path)
    times = []

    for img in os.listdir(val_path):
        if img.endswith((".jpg", ".png")):
            img_path = os.path.join(val_path, img)
            start_time = time.time()
            _ = model.predict(source=img_path, device="cuda", half=quantized)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to ms

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return sum(times) / len(times)


# Function to get model info
def get_model_info(model_path):
    if model_path.endswith(".engine"):
        return {
            "No. of Params": "",
            "FLOPS": "",
            "Layers": "",
        }
    model = YOLO(model_path).to('cuda')
    info = model.info()

    layers = info[0]
    params_millions = round(info[1] / 1e6, 2)
    gflops = round(info[3], 2)

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return {
        "No. of Params": params_millions,
        "FLOPS": gflops,
        "Layers": layers,
    }

# Main function
def generate_metrics_table():
    rows = []
    progress = tqdm(product(model_combinations, backends))

    for combo, backend in progress:
        model = combo["Model"]
        task = combo["Task"]
        train_number = combo["Train Number"]

        progress.set_description(f"Model: {model}{backend}")
        progress.set_postfix(task=task, backend=backend)

        # Path to model weights
        format_extension = "pt" if backend == "PT" else "engine"
        model_path = f"{base_path}/{task}/train{train_number}/weights/best.{format_extension}"

        # Extract metrics and model information
        try:
            metrics = get_metrics(task, train_number)
            inference_speed = get_inference_speed(task, model_path)
            inference_speed_quant = get_inference_speed(task, model_path, quantized=True)  # Quantized speed
            model_info = get_model_info(model_path)

            row = {
                "Models": f"{model}{backend}",
                **model_info,
                **metrics,
                "Inference Speed (ms)": round(inference_speed, 2),
                "Inference Speed (ms) [Quant]": round(inference_speed_quant, 2),  # Add quantized column
            }
            rows.append(row)
        except Exception as e:
            print(f"Error processing {model}{backend}: {e}")

    # Convert rows to DataFrame
    return pd.DataFrame(rows)








# Hardcoded model configurations and train numbers
model_combinations = [
    # {"Model": "D1A1", "Task": "detect", "Train Number": "2"},
    # {"Model": "D1A2", "Task": "detect", "Train Number": "3"},
    # {"Model": "S1A1", "Task": "segment", "Train Number": "2"},
    # {"Model": "S1A2", "Task": "segment", "Train Number": "3"},
    {"Model": "D2A2", "Task": "detect", "Train Number": "4"},
    {"Model": "S2A2", "Task": "segment", "Train Number": "4"},
]


# Generate and save table
metrics_table = generate_metrics_table()
metrics_table.to_csv("final_metrics_table.csv", index=False)
print(metrics_table)
