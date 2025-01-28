from ultralytics import YOLO
import os
import torch
import gc


# no nvlink
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
# use a specific GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "2"




task = "detect"
train_numbers = [
                # "2", "3", 
                #  "4", 
                 ]

for train_number in train_numbers:
    ft_model_path = f"/data/students/christian/machine_exercises/me6/runs/{task}/train{train_number}/weights/best.pt"
    ft_model = YOLO(ft_model_path).to('cuda')

    # ft_model.export(format="onnx", device="cuda")
    # time.sleep(60)
    ft_model.export(format="TensorRT", device="cuda")

    del ft_model
    torch.cuda.empty_cache()
    gc.collect()

task = "segment"
# train_numbers = ["2", "3", "4", "5"]

for train_number in train_numbers:
    ft_model_path = f"/data/students/christian/machine_exercises/me6/runs/{task}/train{train_number}/weights/best.pt"
    ft_model = YOLO(ft_model_path).to('cuda')

    # ft_model.export(format="onnx", device="cuda")
    # time.sleep(60)
    ft_model.export(format="TensorRT", device="cuda")

    del ft_model
    torch.cuda.empty_cache()
    gc.collect()