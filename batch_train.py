from ultralytics import YOLO
import os
import torch


# no nvlink
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
# use a specific GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "2"



model_det_list = [
                    # "yolo11n.pt",
                    "yolo11x.pt"
                  ]
model_seg_list = [
                    # "yolo11n-seg.pt",
                    "yolo11x-seg.pt"
                  ]
                 


epochs = 200
batch = 32


for model in model_det_list:
    model_det = YOLO(model).to('cuda')


    # Train the model
    train_results = model_det.train(
        data="/data/students/christian/machine_exercises/me6/data_det/data_det.yaml",  # path to dataset YAML
        epochs=epochs,  # number of training epochs
        imgsz=640,  # training image size
        patience = 40,
        verbose = True,
        seed = 42,
        plots = True,
        batch = batch, # try 16, 32, 64
        device="cuda",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
        # device=[0, 1], for multi-gpu training or device=["cuda:0", "cuda:1"]

        # # Augmentation parameters
        # hsv_h=0.015,        # Default value to adjust hue
        # hsv_s=0.7,          # Default value to adjust saturation
        # hsv_v=0.4,          # Default value to adjust brightness
        # degrees=5.0,        # Small rotation to handle slight orientation changes
        # translate=0.1,      # Default translation to shift objects slightly
        # scale=0.5,          # Default scaling to simulate distance variations
        # shear=0.0,          # No shearing as it's less relevant for product images
        # perspective=0.0,    # No perspective transformation
        # flipud=0.0,         # No vertical flipping; products are rarely upside down
        # fliplr=0.5,         # Horizontal flipping to augment data diversity
        # mosaic=1.0,         # Use mosaic for complex scene understanding
        # mixup=0.1,          # Small mixup to introduce label noise and variability
        # copy_paste=0.1      # Use copy-paste to increase object instances
    )


    # Delete the model and free memory
    del model_det
    torch.cuda.empty_cache()

for model in model_det_list:
    model_det = YOLO(model).to('cuda')


    # Train the model
    train_results = model_det.train(
        data="/data/students/christian/machine_exercises/me6/data_det/data_det.yaml",  # path to dataset YAML
        epochs=epochs,  # number of training epochs
        imgsz=640,  # training image size
        patience = 40,
        verbose = True,
        seed = 42,
        plots = True,
        batch = batch, # try 16, 32, 64
        device="cuda",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
        # device=[0, 1], for multi-gpu training or device=["cuda:0", "cuda:1"]

        # Augmentation parameters
        hsv_h=0.015,        # Default value to adjust hue
        hsv_s=0.7,          # Default value to adjust saturation
        hsv_v=0.4,          # Default value to adjust brightness
        degrees=5.0,        # Small rotation to handle slight orientation changes
        translate=0.1,      # Default translation to shift objects slightly
        scale=0.5,          # Default scaling to simulate distance variations
        shear=0.0,          # No shearing as it's less relevant for product images
        perspective=0.0,    # No perspective transformation
        flipud=0.0,         # No vertical flipping; products are rarely upside down
        fliplr=0.5,         # Horizontal flipping to augment data diversity
        mosaic=1.0,         # Use mosaic for complex scene understanding
        mixup=0.1,          # Small mixup to introduce label noise and variability
        copy_paste=0.1      # Use copy-paste to increase object instances
    )

    # Delete the model and free memory
    del model_det
    torch.cuda.empty_cache()







for model in model_seg_list:
    model_seg = YOLO(model).to('cuda')

    # Train the model
    train_results = model_seg.train(
        data="/data/students/christian/machine_exercises/me6/data_seg/data_seg.yaml",  # path to dataset YAML
        epochs=epochs,  # number of training epochs
        imgsz=640,  # training image size
        patience = 40,
        verbose = True,
        seed = 42,
        plots = True,
        batch = batch, # try 16, 32, 64
        device="cuda",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
        # device=[0, 1], for multi-gpu training or device=["cuda:0", "

        # # Augmentation parameters
        # hsv_h=0.015,        # Default value to adjust hue
        # hsv_s=0.7,          # Default value to adjust saturation
        # hsv_v=0.4,          # Default value to adjust brightness
        # degrees=5.0,        # Small rotation to handle slight orientation changes
        # translate=0.1,      # Default translation to shift objects slightly
        # scale=0.5,          # Default scaling to simulate distance variations
        # shear=0.0,          # No shearing as it's less relevant for product images
        # perspective=0.0,    # No perspective transformation
        # flipud=0.0,         # No vertical flipping; products are rarely upside down
        # fliplr=0.5,         # Horizontal flipping to augment data diversity
        # mosaic=1.0,         # Use mosaic for complex scene understanding
        # mixup=0.1,          # Small mixup to introduce label noise and variability
        # copy_paste=0.1      # Use copy-paste to increase object instances
    )

    # Delete the model and free memory
    del model_seg
    torch.cuda.empty_cache()




for model in model_seg_list:
    model_seg = YOLO(model).to('cuda')

    # Train the model
    train_results = model_seg.train(
        data="/data/students/christian/machine_exercises/me6/data_seg/data_seg.yaml",  # path to dataset YAML
        epochs=epochs,  # number of training epochs
        imgsz=640,  # training image size
        patience = 40,
        verbose = True,
        seed = 42,
        plots = True,
        batch = batch, # try 16, 32, 64
        device="cuda",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
        # device=[0, 1], for multi-gpu training or device=["cuda:0", "

        # Augmentation parameters
        hsv_h=0.015,        # Default value to adjust hue
        hsv_s=0.7,          # Default value to adjust saturation
        hsv_v=0.4,          # Default value to adjust brightness
        degrees=5.0,        # Small rotation to handle slight orientation changes
        translate=0.1,      # Default translation to shift objects slightly
        scale=0.5,          # Default scaling to simulate distance variations
        shear=0.0,          # No shearing as it's less relevant for product images
        perspective=0.0,    # No perspective transformation
        flipud=0.0,         # No vertical flipping; products are rarely upside down
        fliplr=0.5,         # Horizontal flipping to augment data diversity
        mosaic=1.0,         # Use mosaic for complex scene understanding
        mixup=0.1,          # Small mixup to introduce label noise and variability
        copy_paste=0.1      # Use copy-paste to increase object instances
    )



    # Delete the model and free memory
    del model_seg
    torch.cuda.empty_cache()