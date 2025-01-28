from ultralytics import YOLO
import gradio as gr
from gradio_toggle import Toggle
import cv2
import numpy as np
import time
import os
import argparse
# import onnxruntime as ort # For ONNX
import tensorrt as trt # For TensorRT

# Set environment variables
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Use GPU 2




# Argument parsing
parser = argparse.ArgumentParser(description="Run Gradio demo with specified resolution.")
parser.add_argument(
    'resolution', 
    choices=['small', 'large'], 
    nargs='?',  # Makes the argument optional
    default='large',  # Default value if no argument is provided
    help="Choose the resolution: small or large (default: large)"
)
args = parser.parse_args()

# Set resolution
if args.resolution == 'small':
    desired_size = (320, 240)  # Small resolution
elif args.resolution == 'large':
    desired_size = None  # Large resolution





def initialize_models():
    """Initialize models at startup"""

    
    print("Loading detection model...")
    det_model = YOLO("/data/students/christian/machine_exercises/me6/runs/detect/train4/weights/best.engine")
 
    print("Loading segmentation model...")
    seg_model = YOLO("/data/students/christian/machine_exercises/me6/runs/segment/train4/weights/best.engine")
    # seg_model = YOLO("/data/students/christian/machine_exercises/me6/runs/train4_backup/weights/best.engine")


    # Determine input size for warmup
    input_height = desired_size[1] if desired_size else 480  # Default height if large resolution
    input_width = desired_size[0] if desired_size else 640   # Default width if large resolution

    # Create a dummy input image
    warmup_input = np.zeros((input_height, input_width, 3), dtype=np.uint8)

    # Warm up the detection model
    print("Warming up detection model...")
    for _ in range(20):
        det_model.predict(
            source=warmup_input,
            device=0,
            # half=True
        )

    # Warm up the segmentation model
    print("Warming up segmentation model...")
    for _ in range(20):
        seg_model.predict(
            source=warmup_input,
            device=0,
            # half=True
        )

    return det_model, seg_model

# Initialize models globally before Gradio interface
model_det, model_seg = initialize_models()


global prev_frame_timestamp  
prev_frame_timestamp = 0


def process_frame(frame, is_segmentation, enable_preprocessing):
    global prev_frame_timestamp
    start_time = time.time()
    
    if enable_preprocessing:
        # Increase brightness and contrast
        alpha = 1.3  # Contrast control (1.0-3.0)
        beta = 10    # Brightness control (0-100)
        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
        

        
        # Noise reduction
        kernel_size = (5, 5)
        frame = cv2.GaussianBlur(frame, kernel_size, 0)
        

        # seems like this is helping with occlusion especially with fingers
        # Apply CLAHE for better contrast
        yuv = cv2.cvtColor(frame, cv2.COLOR_RGB2YUV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        yuv[:,:,0] = clahe.apply(yuv[:,:,0])
        frame = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
        

        # Image Sharpening
        sharpening_kernel = np.array([[0, -1, 0],
                                    [-1, 5,-1],
                                    [0, -1, 0]])
        frame = cv2.filter2D(frame, -1, sharpening_kernel)
        
        # Morphological Operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
        






    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)







    # Reduce the resolution of the frame
    if desired_size:
        frame = cv2.resize(frame, desired_size, interpolation=cv2.INTER_AREA)

    # Use detection when toggle is off, segmentation when on
    model = model_seg if is_segmentation else model_det
    
    results = model.predict(
        source=frame,
        conf=0.15,
        iou=0.35,
        line_width=2,
        show_labels=True,
        show_conf=True,
        device=0,
        # half=True, #True
    )
    
    annotated_img = results[0].plot()

    inference_time = results[0].speed["preprocess"] + results[0].speed["inference"] + results[0].speed["postprocess"]

    
    inf_text = f"Inference: {inference_time:.1f}ms"

    # Draw FPS and inference time counters
    font = cv2.FONT_HERSHEY_SIMPLEX

    font_scale = 0.4 if desired_size else 1
    font_thickness = 1 if desired_size else 2
    x_scale = 1 if desired_size else 2
    y_scale = 1 if desired_size else 2


    display_time = time.time()  # Timestamp before displaying the frame
    latency = (display_time - start_time) * 1000  # Latency in milliseconds
    
    # latency here is the inference time + image annotation time
    # the network latency is not included here
    latency_text = f"Latency: {latency:.1f}ms"

    current_time = time.time()
    if prev_frame_timestamp != 0:
        fps = 1.0 / (current_time - prev_frame_timestamp)
    else:
        fps = 0.0
    prev_frame_timestamp = current_time

    fps_text = f"FPS: {fps:.1f}"

    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

    cv2.putText(annotated_img, fps_text, (annotated_img.shape[1]-(90*x_scale), 12*y_scale), 
                font, font_scale, (0, 255, 0), font_thickness)

    
    print(f"{fps_text}, {inf_text}, {latency_text}")

    return annotated_img


css=""".my-group {max-width: 800px !important; margin: auto !important;}
      .toggle-row {display: flex !important; justify-content: center !important; margin-bottom: 20px !important;}
      .container {display: flex !important; flex-direction: column !important; align-items: center !important;}"""

with gr.Blocks(css=css) as demo:
    with gr.Group(elem_classes="my-group"):
        with gr.Row(elem_classes="toggle-row"):
            mode_toggle = Toggle(
                label="Switch to Segmentation",
                value=True,
                interactive=True,
                color="green",
                info="Switch between object detection and segmentation (Default: Segmentation)"
            )
            preprocessing_toggle = Toggle(
                label="Enable Hyperpreprocessing",
                value=False,
                interactive=True,
                color="purple",
                info="Enable or disable hyperpreprocessing steps (Default: Disabled)"
            )
        input_img = gr.Image(sources=['webcam'], 
                            type="numpy", streaming=True, label="Webcam",
                            # mirror_webcam=False,
                            )
        input_img.stream(
            fn=process_frame,
            inputs=[input_img, mode_toggle, preprocessing_toggle],
            outputs=input_img,
            time_limit=10, 
            concurrency_limit=10,


            # stream_every=0.6, # BEST SO FAR IN CONDO
                                # # use this for now, in order to still have a decent FPS
                                # 1.6 fps
            stream_every=0.075, # BEST SO FAR IN EDUROAM, 0.033 still laggy
            

            
        )

# demo.launch(share=True)

# demo.queue()
demo.launch()