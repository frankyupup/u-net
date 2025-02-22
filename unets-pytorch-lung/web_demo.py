#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import gradio as gr
import PIL.Image as Image
from model_data_inference.unet import Unet


model  = Unet(model_path="runs/unet_resnet50/unet_resnet50_model.pth", backbone="resnet50")

def predict_image(img):
    """Predicts objects in an image using a YOLO11 model with adjustable confidence and IOU thresholds."""
    #
    im = model.get_miou_png(img)

    return im


iface = gr.Interface(
    fn=predict_image,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        # gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold"),
        # gr.Slider(minimum=0, maximum=1, value=0.45, label="IoU threshold"),
    ],
    outputs=gr.Image(type="pil", label="Result"),
    title="基于残差unet的肺部图像分割系统",
    description="Upload images for inference.",
    # examples=[
    #     [ASSETS / "bus.jpg", 0.25, 0.45],
    #     [ASSETS / "zidane.jpg", 0.25, 0.45],
    # ],
)

if __name__ == "__main__":
    # iface.launch(share=True)
    # iface.launch(share=True)
    iface.launch()