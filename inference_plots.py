import os
import torch
import numpy as np 
from src.segment_anything import build_sam_vit_b, SamPredictor, sam_model_registry
from src.processor import Samprocessor
from src.lora import LoRA_sam
from PIL import Image
import matplotlib.pyplot as plt
import src.utils as utils
from PIL import Image, ImageDraw
import yaml
import json
from torchvision.transforms import ToTensor

"""
This file is used to plots the predictions of a model (either baseline or LoRA) on the train or test set. Most of it is hard coded so I would like to explain some parameters to change 
referencing by lines : 
line 22: change the rank of lora; line 98: Do inference on train (inference_train=True) else on test; line 101 and 111 is_baseline arguments in fuction: True to use baseline False to use LoRA model. 

CUDA_VISIBLE_DEVICES=? nohup poetry run python inference_plots.py > /home/lq/Projects_qin/surgical_semantic_seg/proposed_algorithm/SAM_LoRA/inf_plot1.log 2>&1 &

"""
# Open configuration file
with open("./config.yaml", "r") as ymlfile:
   config_file = yaml.load(ymlfile, Loader=yaml.Loader)


sam_checkpoint = config_file["SAM"]["CHECKPOINT"]
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = build_sam_vit_b(checkpoint=sam_checkpoint)

rank = 2
safetensors_path = f"/home/lq/Projects_qin/surgical_semantic_seg/experiments/SAM_LoRA/experiment_2/best_model_rank2_7_epoch_in100epochs.safetensors"
fig_path = f"/mnt/hdd2/task2/sam_lora/plots"
if not os.path.exists(fig_path):
    os.makedirs(fig_path)
is_baseline = False # Change to True if you want to use the baseline model

if is_baseline:
    model = build_sam_vit_b(checkpoint=sam_checkpoint).to(device)
else:
    sam = build_sam_vit_b(checkpoint=sam_checkpoint)
    sam_lora = LoRA_sam(sam, rank)
    sam_lora.load_lora_parameters(safetensors_path)
    model = sam_lora.sam.to(device)

def inference_model(image_path, filename, mask_path=None, bbox=None):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    if mask_path != None:
        mask = Image.open(mask_path)
        mask = mask.convert('1')
        ground_truth_mask =  np.array(mask)
        box = utils.get_bounding_box(ground_truth_mask)
    else:
        box = bbox

    predictor = SamPredictor(model)
    predictor.set_image(np.array(image))
    masks, iou_pred, low_res_iou = predictor.predict(
        box=np.array(bbox),
        multimask_output=False,
    )

    if mask_path == None:
        fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(15, 15))
        draw.rectangle(bbox, outline ="red")
        ax1.imshow(image)
        ax1.set_title(f"Original image + Bounding box: {filename}")

        ax2.imshow(masks[0])
        if is_baseline:
            ax2.set_title(f"Baseline SAM prediction: {filename}")
            plt.savefig(f"{fig_path}/{os.path.basename(mask_path)}_baseline.jpg")
        else:
            ax2.set_title(f"SAM LoRA rank {rank} prediction: {filename}")
            plt.savefig(f"{fig_path}/{os.path.basename(mask_path)}_rank{rank}.jpg")
        
    else:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(15, 15))
        draw.rectangle(bbox, outline ="red", width=10)
        ax1.imshow(image)
        ax1.set_title(f"Original image + Bounding box: {filename}")

        ax2.imshow(ground_truth_mask)
        ax2.set_title(f"Ground truth mask: {filename}")

        ax3.imshow(masks[0])
        if is_baseline:
            ax3.set_title(f"Baseline SAM prediction: {filename}")
            plt.savefig(f"{fig_path}/{os.path.basename(mask_path)}_baseline.jpg")
        else:
            ax3.set_title(f"SAM LoRA rank {rank} prediction: {filename}")
            plt.savefig(f"{fig_path}/{os.path.basename(mask_path)}_rank{rank}.jpg")
    plt.close(fig)


# Open annotation file
json_path = "/mnt/hdd2/task2/sam_lora/output_bbox_test1.json"
with open(json_path) as f:
    annotations = json.load(f)

test_set = annotations["test"]

base_path = config_file["DATASET"]["TEST_PATH"]

for image_name, info_list in test_set.items():
    # image_path = f"{base_path}/images/{image_name}"
    image_path = os.path.join(base_path, 'images', image_name)
    for dict_annot in info_list:
        inference_model(
            image_path, 
            filename=image_name, 
            mask_path=dict_annot["mask_path"], 
            bbox=dict_annot["bbox"])
        
        
