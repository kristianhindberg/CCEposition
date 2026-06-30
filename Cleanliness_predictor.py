# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Kristian Dalsbø Hindberg
"""Neural network-based cleanliness classifier for Colon Capsule Endoscopy recordings.

Classifies each image in a CCE recording into one of four Leighton Rex cleanliness
categories using a trained ResNet18 model. Results are used by the position
inference pipeline to optionally exclude dirty images from segment classification.

Used by CCE_position.py as part of the full position and cleanliness
inference pipeline.

Author
------
    Kristian Dalsbø Hindberg, UiT — The Arctic University of Norway, 2025—2026

GitHub
------
    https://github.com/kristianhindberg/CCEposition
"""

import os
import re
from os.path import join
from pathlib import Path
from PIL import Image
from glob import glob
import pandas as pd
import timm
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from torchvision.transforms import ToTensor

from Position_aux_funcs import Mean_and_stdev_of_pillCameraImages, PID_raw_info

# Path to cleanliness classifier model — assumed to be in the same directory as this script
fd_cleanliness_model = os.path.dirname(os.path.abspath(__file__))
fn_cleanliness_model = "CCE_model_cleanliness.pth"

# Neural network architecture — must match the architecture used during training
model_name = "resnet18"

# Select compute device
device = "cuda" if torch.cuda.is_available() else "cpu"


class ResNetCustom(nn.Module):
    """Custom ResNet classifier with adaptive pooling and optional dropout.

    Parameters
    ----------
    base_model_name : str
        Name of the timm base model (e.g. 'resnet18').
    num_classes_category : int
        Number of output classes.
    dropout_rate : float, optional
        Dropout probability applied before the final FC layer. Default is 0.0.
    """

    def __init__(self, base_model_name, num_classes_category, dropout_rate=0.0):
        super(ResNetCustom, self).__init__()
        self.base_model = timm.create_model(base_model_name, pretrained=False, num_classes=0)
        num_features = self.base_model.num_features
        self.pool    = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.fc      = nn.Linear(num_features, num_classes_category)

    def forward(self, x):
        """Forward pass."""
        x = self.base_model.forward_features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


def load_resnet_model(model_name=model_name, device=device):
    """Load the cleanliness ResNet model from a saved state dict.

    Parameters
    ----------
    model_name : str
        timm model name — must match the architecture the weights were trained with.
    device : str
        Compute device — 'cuda' or 'cpu'.

    Returns
    -------
    ResNetCustom
        Model in evaluation mode on the specified device.
    """
    load_model_path = join(fd_cleanliness_model, fn_cleanliness_model)
    model = ResNetCustom(base_model_name=model_name, num_classes_category=4)
    model.to(device)
    model.eval()
    checkpoint = torch.load(load_model_path, map_location=torch.device(device), weights_only=True)
    model.load_state_dict(checkpoint)
    return model


def cleanliness_inference_folder(fd_inn, verbose=True):
    """Run cleanliness classification on all images in a stream folder.

    Parameters
    ----------
    fd_inn : str
        Path to a stream/ or stream2/ subfolder of a patient folder (raw format).
    verbose : bool, optional
        Print progress to the terminal. Default is True.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: class_out, probs_out, probs_all, imlist, filename.
    """
    update_iterations = 2500

    if not os.path.isdir(fd_inn):
        raise ValueError("Input must be a path to a stream/ or stream2/ folder")

    parent_folder = Path(fd_inn).parent
    Info_raw_PID = PID_raw_info(parent_folder)
    head_curr = 2 if re.search("stream2", fd_inn) else 1
    imlist = list(range(getattr(Info_raw_PID, f"n_frames_{head_curr}")))

    # Load cleanliness classifier
    model = load_resnet_model(model_name=model_name, device=device)

    mean_per_channel, stdv_per_channel = Mean_and_stdev_of_pillCameraImages()
    norm_function   = transforms.Normalize(mean=mean_per_channel, std=stdv_per_channel)
    resize_function = transforms.Resize(224, antialias=True)

    class_out = []
    probs_out = []
    probs_all = []

    with torch.no_grad():
        for immy, im_curr in enumerate(imlist):

            if verbose and (immy % update_iterations) == 0:
                print(f"Doing image {immy:5} of {len(imlist)}")

            image = Info_raw_PID.get_image(immy, head_curr)
            image = torch.unsqueeze(resize_function(norm_function(ToTensor()(image))), dim=0)

            outputs = model.forward(image.to(device=device)).cpu()
            maxprob, predicted = outputs.max(dim=1)
            class_out.append(predicted.item())
            probs_out.append(np.exp(maxprob.item()) / np.exp(outputs).sum().numpy())
            probs_all.append([np.exp(outy.numpy()) / np.exp(outputs).sum().numpy() for outy in outputs])

    return pd.DataFrame({"class_out": class_out, "probs_out": probs_out, "probs_all": probs_all,
                          "imlist": imlist, "filename": imlist})
