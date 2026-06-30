# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Kristian Dalsbø Hindberg
"""Wall-facing image classifier for Colon Capsule Endoscopy recordings.

Classifies each image in a CCE recording as wall-facing or non-wall-facing
using a trained ResNet50 model. Wall-facing images are uninformative for
position and cleanliness estimation and can be excluded from those pipelines.

Used by CCE_position.py as part of the full position and cleanliness
inference pipeline.

Author
------
    Kristian Dalsbø Hindberg, UiT — The Arctic University of Norway, 2024—2026

GitHub
------
    https://github.com/kristianhindberg/CCEposition
"""

import os
import re
from os.path import join
import sys
from PIL import Image
from glob import glob
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
from torchvision.transforms import ToTensor
from pathlib import Path

from Position_aux_funcs import Mean_and_stdev_of_pillCameraImages, PID_raw_info

# Path to wall classifier model — assumed to be in the same directory as this script
wall_path = join(os.path.dirname(os.path.abspath(__file__)), "CCE_model_wall.pth")

# Select compute device
device = "cuda" if torch.cuda.is_available() else "cpu"


def Load_model(load_model_path, n_classes, device):
    """Load a ResNet50 wall classifier from a saved state dict.

    Parameters
    ----------
    load_model_path : str
        Full path to the .pth model file.
    n_classes : int
        Number of output classes (2 for wall / non-wall).
    device : str
        Compute device — 'cuda' or 'cpu'.

    Returns
    -------
    torch.nn.Module
        Model in evaluation mode on the specified device.
    """
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, n_classes)
    checkpoint = torch.load(load_model_path, map_location=torch.device(device), weights_only=True)
    model.load_state_dict(checkpoint)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.to(device)
    model.eval()
    return model


def wall_inference_folder(fd_inn, verbose=True):
    """Run wall-facing classification on all images in a stream folder.

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

    # Load wall classifier
    wall_model = Load_model(load_model_path=wall_path, n_classes=2, device=device)

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

            outputs = wall_model.forward(image.to(device=device)).cpu()
            maxprob, predicted = outputs.max(dim=1)
            class_out.append(predicted.item())
            probs_out.append(np.exp(maxprob.item()) / np.exp(outputs).sum().numpy())
            probs_all.append([np.exp(outy.numpy()) / np.exp(outputs).sum().numpy() for outy in outputs])

    return pd.DataFrame({"class_out": class_out, "probs_out": probs_out, "probs_all": probs_all,
                          "imlist": imlist, "filename": imlist})


if __name__ == "__main__":
    if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]):
        fd_inn = sys.argv[1]
    else:
        fd_inn = os.getcwd()
        print(f"No valid folder given as input, using current working directory:\n  {fd_inn}")

    wall_inference_folder(fd_inn)
