# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Kristian Dalsbø Hindberg
"""Color-based cleanliness estimation for Colon Capsule Endoscopy recordings.

Estimates the cleanliness of each image in a CCE recording using a color-channel
method developed at SDU. Images are classified as clean or dirty based on the
ratio of clean to dirty pixels, where dirty pixels are identified by their color
characteristics.

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
from PIL import Image
from glob import glob
import pandas as pd
import numpy as np
from os.path import join
from pathlib import Path

from Position_aux_funcs import classify_frame_cleanliness, PID_raw_info



def cleanliness_inference_folder_colorbased(fd_inn, clean_threshold=0.9, verbose=False):
    """Estimate cleanliness based on color channels for all images in a given folder.

    Parameters
    ----------
    fd_inn : str
        Path to a stream/ or stream2/ subfolder of a patient folder (raw format).
    clean_threshold : float, optional
        Threshold for classifying a pixel as clean. Default is 0.9.
    verbose : bool, optional
        Print progress to the terminal. Default is False.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns: n_clean_pixels, n_dirty_pixels, imlist, filename.
    """
    update_iterations = 2500

    if os.path.isdir(fd_inn):
        # Get parent patient folder and determine which head this stream folder belongs to
        parent_folder = Path(fd_inn).parent
        Info_raw_PID = PID_raw_info(parent_folder)
        head_curr = 2 if re.search("stream2", fd_inn) else 1
        imlist = list(range(getattr(Info_raw_PID, f"n_frames_{head_curr}")))
    else:
        raise ValueError("Input must be a path to a stream/ or stream2/ folder")

    # Run cleanliness classifier for all images
    clean_all = []
    dirty_all = []
    for immy, im_curr in enumerate(imlist):

        if verbose and (immy % update_iterations) == 0:
            print(f"Doing image {immy:5} of {len(imlist)}")

        image = Info_raw_PID.get_image(immy, head_curr)

        # Convert to numpy array and swap to BGR channel order assumed by classify_frame_cleanliness
        frame = np.array(image)
        frame = frame[:, :, ::-1]

        dirty_pixels, clean_pixels = classify_frame_cleanliness(frame, clean_threshold)
        clean_all.append(clean_pixels)
        dirty_all.append(dirty_pixels)

    return pd.DataFrame({"n_clean_pixels": clean_all, "n_dirty_pixels": dirty_all,
                          "imlist": imlist, "filename": imlist})




