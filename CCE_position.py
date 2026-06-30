# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Kristian Dalsbø Hindberg
"""Position and cleanliness inference for Colon Capsule Endoscopy recordings.

Classifies each image in a dual-headed CCE recording (Colon2 pill camera, raw
.gfd/.gvf format) into one of four intestinal segments: small intestine (SB),
ascending colon (AC), transverse colon (TC), or descending colon (DC).

Cleanliness and wall-facing estimates are computed first and can optionally be
used to exclude uninformative images from the position classifier. A running-mode
smoothing step with a fixed image-count window is then applied to the per-image
classifier output to produce a robust segment estimate over time.

A fixed image-count window is used rather than a fixed time-window. The pill
camera dynamically adjusts its frame rate, so a fixed image-count window
naturally covers more time in low-activity regions and less in high-activity
regions.

Transition probabilities
------------------------
An optional Markov-style transition matrix can be applied during smoothing so
that physiologically unlikely segment transitions (e.g. descending colon back
to small intestine) are down-weighted. See get_transition_probability_matrix()
for the parameter values. A small non-zero back-transition probability is
retained to avoid the classifier getting permanently stuck in the colon when a
few small-intestine images are misclassified.

Output
------
Results are written as pickle files to the patient folder (or to a separate
output folder if fd_out_main is set). The viewer CCE_viewer.py reads these
files to display the position and cleanliness plots.

Usage
-----
    python CCE_position.py [ARGUMENT]

    ARGUMENT is optional and can be one of:

    - A path to a single patient folder (containing stream/ and stream2/
      subdirectories): runs inference on that patient only.
    - A path to a parent directory: runs inference on all valid patient
      folders found inside it.
    - No argument: scans the current working directory for patient folders.

    All model files must be present in the same directory as this script.

Requirements
------------
    torch, torchvision, matplotlib, timm
    See environment.yml for full package requirements.

Future improvements
-------------------
- Consider using both heads jointly around a given time point to assign a
  segment, rather than treating them independently. The two heads almost
  always face the same segment, including during transitions.
- Consider using a smaller window size until the pill has entered the colon,
  since a large window (e.g. 201) tends to produce a delayed transition.

Author
------
    Kristian Dalsbø Hindberg, UiT — The Arctic University of Norway, 2023—2026

GitHub
------
    https://github.com/kristianhindberg/CCEposition
"""
from os import getcwd, makedirs, listdir
from os.path import exists, isdir, basename, join, normpath
import sys
import re
from shutil import copytree, copy2
import pickle
import numpy as np
import time
from PIL import Image
from glob import glob

import torch
from torch import autocast
from torchvision import transforms, models

import matplotlib.pyplot as plt
from matplotlib import rcParams
# Set plot font size
rcParams.update({'font.size': 14})

# Import position aux functions
from Position_aux_funcs import Mean_and_stdev_of_pillCameraImages, str_to_ms, classify_frame_cleanliness
# and classes
from Position_aux_funcs import PID_raw_info

# Import classifier functions
from Cleanliness_predictor            import cleanliness_inference_folder
from Cleanliness_predictor_colorbased import cleanliness_inference_folder_colorbased
from Wall_predictor                   import wall_inference_folder

# Set parameters that influence how the output of the per-image classifier
# is made into position predictions
DO_skip_wall_images     = True
DO_skip_dirty_images    = True
DO_use_transition_probs = True
DO_include_uncertainty  = True
DO_include_reverse      = False
uncertainty_method      = 'count'  # How to compute uncertainty when DO_include_uncertainty=True.
                                    # Options: 'count'     - fraction of valid images voting for winning class
                                    #          'mean_prob' - mean probability of winning-class images
DO_redo_calc_even_if_done = False
DO_not_copy_video_data_even_if_making_new_dir = False  # If True, video data is never copied even when a separate output folder is used.

verbose                 = True
window_size             = 201 # window size for the running mode

DO_detect_landmarks = False # Whether to detect landmarks (ICV, appendix, hepatic/splenic flexures)
DO_save_landmark_images = True # Whether to save landmark images to disk
verbose_landmark_findings = False

Remove_wall_p_value_threshold = 0.95 # This should be [0.5-1>
clean_cat_lowest_accepted = 2
cleanliness_threshold   = 1 # wrt #dirty_pixels / #clean_pixels ==> 1 ==> 50% dirty
clean_threshold = 0.9

# Path to load position classifier from - assumed that it exists in the folder the script is run from
fd_main = getcwd()
position_model_used = 'CCE_model_position.pth'
position_model_path = f"{fd_main}/{position_model_used}"

# Define intestine segment predictor model
n_classes = 4

fd_out_main = None

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define paths to load landmark classifier models.
# All model files are expected to be in the same folder as this script (fd_main).
ICV_model_path              = f"{fd_main}/CCE_model_ICV.pth"
appendix_model_path         = f"{fd_main}/CCE_model_appendix.pth"
hepatic_flexure_model_path  = f"{fd_main}/CCE_model_hepatic_flexure.pth"
# Note: the splenic flexure model is included for completeness but performs poorly due to
# noisy training data. It should be replaced with a better-trained model before relying on it.
splenic_flexure_model_path  = f"{fd_main}/CCE_model_splenic_flexure.pth"

#########################
# NO MORE INPUTS NEEDED #
#########################

fn_ext_out = ""
if DO_skip_wall_images:
    fn_ext_out = fn_ext_out + "_wallRemoved"
if DO_skip_dirty_images:
    fn_ext_out = fn_ext_out + "_DirtySkip"
if DO_use_transition_probs:
    fn_ext_plot = fn_ext_out + f'_window{window_size}' + "_logic"
else:
    fn_ext_plot = fn_ext_out + f'_window{window_size}'
if DO_include_reverse:
    fn_ext_plot = fn_ext_out + "_reverse"

#############
# FUNCTIONS #
#############

def idxx(arr, target):
    """Get index of array 'arr' that is closest in value to 'target' value.
    
    If this is more than one index, only the lowest index number will be returned.
    This can be changed to return all entries with minimal distance to target by returning this
    return np.where((np.abs(arr - target)).min() == (np.abs(arr - target)))
    """
    return (np.abs(np.asarray(arr) - target)).argmin()

def load_resnet50_model(load_model_path, n_classes, device):
    """Load resnet50 model."""
    # Load only the saved state dict into a new instance of the resnet50 model.
    model = models.resnet50(weights=None) # Make new model object
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, n_classes)
    
    # Insert trained and save parameters
    checkpoint = torch.load(load_model_path, map_location=torch.device(device), weights_only=True)
    model.load_state_dict(checkpoint)
    
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Send it to cuda if available
    model.to(device)

    # Set model to evaluation mode
    model.eval()

    return model

def PID_position_dict_init(PID, fd_PID, n_classes, position_model_used, fd_out = None):
    """Initialize a dictionary for the current PID."""
    # Use global/static variables set at start of code, which do not change.
    PID_position = {}
    PID_position['PID'] = PID
    PID_position['n_classes'] = n_classes
    PID_position['classes'] = {0:"Ascending colon", 1:"Transverse colon", 2:"Descending colon", 3:"Small intestine"}
    PID_position['position_model_used'] = position_model_used
    PID_position['DO_skip_wall_images'] = DO_skip_wall_images
    PID_position['Remove_wall_p_value_threshold'] = Remove_wall_p_value_threshold
    PID_position['DO_skip_dirty_images'] = DO_skip_dirty_images
    PID_position['cleanliness_threshold'] = cleanliness_threshold
    PID_position['DO_use_transition_probs'] = DO_use_transition_probs
    if fd_out is None:
        PID_position['fd_out'] = fd_PID
    else:
        PID_position['fd_out'] = fd_out
        
    # Load raw binary data
    PID_position['Raw'] = PID_raw_info(fd_PID=fd_PID, fd_out=PID_position['fd_out'])

    return PID_position

def cleanup_raw_object(PID_position):
    """Extract needed data from Raw and delete the heavy object before pickling."""
    if 'Raw' not in PID_position:
        return PID_position
    
    raw = PID_position['Raw']
    
    # Extract useful metadata
    PID_position['n_frames_1'] = getattr(raw, "n_frames_1", None)
    PID_position['n_frames_2'] = getattr(raw, "n_frames_2", None)
    PID_position['time_table_1'] = getattr(raw, "time_table_1")[:, 1].copy()
    PID_position['time_table_2'] = getattr(raw, "time_table_2")[:, 1].copy()
    
    del PID_position['Raw']
    return PID_position

def classify_position(model_pos_class, fd_PID, PID_position, fn_results, fn_PID_curr = None, PDF_report_exists = False):
    """Run position classifier (or load it if already done) for given patient."""
    # Load mean and stdevs for the different color channels
    mean_per_channel, stdv_per_channel =  Mean_and_stdev_of_pillCameraImages()

    # Define preprocessing steps for each image
    norm_function   = transforms.Normalize(mean=mean_per_channel, std=stdv_per_channel )
    resize_function = transforms.Resize(224, antialias=True)

    # Extract PID number
    PID = PID_position['PID']
    # and output folder
    fd_out = PID_position['fd_out']

    # Load results if already done
    if exists(fn_results):
        with open(fn_results, 'rb') as f:
            PID_position_loaded = pickle.load(f)

        # Only return results if the modeled used to produce the loaded results
        # is the same as the current specified model
        if PID_position_loaded['position_model_used'] == PID_position['position_model_used']:
            if PID_position_loaded['DO_skip_wall_images'] == PID_position['DO_skip_wall_images']:
                if PID_position_loaded['Remove_wall_p_value_threshold'] == PID_position['Remove_wall_p_value_threshold']:
                    if PID_position_loaded['DO_skip_dirty_images'] == PID_position['DO_skip_dirty_images']:
                        if PID_position_loaded['cleanliness_threshold'] == PID_position['cleanliness_threshold']:
                            print(f'Done already - PID {PID}')
                            return PID_position_loaded

    # Load precalculated or calculate wall probabilities and cleanliness estimates
    if DO_skip_wall_images or DO_skip_dirty_images:
        if DO_skip_dirty_images is True  and DO_skip_wall_images is True:
            fn_PID_wall = f'{fd_out}/PID_{PID}_WallProb_and_cleanliness.pickle'
        elif DO_skip_dirty_images is False and DO_skip_wall_images is True:
            fn_PID_wall = f'{fd_out}/PID_{PID}_WallProb_and_cleanliness_wallOnly.pickle'
        elif DO_skip_dirty_images is True  and DO_skip_wall_images is False:
            fn_PID_wall = f'{fd_out}/PID_{PID}_WallProb_and_cleanliness_cleanOnly.pickle'

        if exists(fn_PID_wall):
            with open(fn_PID_wall, 'rb') as f:
                WallProb_and_clean_curr = pickle.load(f)
        else:

            # Run wall and cleanliness classifier on both heads
            WallProb_and_clean_curr = dict()
            WallProb_and_clean_curr['PID'] = PID
            WallProb_and_clean_curr['fd_PID'] = fd_PID
            WallProb_and_clean_curr['clean_threshold'] = clean_threshold

            for head in [1, 2]:
                if head == 1:
                    data_dir_temp = f'{fd_PID}/stream/'
                elif head == 2:
                    data_dir_temp = f'{fd_PID}/stream2/'
                # Cleanliness
                if DO_skip_dirty_images:
                    if verbose:
                        print('Running cleanliness estimation', end="")
                    time_before_clean = time.time()
                    clean_out = cleanliness_inference_folder(data_dir_temp, verbose = False)
                    if verbose:
                        print(f" - Time spent on head {head}: {time.time()-time_before_clean:.0f} secs")
                    WallProb_and_clean_curr[f'Clean_NN_est_res_head_{head}_LeightonRex'] = clean_out

                    clean_color_out = cleanliness_inference_folder_colorbased(data_dir_temp,  clean_threshold = clean_threshold,  verbose = False)
                    WallProb_and_clean_curr[f'Clean_color_est_res_head_{head}'] = clean_color_out

                if DO_skip_wall_images:
                    if verbose:
                        print('Running wall estimation', end = "")
                    # Get the probability of wall for each image
                    time_before_wall = time.time()
                    wall_out_temp  = wall_inference_folder(data_dir_temp, verbose = False)
                    if verbose:
                        print(f" - Time spent on head {head}: {time.time()-time_before_wall:.0f} secs")
                    WallProb_and_clean_curr[f'Wall_est_res_head_{head}'] = wall_out_temp

                    """
                        CHECK THIS BELOW
                    """
                    P_wall_curr = [ppp if ccc == 1 else (1 - ppp) for ppp, ccc in zip(wall_out_temp['probs_out'], wall_out_temp['class_out'])]
                    WallProb_and_clean_curr[f'Wall_probability_head_{head}'] = P_wall_curr

            with open(fn_PID_wall, 'wb') as f:
                 pickle.dump(WallProb_and_clean_curr, f)

    # Run classifier on both heads
    for head in [1, 2]:

        imlist = list(range(getattr(PID_position['Raw'], f"n_frames_{head}")))

        if verbose:
            print(f'Running position estimation on head {head} ({len(imlist)} images)', end = "")
            time_pos_start = time.time()

        with torch.no_grad():

            # Store classifications of both heads with probs
            class_out = []
            probs_out = []
            probs_all = []
            clean_all = []

            for immy, im_curr in enumerate(imlist):

                # Load current image
                image = PID_position['Raw'].get_image(immy, head)

                # Estimate cleanliness
                if DO_skip_dirty_images:


                    # Reject image of cleanliness category is below limit
                    if WallProb_and_clean_curr[f'Clean_NN_est_res_head_{head}_LeightonRex']['class_out'][immy] < clean_cat_lowest_accepted:
                        # If the image is labeled as the dirtiest category
                        class_out.append(np.nan)
                        probs_out.append(np.nan)
                        probs_all.append(np.nan)
                        clean_all.append(np.nan)
                        continue

                    if f'Clean_color_est_res_head_{head}' in WallProb_and_clean_curr:
                        # Use already stored cleanliness values
                        clean_pixels = WallProb_and_clean_curr[f'Clean_color_est_res_head_{head}']['n_clean_pixels'][immy]
                        dirty_pixels = WallProb_and_clean_curr[f'Clean_color_est_res_head_{head}']['n_dirty_pixels'][immy]
                    else:
                        # Make image into numpy array and change color channel order to
                        # match what is the default in CV2 and assumed in clean est.
                        frame = np.array(image)
                        frame = frame[:,:,::-1] # From RGB to BGR

                        dirty_pixels, clean_pixels = classify_frame_cleanliness(frame)

                    # Skip if above threshold
                    if clean_pixels > 0:
                        ratio_curr = dirty_pixels / clean_pixels
                    else:
                        ratio_curr = 1e10
                    clean_all.append(ratio_curr)
                    if ratio_curr > cleanliness_threshold:
                        class_out.append(np.nan)
                        probs_out.append(np.nan)
                        probs_all.append(np.nan)
                        continue

                # Make tensor
                image = transforms.ToTensor()(image)

                # Normalize and resize image as done in the training phase
                image = torch.unsqueeze(resize_function(norm_function(image)), dim=0)

                if DO_skip_wall_images:
                    Prob_curr = WallProb_and_clean_curr[f'Wall_est_res_head_{head}']['probs_out'][immy]
                    if WallProb_and_clean_curr[f'Wall_est_res_head_{head}']['class_out'][immy] == 1:
                        wall_prob = Prob_curr
                    else:
                        wall_prob = 1 - Prob_curr

                    if wall_prob > Remove_wall_p_value_threshold:

                        # Only skip highly likely wall images
                        class_out.append(np.nan)
                        probs_out.append(np.nan)
                        probs_all.append(np.nan)
                        clean_all.append(np.nan)
                        continue

                # Run forward pass (autocast only has effect for GPU, and produce datatype dtype=torch.bfloat16 for cpu)
                if device == "cuda":
                    with autocast(device_type=device):
                        outputs = model_pos_class.forward(image.to(device=device)).cpu()
                else:
                    outputs = model_pos_class.forward(image.to(device=device))


                # Extract and return predicted class and probability
                _, predicted = outputs.max(dim=1)
                class_out.append(predicted.item())

                # Calculate probabilities using softmax (numerically stable)
                probs = torch.softmax(outputs, dim=1)
                probs_out.append(probs[0, predicted].item())
                probs_all.append(probs[0].numpy())

        if verbose:
            print(f" - Time used: {time.time()-time_pos_start:.0f} secs")

        # Store results in dictionary
        PID_position[f'Head_{head}_class_out']   = class_out
        PID_position[f'Head_{head}_probs_out']   = probs_out
        PID_position[f'Head_{head}_probs_all']   = probs_all

    # Merge results dict with PID meta information dict
    if PDF_report_exists:
        with open(fn_PID_curr, 'rb') as fp:
            PID_dict_curr = pickle.load(fp)

        if PID_dict_curr['PID'] == PID_position['PID']:
            PID_position = {**PID_dict_curr, **PID_position}

    return PID_position

def compute_uncertainty(mode_count, mode_prob_sum, valid_count, window_size, method='count'):
    """Compute uncertainty of the running mode prediction.

    Parameters
    ----------
    mode_count : int
        Integer count of images voting for the winning class.
    mode_prob_sum : float
        Sum of winning-class probabilities.
    valid_count : int
        Number of non-NaN images in the window.
    window_size : int
        Total window size.
    method : str, optional
        How to compute uncertainty. One of:

        - ``'count'`` — fraction of valid images voting for the winning class.
          Nominally in [0, 0.5] (0 = certain, 0.5 = maximally uncertain).
          Uses vote counts only.
        - ``'mean_prob'`` — mean probability of images voting for the winning class.
          Nominally in [0, 0.5]. Uses model confidence only.

    Returns
    -------
    float
        Uncertainty, nominally in [0, 0.5] (0 = certain, 0.5 = maximally uncertain).

    Notes
    -----
    When DO_use_transition_probs is True, the transition-probability override in
    running_mode can force the assigned class to the previous window's class even
    when it is not the plurality winner. In that case mode_count reflects a
    non-maximum vote count, which can fall below ceil(valid_count/4) — the
    pigeonhole lower bound that the [0, 0.5] scaling assumes. This causes
    uncertainty to marginally exceed 0.5 for those windows. This is not a
    formula error: the value correctly reflects that the assigned class is not
    the most-voted-for class in that window. Without the transition matrix,
    mode_count always equals the true plurality count and uncertainty stays
    within [0, 0.5].
    """
    if valid_count == 0:
        return 0.5

    if method == 'count':
        fraction = mode_count / valid_count
        return (1 - fraction) * (0.5 / 0.75)

    elif method == 'mean_prob':
        if mode_count == 0:
            return 0.5
        mean_prob = mode_prob_sum / mode_count
        return (1 - mean_prob) * (0.5 / 0.75)

    else:
        raise ValueError(f"Unknown uncertainty method: '{method}'. Choose 'count' or 'mean_prob'.")

def find_and_plot_running_mode(fd_out, window_size, PID_position, fn_ext_plot, fn_results,
                               DO_use_transition_probs = True,
                               DO_include_uncertainty = False, PDF_report_exists = False):
    """Find, save and plot the running mode within sliding windows."""
    Head_1_class_out = PID_position['Head_1_class_out']
    Head_2_class_out = PID_position['Head_2_class_out']
    Head_1_time_axis = PID_position.get('time_table_1') or getattr(PID_position['Raw'], "time_table_1")[:,1]
    Head_2_time_axis = PID_position.get('time_table_2') or getattr(PID_position['Raw'], "time_table_2")[:,1]
        
    if DO_include_uncertainty:
        # Extract probability value of most probable class of each head
        Head_1_probs_out = PID_position['Head_1_probs_out']
        Head_2_probs_out = PID_position['Head_2_probs_out']

    # Convert time axis to hours — running_mode_full produces one output per image,
    # so the time axis needs no trimming.
    time_axis_1_plot = np.array(Head_1_time_axis) / (1000*60*60)
    time_axis_2_plot = np.array(Head_2_time_axis) / (1000*60*60)

    if DO_include_reverse:
        # Define reverse data set and times
        Head_1_class_out_reversed = Head_1_class_out.copy()
        Head_1_class_out_reversed.reverse()
        Head_2_class_out_reversed = Head_2_class_out.copy()
        Head_2_class_out_reversed.reverse()
        time_axis_1_plot_reversed = np.flip(time_axis_1_plot)
        time_axis_2_plot_reversed = np.flip(time_axis_2_plot)

    # Calculate the running mode
    # Probabilities are passed when uncertainty is requested; running_mode always
    # tracks both integer counts and probability sums regardless.
    probs_1 = Head_1_probs_out if DO_include_uncertainty else ''
    probs_2 = Head_2_probs_out if DO_include_uncertainty else ''
    running_mode_data_1, running_mode_count_1, running_mode_prob_sum_1 = running_mode_full(
        data = Head_1_class_out, window_size = window_size, probs = probs_1, DO_use_transition_probs = DO_use_transition_probs)
    running_mode_data_2, running_mode_count_2, running_mode_prob_sum_2 = running_mode_full(
        data = Head_2_class_out, window_size = window_size, probs = probs_2, DO_use_transition_probs = DO_use_transition_probs)

    if DO_include_reverse:
        Head_1_probs_out_reversed = Head_1_probs_out.copy() if DO_include_uncertainty else ''
        Head_2_probs_out_reversed = Head_2_probs_out.copy() if DO_include_uncertainty else ''
        if DO_include_uncertainty:
            Head_1_probs_out_reversed.reverse()
            Head_2_probs_out_reversed.reverse()
        running_mode_data_1_reversed, running_mode_count_1_reversed, running_mode_prob_sum_1_reversed = running_mode_full(
            data = Head_1_class_out_reversed, window_size = window_size, probs = Head_1_probs_out_reversed, DO_use_transition_probs = DO_use_transition_probs)
        running_mode_data_2_reversed, running_mode_count_2_reversed, running_mode_prob_sum_2_reversed = running_mode_full(
            data = Head_2_class_out_reversed, window_size = window_size, probs = Head_2_probs_out_reversed, DO_use_transition_probs = DO_use_transition_probs)

    # Determine uncertainty
    if DO_include_uncertainty:

        # Compute NaN counts per window (number of NaN images in each window position).
        # Uses the same asymmetric edge handling as running_mode_full so that the output
        # length matches: one count per input image.
        def running_nan_count(data, window_size=201):
            counts = []
            for i in range(len(data)):
                min_ind = max(0,         i - int(np.floor(window_size / 2)))
                max_ind = min(len(data), i + int(np.ceil(window_size / 2)))
                window  = data[min_ind:max_ind]
                counts.append(sum(1 for x in window if (isinstance(x, float) and np.isnan(x))))
            return counts

        running_nan_count_1 = running_nan_count(data = Head_1_class_out, window_size = window_size)
        running_nan_count_2 = running_nan_count(data = Head_2_class_out, window_size = window_size)

        valid_count_1 = np.array([window_size - n for n in running_nan_count_1])
        valid_count_2 = np.array([window_size - n for n in running_nan_count_2])

        uncertainty_1 = np.array([compute_uncertainty(mode_count    = running_mode_count_1[i],
                                                       mode_prob_sum = running_mode_prob_sum_1[i],
                                                       valid_count   = valid_count_1[i],
                                                       window_size   = window_size,
                                                       method        = uncertainty_method)
                                   for i in range(len(running_mode_count_1))])
        uncertainty_2 = np.array([compute_uncertainty(mode_count    = running_mode_count_2[i],
                                                       mode_prob_sum = running_mode_prob_sum_2[i],
                                                       valid_count   = valid_count_2[i],
                                                       window_size   = window_size,
                                                       method        = uncertainty_method)
                                   for i in range(len(running_mode_count_2))])


        if DO_include_reverse:
            running_nan_count_1_reversed = running_nan_count(data = Head_1_class_out_reversed, window_size = window_size)
            running_nan_count_2_reversed = running_nan_count(data = Head_2_class_out_reversed, window_size = window_size)
            valid_count_1_reversed = np.array([window_size - n for n in running_nan_count_1_reversed])
            valid_count_2_reversed = np.array([window_size - n for n in running_nan_count_2_reversed])
            uncertainty_1_reversed = np.array([compute_uncertainty(mode_count    = running_mode_count_1_reversed[i],
                                                                    mode_prob_sum = running_mode_prob_sum_1_reversed[i],
                                                                    valid_count   = valid_count_1_reversed[i],
                                                                    window_size   = window_size,
                                                                    method        = uncertainty_method)
                                               for i in range(len(running_mode_count_1_reversed))])
            uncertainty_2_reversed = np.array([compute_uncertainty(mode_count    = running_mode_count_2_reversed[i],
                                                                    mode_prob_sum = running_mode_prob_sum_2_reversed[i],
                                                                    valid_count   = valid_count_2_reversed[i],
                                                                    window_size   = window_size,
                                                                    method        = uncertainty_method)
                                               for i in range(len(running_mode_count_2_reversed))])

    # Plot the original data and the running mode
    color_head_1_forw = 'b'
    color_head_2_forw = 'r'
    color_head_1_reve = 'c'
    color_head_2_reve = 'm'

    fig, ax = plt.subplots()

    if not DO_include_reverse:
        plt.plot(time_axis_1_plot,          running_mode_data_1+0.02,          label='Head 1', color = color_head_1_forw, linewidth=3)
    else:
        plt.plot(time_axis_1_plot,          running_mode_data_1+0.02,          label='Head 1 forward', color = color_head_1_forw, linewidth=3)
        plt.plot(time_axis_1_plot_reversed, running_mode_data_1_reversed-0.02, label='Head 1 reverse', color = color_head_1_reve, linewidth=3)
    if DO_include_uncertainty:
        plt.plot(time_axis_1_plot, running_mode_data_1 + uncertainty_1, color = color_head_1_forw, linestyle='dashed')
        plt.plot(time_axis_1_plot, running_mode_data_1 - uncertainty_1, color = color_head_1_forw, linestyle='dashed')
        if DO_include_reverse:
            plt.plot(time_axis_1_plot_reversed , running_mode_data_1_reversed  + uncertainty_1_reversed, color = color_head_1_reve, linestyle='dashed')
            plt.plot(time_axis_1_plot_reversed , running_mode_data_1_reversed  - uncertainty_1_reversed, color = color_head_1_reve, linestyle='dashed')

    if not DO_include_reverse:
        plt.plot(time_axis_2_plot,          running_mode_data_2+0.04,          label='Head 2',           color = color_head_2_forw, linewidth=3)
    else:
        plt.plot(time_axis_2_plot,          running_mode_data_2+0.04,          label='Head 2 forward',           color = color_head_2_forw, linewidth=3)
        plt.plot(time_axis_2_plot_reversed, running_mode_data_2_reversed-0.04, label='Head 2 reverse', color = color_head_2_reve, linewidth=3)
    if DO_include_uncertainty:
        plt.plot(time_axis_2_plot, running_mode_data_2 + uncertainty_2, color = color_head_2_forw, linestyle='dashed')
        plt.plot(time_axis_2_plot, running_mode_data_2 - uncertainty_2, color = color_head_2_forw, linestyle='dashed')
        if DO_include_reverse:
            plt.plot(time_axis_2_plot_reversed , running_mode_data_2_reversed  + uncertainty_2_reversed, color = color_head_2_reve, linestyle='dashed')
            plt.plot(time_axis_2_plot_reversed , running_mode_data_2_reversed  - uncertainty_2_reversed, color = color_head_2_reve, linestyle='dashed')

    # Determine where all agree on the class and plot green on these times
    t_all = np.concatenate([time_axis_1_plot, time_axis_2_plot])
    t_all.sort()

    class_all_agree = []
    for t in t_all:
        # Use timepoint that is closest in time to the current
        # (one of the two heads will have an image at the given current time,
        #  while the other will not have one).
        idx_1 = (np.abs(time_axis_1_plot - t)).argmin()
        idx_2 = (np.abs(time_axis_2_plot - t)).argmin()
        if DO_include_reverse:
            idx_1_rev = (np.abs(time_axis_1_plot_reversed - t)).argmin()
            idx_2_rev = (np.abs(time_axis_2_plot_reversed - t)).argmin()
            if running_mode_data_1[idx_1] == running_mode_data_2[idx_2] == running_mode_data_1_reversed[idx_1_rev] == running_mode_data_2_reversed[idx_2_rev]:
                class_all_agree.append(running_mode_data_1[idx_1])
            else:
                class_all_agree.append(np.nan)
        else:
            if running_mode_data_1[idx_1] == running_mode_data_2[idx_2]:
                class_all_agree.append(running_mode_data_1[idx_1])
            else:
                class_all_agree.append(np.nan)

    plt.plot(t_all , class_all_agree, color = "green", linewidth=10)
    plt.legend(loc = 'lower right')

    # Set y-axis limits
    ax.set_ylim(-0.1, 3.1)

    # Remove y-axis ticks
    ax.set_yticks([])

    # Add labels to the plot
    y_labels = ['Small\nintestine', 'Ascending\ncolon', 'Transverse\ncolon', 'Descending\ncolon']
    y_labels_pos = np.array([0, 1, 2, 3])

    # Set the custom y-tick positions and labels
    plt.yticks(y_labels_pos, y_labels)

    # Set labels and title
    ax.set_xlabel('Time [hours]')
    ax.set_ylabel('')
    ax.set_title(f'Running Mode Plot (Window Size {window_size})')

    # Get and plot time of known cecum and flexure points times
    if PDF_report_exists:
        if "T_cecum" in PID_position:
            T_cecum = str_to_ms(PID_position['T_cecum']) / (1000*60*60)
            ax.scatter(T_cecum,   0.5, color='black', marker='x', s=200)
        if "T_flexure_hepatic" in PID_position:   
            T_hepatic = str_to_ms(PID_position['T_flexure_hepatic']) / (1000*60*60)
            ax.scatter(T_hepatic, 1.5, color='black', marker='x', s=200)
        if "T_flexure_splenic" in PID_position:   
            T_splenic = str_to_ms(PID_position['T_flexure_splenic']) / (1000*60*60)
            ax.scatter(T_splenic, 2.5, color='black', marker='x', s=200)

    # Write plot to file
    plt.savefig(f'{fd_out}/Intestine_classifier_results{fn_ext_plot}.png', dpi = 300, bbox_inches='tight')
    plt.close()

    Running_mode_results = dict()
    Running_mode_results['window_size'] = window_size
    Running_mode_results['running_mode_data_1']             = running_mode_data_1
    Running_mode_results['running_mode_data_2']             = running_mode_data_2
    Running_mode_results['time_axis_1_plot']                = time_axis_1_plot
    Running_mode_results['time_axis_2_plot']                = time_axis_2_plot

    if DO_include_reverse:
         Running_mode_results['running_mode_data_1_reversed']    = running_mode_data_1_reversed
         Running_mode_results['running_mode_data_2_reversed']    = running_mode_data_2_reversed
         Running_mode_results['time_axis_1_plot_reversed']       = time_axis_1_plot_reversed
         Running_mode_results['time_axis_2_plot_reversed']       = time_axis_2_plot_reversed

    PID_position['Running_mode_results'] = Running_mode_results

    # Get transition times
    PID_position = get_transition_times(PID_position)

    # Write results dict to file
    with open(fn_results, 'wb') as f:
        pickle.dump(PID_position, f)

    return PID_position

def running_mode_full(data, window_size, probs='', DO_use_transition_probs=True):
    """Find running mode (majority vote) for every image in the sequence.

    Unlike running_mode, this produces one output per input image by using
    an asymmetric window near the edges of the data — the window shrinks
    rather than refusing to produce output. This means no images are
    discarded at the start or end of the recording.

    Parameters
    ----------
    data : list
        Per-image class predictions (int or np.nan).
    window_size : int
        Sliding window width (must be odd).
    probs : list, optional
        Per-image winning-class probabilities (float or np.nan).
        Pass an empty string ``''`` to omit probability tracking.
    DO_use_transition_probs : bool, optional
        If True, apply transition-probability weighting to discourage
        anatomically unlikely class transitions.

    Returns
    -------
    modes : np.ndarray
        Per-image majority-vote class labels, length = len(data).
    mode_count : np.ndarray
        Integer vote counts for the winning class.
    mode_prob_sum : np.ndarray
        Probability sums for the winning class.
    """
    if DO_use_transition_probs:
        transition_probability_matrix = get_transition_probability_matrix()

    use_probs = probs != '' and len(probs) == len(data)

    modes         = np.zeros(len(data)) - 1
    mode_count    = np.zeros(len(data))
    mode_prob_sum = np.zeros(len(data))

    ind_0 = 0
    while ind_0 < len(data):

        min_ind = max(0,         ind_0 - int(np.floor(window_size / 2)))
        max_ind = min(len(data), ind_0 + int(np.ceil(window_size / 2)))
        x_curr  = data[min_ind:max_ind]

        # --- First pass: initialize county from scratch ---
        if ind_0 == 0:
            county = [data[min_ind:max_ind].count(0), data[min_ind:max_ind].count(1),
                      data[min_ind:max_ind].count(2), data[min_ind:max_ind].count(3)]
            if use_probs:
                data_win  = data[min_ind:max_ind]
                probs_win = probs[min_ind:max_ind]
                county_prob = []
                for target_integer in range(4):
                    probby = [probs_win[i] for i, x in enumerate(data_win) if x == target_integer]
                    county_prob.append(np.nansum(probby))
            else:
                county_prob = [0.0, 0.0, 0.0, 0.0]

            max_count = max(county)
            if max_count > 0:
                winning = county.index(max_count)
                modes[ind_0]         = winning
                mode_count[ind_0]    = max_count
                mode_prob_sum[ind_0] = county_prob[winning]
            else:
                modes[ind_0]         = 0
                mode_count[ind_0]    = 0
                mode_prob_sum[ind_0] = 0.0

            ind_0 += 1
            continue

        # --- Subtract image leaving the back of the window (only if window start moved) ---
        if min_ind > 0:
            leaving = min_ind - 1
            if not np.isnan(data[leaving]):
                county[data[leaving]] -= 1
                if use_probs and not np.isnan(probs[leaving]):
                    county_prob[data[leaving]] -= probs[leaving]

        # --- Add image entering the front of the window ---
        # During the growing phase (start) and full-window sliding phase, the front advances.
        # Near the end, max_ind is clamped to len(data) so the window shrinks and i_last
        # does not change, meaning nothing new enters.
        i_last = max_ind - 1
        if ind_0 <= window_size or len(x_curr) == window_size:
            if not np.isnan(data[i_last]):
                county[data[i_last]] += 1
                if use_probs and not np.isnan(probs[i_last]):
                    county_prob[data[i_last]] += probs[i_last]

        # --- Compute mode from current county ---
        max_count = max(county)
        if max_count > 0:
            winning              = county.index(max_count)
            modes[ind_0]         = winning
            mode_count[ind_0]    = max_count
            mode_prob_sum[ind_0] = county_prob[winning]
        else:
            modes[ind_0]         = modes[ind_0-1]
            mode_count[ind_0]    = 0
            mode_prob_sum[ind_0] = 0.0

        # Do not output any mode value if 25% or more of the images in the window are NaN
        #if np.sum(county) < (window_size*0.75):
        #    modes[ind_0] = np.nan
        #    mode_count[ind_0] = np.nan
        if DO_use_transition_probs:
            if not np.isnan(modes[ind_0]) and not np.isnan(modes[ind_0-1]):
                if modes[ind_0] != modes[ind_0-1]:
                    weight = transition_probability_matrix[int(modes[ind_0-1])][int(modes[ind_0])]
                    if county[int(modes[ind_0])] * weight < county[int(modes[ind_0-1])]:
                        modes[ind_0]         = modes[ind_0-1]
                        mode_count[ind_0]    = county[int(modes[ind_0])]
                        mode_prob_sum[ind_0] = county_prob[int(modes[ind_0])]

        ind_0 += 1

    return modes, mode_count, mode_prob_sum


def get_transition_times(PID_position):
    """Get transition times."""
    Running_mode_results = PID_position['Running_mode_results']
    # Transition times are time points when the smooth (mode or mean) of the
    # windows smoother changes from one class to another.
    diffy1  = [Running_mode_results['running_mode_data_1'][a+1]-Running_mode_results['running_mode_data_1'][a] for a in range(len(Running_mode_results['running_mode_data_1'])-1)]
    diffy2  = [Running_mode_results['running_mode_data_2'][a+1]-Running_mode_results['running_mode_data_2'][a] for a in range(len(Running_mode_results['running_mode_data_2'])-1)]
    if DO_include_reverse:
        diffy1rev  = [Running_mode_results['running_mode_data_1_reversed'][a+1]-Running_mode_results['running_mode_data_1_reversed'][a] for a in range(len(Running_mode_results['running_mode_data_1_reversed'])-1)]
        diffy2rev  = [Running_mode_results['running_mode_data_2_reversed'][a+1]-Running_mode_results['running_mode_data_2_reversed'][a] for a in range(len(Running_mode_results['running_mode_data_2_reversed'])-1)]


    transition_indices_1     = [i for i, e in enumerate(diffy1)    if e != 0 and np.isnan(e)==False]
    transition_indices_2     = [i for i, e in enumerate(diffy2)    if e != 0 and np.isnan(e)==False]
    if DO_include_reverse:
        transition_indices_1_rev = [i for i, e in enumerate(diffy1rev) if e != 0 and np.isnan(e)==False]
        transition_indices_2_rev = [i for i, e in enumerate(diffy2rev) if e != 0 and np.isnan(e)==False]

    transition_times =      [ttime for i, ttime in enumerate(Running_mode_results['time_axis_1_plot'])          if i in transition_indices_1]
    transition_times.extend([ttime for i, ttime in enumerate(Running_mode_results['time_axis_2_plot'])          if i in transition_indices_2])
    if DO_include_reverse:
        transition_times.extend([ttime for i, ttime in enumerate(Running_mode_results['time_axis_1_plot_reversed']) if i in transition_indices_1_rev])
        transition_times.extend([ttime for i, ttime in enumerate(Running_mode_results['time_axis_2_plot_reversed']) if i in transition_indices_2_rev])

    transition_times = list(set(transition_times))
    transition_times.sort()

    transition_times_rounded = list(set([np.round(i, 2) for i in transition_times]))
    transition_times_rounded.sort()

    PID_position['transition_times_rounded']  = transition_times_rounded
    PID_position['transition_index_head_1']   = transition_indices_1
    PID_position['transition_index_head_2']   = transition_indices_2
    return PID_position

def get_transition_probability_matrix(seg_change_prob = 0.9, seg_jump_prob = 0.05):
    """Define probabilities of transitioning from one segment to the next."""
    # Row is current segment, column is next segment
    transition_probability_matrix = [[            1, seg_change_prob,   seg_jump_prob,  seg_jump_prob],
                                     [seg_jump_prob,               1, seg_change_prob,  seg_jump_prob],
                                     [seg_jump_prob, seg_change_prob,               1, seg_change_prob],
                                     [seg_jump_prob,   seg_jump_prob, seg_change_prob,               1]]
    # This definition here makes it very hard to jump from segment 2 back to segment 1,
    # which makes sense as this is from ascending colon back to small intestine.
    # However, when doing the backwards approach, this does not make sense.
    # When doing backwards, this matrix should be transposed.

    return transition_probability_matrix

def get_transition_type(Running_mode_results, head, transition_index):
    """Determine between which colon segments the pill camera is classified to move."""
    seg_from = int(Running_mode_results[f'running_mode_data_{head}'][transition_index])
    seg_to   = int(Running_mode_results[f'running_mode_data_{head}'][transition_index+1])
    if   seg_from == 0 and seg_to == 1:
        return "SB_to_AC"
    elif seg_from == 1 and seg_to == 0:
        return "AC_to_SB"
    elif seg_from == 1 and seg_to == 2:
        return "AC_to_TC"
    elif seg_from == 2 and seg_to == 1:
        return "TC_to_AC"
    elif seg_from == 2 and seg_to == 3:
        return "TC_to_DC"
    elif seg_from == 3 and seg_to == 2:
        return "DC_to_TC"
    else:
        return "Multisegment_jump"

def find_flexures_in_transition_regions(fd_PID, fd_out, window_size, PID_position, landmark_to_look_for,
                                        verbose = False):
    """Detect flexures image in transition regions."""
    Running_mode_results = PID_position['Running_mode_results']

    # Load mean and stdevs for the different color channels
    mean_per_channel, stdv_per_channel =  Mean_and_stdev_of_pillCameraImages()

    # Define preprocessing steps for each image
    norm_function   = transforms.Normalize(mean=mean_per_channel, std=stdv_per_channel )
    resize_function = transforms.Resize(224, antialias=True)

    if   landmark_to_look_for == "Hepatic_flexure":
        model_flexure = load_resnet50_model(load_model_path = hepatic_flexure_model_path, n_classes = 2, device = device)
    elif landmark_to_look_for == "Splenic_flexure":
        model_flexure = load_resnet50_model(load_model_path = splenic_flexure_model_path, n_classes = 2, device = device)
    else:
        print("Non-valid flexure to look for given - exiting landmark hunt!")
        return PID_position

    if DO_save_landmark_images:
        fd_landmark = f"{fd_out}/Landmark_images/{landmark_to_look_for}/"
        makedirs(fd_landmark, exist_ok=True)

    for head in [1, 2]:

        if verbose:
           print(f"Finding images of {landmark_to_look_for} for head {head}", end = "")
           time_landmark_start = time.time()

        found_landmarks_curr_head = []

        # Note: PID_position['Raw'] must still be present here.
        # cleanup_raw_object() which removes it is always called after landmark detection.
        imlist = list(range(getattr(PID_position['Raw'], f"n_frames_{head}")))

        for trans_ind in PID_position[f'transition_index_head_{head}']:

            # Determine what kind of transitions happens
            transition_type = get_transition_type(Running_mode_results, head, trans_ind)
            if   landmark_to_look_for == "Hepatic_flexure":
                if transition_type not in ['AC_to_TC', 'TC_to_AC']:
                    # Only look for hepatic flexure if AC --> TC or vice versa.
                    continue
            elif landmark_to_look_for == "Splenic_flexure":
                if transition_type not in ['TC_to_DC', 'DC_to_TC']:
                    # Only look for splenic flexure if TC --> DC or vice versa.
                    continue

            curr_region = [trans_ind - int(np.floor(window_size/2)), trans_ind + int(np.floor(window_size/2))]
            # Clamp to valid image index range
            ind_max_curr = len(Running_mode_results[f'running_mode_data_{head}']) - 1
            curr_region = [max(0, min(ind_max_curr, i)) for i in curr_region]

            # torch.no_grad() prevents gradient computation during inference, saving memory.
            # This is separate from model.eval() which controls dropout/batchnorm behaviour.
            with torch.no_grad():

                # Store classifications of both heads with probs
                class_out = []
                probs_out = []
                probs_all = []

                for immy, im_curr in enumerate(imlist):
                    if curr_region[0] <= immy <= curr_region[1]:

                        # Load current image
                        image_PIL = PID_position['Raw'].get_image(immy, head)
                        image = transforms.ToTensor()(image_PIL)

                        # Normalize and resize image as done in the training phase
                        image = torch.unsqueeze(resize_function(norm_function(image)), dim=0)

                        # Run forward pass
                        outputs = model_flexure.forward(image.to(device=device)).cpu()

                        # Extract and return predicted class and probability
                        maxprob, predicted = outputs.max(dim=1)
                        class_out.append(predicted.item())
                        # Calculate probabilities using softmax (numerically stable)
                        probs = torch.softmax(outputs, dim=1)
                        probs_out.append(probs[0, predicted].item())
                        probs_all.append(probs[0].numpy())

                        if predicted == 0 and  probs_out[-1] > 0.9:
                            found_landmarks_curr_head.append(im_curr)
                            if verbose_landmark_findings:
                                print(f"{landmark_to_look_for} image in head {head}, im_num {immy}, imagename {im_curr}, transition point {trans_ind}")
                            if DO_save_landmark_images:
                                fn_out = f"{fd_landmark}/{landmark_to_look_for}_head_{head}_imNum{immy}.jpg"
                                image_PIL.save(fn_out, format="JPEG")
                                        

        if landmark_to_look_for == "Hepatic_flexure":
           PID_position[f'hepatic_flexure_head_{head}'] = found_landmarks_curr_head
        elif landmark_to_look_for == "Splenic_flexure":
           PID_position[f'splenic_flexure_head_{head}'] = found_landmarks_curr_head


        if verbose:
            print(f" - Time spent on head {head}: {time.time()-time_landmark_start:.0f} secs")

    return PID_position

def find_landmark_in_colon_region(fd_PID, fd_out, window_size, PID_position, landmark_to_look_for,
                                  verbose = False):
    """Find landmarks in a given colon region."""
    Running_mode_results = PID_position['Running_mode_results']

    # Load mean and stdevs for the different color channels
    mean_per_channel, stdv_per_channel =  Mean_and_stdev_of_pillCameraImages()

    # Define preprocessing steps for each image
    norm_function   = transforms.Normalize(mean=mean_per_channel, std=stdv_per_channel )
    resize_function = transforms.Resize(224, antialias=True)

    if   landmark_to_look_for == "ICV":
        model_landmark = load_resnet50_model(load_model_path = ICV_model_path, n_classes = 2, device = device)
    elif landmark_to_look_for == "Appendix":
        model_landmark = load_resnet50_model(load_model_path = appendix_model_path, n_classes = 2, device = device)
    else:
        print("Non-valid landmark to look for given - exiting landmark hunt!")
        return PID_position
    if DO_save_landmark_images:
        fd_landmark = f"{fd_out}/Landmark_images/{landmark_to_look_for}/"
        makedirs(fd_landmark, exist_ok=True)

    def _get_colon_segment(landmark_to_look_for):
        """Return the colon segment name and class index to search for a given landmark."""
        if landmark_to_look_for == "ICV":
            return ("AC", 1)
        if landmark_to_look_for == "Appendix":
            return ("AC", 1)

    colon_segment_to_look_in = _get_colon_segment(landmark_to_look_for)

    for head in [1, 2]:

        found_landmarks_curr_head = []

        if verbose:
           print(f"Finding images of {landmark_to_look_for} for head {head}", end = "")
           time_landmark_start = time.time()

        # Note: PID_position['Raw'] must still be present here.
        # cleanup_raw_object() which removes it is always called after landmark detection.
        imlist = list(range(getattr(PID_position['Raw'], f"n_frames_{head}")))

        # torch.no_grad() prevents gradient computation during inference, saving memory.
        # This is separate from model.eval() which controls dropout/batchnorm behaviour.
        with torch.no_grad():

            # Store classifications of both heads with probs
            class_out = []
            probs_out = []
            probs_all = []

            for immy, im_curr in enumerate(imlist):

                if int(Running_mode_results[f'running_mode_data_{head}'][immy]) == colon_segment_to_look_in[1]:

                    # Load current image
                    image_PIL = PID_position['Raw'].get_image(immy, head)
                    image = transforms.ToTensor()(image_PIL)

                    # Normalize and resize image as done in the training phase
                    image = torch.unsqueeze(resize_function(norm_function(image)), dim=0)

                    # Run forward pass
                    outputs = model_landmark.forward(image.to(device=device)).cpu()

                    # Extract and return predicted class and probability
                    maxprob, predicted = outputs.max(dim=1)
                    class_out.append(predicted.item())
                    # Calculate probabilities using softmax (numerically stable)
                    probs = torch.softmax(outputs, dim=1)
                    probs_out.append(probs[0, predicted].item())
                    probs_all.append(probs[0].numpy())

                    if predicted == 0 and  probs_out[-1] > 0.9:
                        found_landmarks_curr_head.append(im_curr)
                        if verbose_landmark_findings:
                            print(f"{landmark_to_look_for} image in head {head}, im_num {immy}, imagename {im_curr}")
                        if DO_save_landmark_images:
                            fn_out = f"{fd_landmark}/{landmark_to_look_for}_head_{head}_imNum{immy}.jpg"
                            image_PIL.save(fn_out, format="JPEG")
                                

        if   landmark_to_look_for == "ICV":
           PID_position[f'ICV_head_{head}'] = found_landmarks_curr_head
        elif landmark_to_look_for == "Appendix":
           PID_position[f'appendix_head_{head}'] = found_landmarks_curr_head

        if verbose:
            print(f" - Time spent on head {head}: {time.time()-time_landmark_start:.0f} secs")

    return PID_position

def get_PID_list(folder_look_in = None):
    """Get list of PIDs to loop over."""
    # Retrieve the list of available PIDs from the working directory
    if folder_look_in is None:
        folder_look_in = getcwd()
    # Replace back-slashes with forward slashes to ensure glob works
    folder_look_in = normpath(folder_look_in).replace('\\', '/')
        
    # --- Look for PIDs in the pillcam raw format ---
    all_folders = sorted([d for d in listdir(folder_look_in) if isdir(join(folder_look_in, d))])

    PID_list_raw = []
    for fd_curr in all_folders:
        fd_PID_curr = join(folder_look_in, fd_curr)
        if (len(glob(f"{fd_PID_curr}/stream/" + '*.gfd')) == 1 and
            len(glob(f"{fd_PID_curr}/stream/" + '*.gvf')) == 1 and
            len(glob(f"{fd_PID_curr}/stream2/" + '*.gfd')) == 1 and
            len(glob(f"{fd_PID_curr}/stream2/" + '*.gvf')) == 1):
            PID_list_raw.append(fd_curr)

    # Try to get five-digit number from folder name
    PID_list_raw_out = []
    for fd_curr in PID_list_raw:
        if re.search(r"\d{6}", fd_curr):
            PID_list_raw_out.append((re.search(r"\d{6}", fd_curr)[0], fd_curr))
        elif re.search(r"\d{5}", fd_curr):
            PID_list_raw_out.append((re.search(r"\d{5}", fd_curr)[0], fd_curr))
        else:
            print(f"Folder '{fd_curr}' has raw CCE files, but folder name does not contain needed five-digit PID.")

    PID_list_raw_out.sort()
    return PID_list_raw_out

def position_inference(fd_PID, fd_out = None):
    r"""Run position inference. The input full path must NOT end with an \ or / ."""
    # Load intestine section classifier
    model_pos_class = load_resnet50_model(load_model_path=position_model_path, n_classes=n_classes, device=device)
    
    if fd_out is None:
        fd_out = fd_PID
        DO_copy_video_data_to_new_folder = False
    else:
        if fd_out != fd_PID:
            DO_copy_video_data_to_new_folder = True
        else:
            DO_copy_video_data_to_new_folder = False

    # --- Check if PID is in the pillcam raw format ---
    fd_only_curr = basename(fd_PID)
    PID = None
    if (len(glob(join(fd_PID, 'stream', '*.gfd'))) == 1 and
        len(glob(join(fd_PID, 'stream', '*.gvf'))) == 1 and
        len(glob(join(fd_PID, 'stream2', '*.gfd'))) == 1 and
        len(glob(join(fd_PID, 'stream2', '*.gvf'))) == 1):

        if DO_copy_video_data_to_new_folder and DO_not_copy_video_data_even_if_making_new_dir is False:
            # Copy 'stream' folder
            copytree(join(fd_PID, 'stream'), join(fd_out, 'stream'), dirs_exist_ok=True)
            # Copy 'stream2' folder
            copytree(join(fd_PID, 'stream2'), join(fd_out, 'stream2'), dirs_exist_ok=True)

        # Try to get five-digit number from folder name
        if re.search(r"\d{6}", fd_only_curr):
            PID = re.search(r"\d{6}", fd_only_curr)[0]
        elif re.search(r"\d{5}", fd_only_curr):
            PID = re.search(r"\d{5}", fd_only_curr)[0]
        else:
            PID = None
            print(f"Folder '{fd_only_curr}' has raw CCE files, but folder name does not contain needed five or six-digit PID.")

    # Ensure Info_PID files are copied to output folder when using separate fd_out_main
    if PID is not None and fd_out != fd_PID:
        for suffix in ['.pickle', '.txt', '_noPDF.pickle', '_noPDF.txt']:
            src = f"{fd_PID}/Info_PID_{PID}{suffix}"
            dst = f"{fd_out}/Info_PID_{PID}{suffix}"
            if exists(src) and not exists(dst):
                try:
                    copy2(src, dst)
                    if verbose:
                        print(f"Copied {basename(src)} to output folder")
                except Exception as e:
                    print(f"Warning: Could not copy {basename(src)}: {e}")
    
    if DO_redo_calc_even_if_done is False and exists(f"{fd_out}/Position_estimate_done.txt"):
        return False
    
    if verbose:
        print(f"\nRunning inference on {basename(fd_PID)}")
        time_PID_start = time.time()

    # Make dict to store results of position estimation
    PID_position = PID_position_dict_init(PID, fd_PID, n_classes, position_model_used, fd_out)

    if DO_detect_landmarks:
        makedirs(f"{fd_out}/Landmark_images", exist_ok=True)

    # --- Define variables for current PID ---
    fn_PID_curr = f"{fd_out}/Info_PID_{PID}.pickle"

    PDF_report_exists = True
    if not exists(fn_PID_curr):
        PDF_report_exists = False
    fn_results = f'{fd_out}/Intestine_classifier_results{fn_ext_out}.pickle'

    # --- Run position classifier (or load if already done) ---
    if verbose:
        print("Run position classifier")
    PID_position = classify_position(model_pos_class, fd_PID, PID_position, fn_results, fn_PID_curr, PDF_report_exists)

    # --- Make intestine segment prediction plot, determine transition times and indices ---
    if verbose:
        print("Find and plot running mode")
    PID_position = find_and_plot_running_mode(fd_out, window_size, PID_position, fn_ext_plot, fn_results,
                      DO_use_transition_probs, DO_include_uncertainty, PDF_report_exists)
    
    if DO_detect_landmarks:
        if verbose:
            print("Detect landmarks")

        # --- Look for ileocecal valve and appendix in the ascending colon ---
        PID_position = find_landmark_in_colon_region(fd_PID, fd_out, window_size, PID_position,
                                                    landmark_to_look_for = "ICV")
        PID_position = find_landmark_in_colon_region(fd_PID, fd_out, window_size, PID_position,
                                                    landmark_to_look_for = "Appendix")

        # --- Look for flexure images in transition regions ---
        PID_position = find_flexures_in_transition_regions(fd_PID, fd_out, window_size, PID_position,
                                                        landmark_to_look_for = "Hepatic_flexure",
                                                        verbose = verbose)
        PID_position = find_flexures_in_transition_regions(fd_PID, fd_out, window_size, PID_position,
                                                        landmark_to_look_for = "Splenic_flexure",
                                                        verbose = verbose)
        
        # Write results dict to file
        with open(fn_results, 'wb') as f:
            pickle.dump(PID_position, f)

        if verbose_landmark_findings:
            for head in [1, 2]:
                if f'hepatic_flexure_head_{head}' in PID_position:
                    print(f"Hepatic flexure head {head}: {PID_position[f'hepatic_flexure_head_{head}']}")
                
    # Remove heavy 'Raw' object and huge probs_all (already cleaned in classify_position)
    PID_position = cleanup_raw_object(PID_position)

    # Re-save the cleaned version (overwrite)
    with open(fn_results, 'wb') as f:
        pickle.dump(PID_position, f, protocol=pickle.HIGHEST_PROTOCOL)

    if verbose:
        print(f"Time spent on PID {PID}: {(time.time()-time_PID_start)/60:.1f} min")

    # Mark that folder has done the position estimate
    with open(f"{fd_out}/Position_estimate_done.txt", mode='a'): pass


if __name__ == "__main__":
    
    single_pid_folder = None  # Set if a single PID_XXXXX path is given directly

    if len(sys.argv) > 1:
        # Enter if this script was called from the command line, which can include an input or not
        input_string = normpath(sys.argv[1].strip())

        if re.search(r'^PID_\d{5}$', basename(input_string)):
            # If input is on the format "PID_XXXXX", use this as the single folder
            single_pid_folder = join(getcwd(), basename(input_string))
        else:
            if exists(input_string) and isdir(input_string):
                # Check if this looks like a patient folder (has stream/ subfolders)
                # If so, treat it as a single patient folder directly
                if (len(glob(f"{input_string}/stream/*.gfd")) > 0 or
                        len(glob(f"{input_string}/stream2/*.gfd")) > 0):
                    single_pid_folder = input_string
                else:
                    folder_look_in = input_string
                    PID_list = get_PID_list(folder_look_in)
            else:
                raise ValueError(f"Input path does not exist or is not a folder: {input_string} - Exiting")
    else:
        # Find list of PID folders to iter over in current working directory
        PID_list = get_PID_list()
        folder_look_in = getcwd()

    if single_pid_folder is not None:
        fd_PID_list = [single_pid_folder]
    else:
        fd_PID_list = [f"{folder_look_in}/{piddy[1]}" for piddy in PID_list]

    if verbose:
        print("Compiled list of folders to analyse:")
        for d in fd_PID_list:
            print(f"  {basename(d)}")

    # Loop over all PIDs in list
    for iii, fd_PID in enumerate(fd_PID_list):
        # Define output folder
        if fd_out_main is not None:
            fd_out = join(fd_out_main, basename(fd_PID))
            makedirs(fd_out, exist_ok=True)
        else:
            fd_out = None

        # Run position estimation
        position_inference(fd_PID, fd_out)
