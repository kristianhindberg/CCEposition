# -*- coding: utf-8 -*-
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Kristian Dalsbø Hindberg
"""Auxiliary classes and functions for CCE position and cleanliness inference.

Provides:

- PID_raw_info — loads and exposes images and time tables from raw Colon2
  pill camera recordings (.gfd / .gvf format).
- classify_frame_cleanliness — color-based per-frame cleanliness estimate.
- Mean_and_stdev_of_pillCameraImages — fixed per-channel normalisation
  statistics for the Colon2 camera.
- str_to_ms — time string to millisecond conversion.

Author
------
    Kristian Dalsbø Hindberg, UiT — The Arctic University of Norway, 2023—2026

GitHub
------
    https://github.com/kristianhindberg/CCEposition
"""

import io
import re
import numpy as np
from os.path import join
from PIL import Image
import pandas as pd
import datetime
from glob import glob
import cv2


class PID_raw_info():
    """Load and expose images and time tables for a single CCE patient recording.

    Reads raw Colon2 pill camera data directly from the .gvf (image) and .gfd
    (time table) binary files in the stream/ and stream2/ subdirectories of a
    patient folder.

    Parameters
    ----------
    fd_PID : str
        Path to the patient folder containing stream/ and stream2/.
    fd_out : str, optional
        Output directory for any files written to disk. Defaults to fd_PID.
    """

    def __init__(self, fd_PID: str, fd_out: str = None):
        self.fd_PID = fd_PID
        self.fd_out = fd_PID if fd_out is None else fd_out

        self._get_all_images_from_given_head(1)
        self._get_all_images_from_given_head(2)

        self._get_time_table_of_given_head(1)
        self._get_time_table_of_given_head(2)

        self._GetMergedTimeTable()

        self.more_info = dict()

    def _get_all_images_from_given_head(self, head):
        """Load the raw image byte stream for a given head and locate frame boundaries."""
        stream_dir = 'stream' if head == 1 else 'stream2'
        fn_rawvideo_in = glob(join(self.fd_PID, stream_dir, '*.gvf'))[0]

        Data_raw = self._read_raw_images(fn_rawvideo_in, np.uint8)

        A  = np.where(Data_raw == 255)
        A0 = A[1]
        A2 = A[1] - 2
        A1 = np.where(Data_raw == 216)[1] - 1

        C1 = np.intersect1d(A0, A1)
        C2 = np.intersect1d(A1, A2)
        C3 = np.intersect1d(C1, C2)

        setattr(self, f"n_frames_{head}", C3.shape[0] - 1)
        setattr(self, f"Data_raw_{head}", Data_raw)
        setattr(self, f"C3_{head}", C3)

    def _read_raw_images(self, filename, precision=np.uint8):
        """Read the raw binary video file into a 1×N byte array."""
        with open(filename, 'rb') as fid:
            data_array = np.fromfile(fid, precision).reshape((-1, 1)).T
        return data_array

    def get_image(self, im_num, head, im_type="PIL"):
        """Return a single frame from the raw recording.

        Parameters
        ----------
        im_num : int
            Zero-based frame index (0 to n_frames_<head> - 1).
        head : int
            Camera head — 1 or 2.
        im_type : str, optional
            'PIL' (default) returns a PIL Image; 'Numpy' returns a raw byte array.

        Returns
        -------
        PIL.Image.Image or numpy.ndarray or None
        """
        n_frames = getattr(self, f"n_frames_{head}")
        if not (0 <= im_num < n_frames):
            print(f"Image number {im_num} out of range. Valid values are 0 to {n_frames - 1}.")
            return None

        Data_raw = getattr(self, f"Data_raw_{head}", None)
        C3       = getattr(self, f"C3_{head}", None)

        if Data_raw is None or C3 is None:
            print("Raw data not loaded.")
            return None

        chunk = Data_raw[0][C3[im_num]:C3[im_num + 1]]
        if im_type == "PIL":
            im_loaded = Image.open(io.BytesIO(chunk))
        elif im_type == "Numpy":
            im_loaded = np.frombuffer(chunk, dtype=np.uint8)
        else:
            raise ValueError(f"Unknown im_type '{im_type}'. Use 'PIL' or 'Numpy'.")

        setattr(self, f"Image_curr_{head}", im_loaded)
        return im_loaded

    def _get_time_table_of_given_head(self, head):
        """Load the .gfd time table for a given head."""
        stream_dir = 'stream' if head == 1 else 'stream2'
        fn_list = glob(join(self.fd_PID, stream_dir, '*.gfd'))

        if len(fn_list) == 0:
            print(f"No time table data found for head {head} of PID {self.fd_PID}")
            return None
        elif len(fn_list) > 1:
            print(f"More than one time table file found for head {head} of PID {self.fd_PID}")
            return None

        setattr(self, f"time_table_{head}", self._GetTimeTable(fn_list[0]))

    def _GetTimeTable(self, fn_timetable_in):
        """Parse a .gfd binary time table file into a (N, 6) integer array.

        Returns
        -------
        numpy.ndarray of shape (N, 6)
            Columns: frame_index, abs_time_ms, hour, minute, second, microsecond.
        """
        with open(fn_timetable_in, 'rb') as _file:
            raw_data = np.fromfile(_file, np.uint32)

        magic_01 = (raw_data[0] & 0xFFFF0000) // 2**16
        if (magic_01 % 4) != 0:
            raise RuntimeError(f'Magic byte at position 3 not divisible by 4 in {fn_timetable_in}')

        _rec_len = magic_01 // 4
        _data    = raw_data[24:]
        _n       = len(_data) // _rec_len
        _data    = np.reshape(_data[:_rec_len * _n], (_n, _rec_len))

        frame_times_abs_ms = _data[:, 3]

        Time_hr  = frame_times_abs_ms // (60 * 60 * 1000)
        _time    = frame_times_abs_ms - Time_hr * 60 * 60 * 1000
        Time_min = _time // (60 * 1000)
        _time   -= Time_min * 60 * 1000
        Time_sec = _time // 1000
        _time   -= Time_sec * 1000
        Time_msec = _time

        vec_time    = np.vectorize(datetime.time)
        frame_times = vec_time(Time_hr, Time_min, Time_sec, Time_msec)

        time_tb = np.zeros((len(frame_times_abs_ms), 6))
        for i, ttt in enumerate(frame_times):
            time_tb[i] = i + 1, frame_times_abs_ms[i], ttt.hour, ttt.minute, ttt.second, ttt.microsecond

        return time_tb.astype(int)

    def _GetMergedTimeTable(self, time_main=None):
        """Merge the time tables of both heads into a single chronological sequence.

        All images of both heads are included. Since the two heads are not
        time-synchronised, each head will have repeated entries in the output
        at time points where the other head had the closest image.

        The merged table is stored as self.time_table_merged — a list of
        (head, image_index, abs_time_ms) tuples in chronological order.
        """
        timetable_1   = pd.DataFrame(self.time_table_1)
        image_1       = timetable_1[0].values
        image_times_1 = timetable_1[1].values

        timetable_2   = pd.DataFrame(self.time_table_2)
        image_2       = timetable_2[0].values
        image_times_2 = timetable_2[1].values

        # Two-pointer merge of both sorted time sequences
        i1, i2 = 0, 0
        n1, n2 = len(image_times_1), len(image_times_2)
        im_time_merged = []

        while i1 < n1 and i2 < n2:
            if image_times_1[i1] < image_times_2[i2]:
                im_time_merged.append((1, image_1[i1], image_times_1[i1]))
                i1 += 1
            else:
                im_time_merged.append((2, image_2[i2], image_times_2[i2]))
                i2 += 1

        while i1 < n1:
            im_time_merged.append((1, image_1[i1], image_times_1[i1]))
            i1 += 1

        while i2 < n2:
            im_time_merged.append((2, image_2[i2], image_times_2[i2]))
            i2 += 1

        self.time_table_merged = im_time_merged

    def _write_merged_timetable_to_file(self, time_main=None):
        """Write the merged time table to a CSV file in the output folder."""
        im_time_merged = self.time_table_merged

        t_first_1  = self.time_table_1[0][1]
        t_first_2  = self.time_table_2[0][1]
        im_first_1 = self.time_table_1[0][0]
        im_first_2 = self.time_table_2[0][0]

        if time_main is None:
            time_main = im_time_merged[0][2]

        with open(f'{self.fd_out}Time_table_merged.txt', 'w') as f:
            f.write('Index_1,Time_1,Index_2,Time_2,T_delta,T_sync_1,T_sync_2\n')
            for t_ind_curr, t_curr in enumerate(im_time_merged):

                if t_curr[0] == 1:
                    t_first_1  = t_curr[2]
                    im_first_1 = t_curr[1]
                if t_curr[0] == 2:
                    t_first_2  = t_curr[2]
                    im_first_2 = t_curr[1]

                t_rel_to_main_1 = t_first_1 - time_main
                t_rel_to_main_2 = t_first_2 - time_main

                if t_ind_curr == 0:
                    first_row = (f"{im_first_1},{t_first_1},{im_first_2},{t_first_2},"
                                 f"{t_first_1 - t_first_2},{t_rel_to_main_1},{t_rel_to_main_2}\n")
                if t_ind_curr == 1:
                    if first_row == (f"{im_first_1},{t_first_1},{im_first_2},{t_first_2},"
                                     f"{t_first_1 - t_first_2},{t_rel_to_main_1},{t_rel_to_main_2}\n"):
                        del first_row
                        continue

                f.write((f"{im_first_1},{t_first_1},{im_first_2},{t_first_2},"
                         f"{t_first_1 - t_first_2},{t_rel_to_main_1},{t_rel_to_main_2}\n"))


def classify_frame_cleanliness(frame, clean_threshold=0.9):
    """Count clean and dirty pixels in a CCE image using a color-channel ratio.

    Applies a circular mask centred on the image and classifies each pixel
    as clean or dirty based on the ratio (R - G) / (G - B). Pixels outside
    the circular field of view are excluded from the count.

    Parameters
    ----------
    frame : numpy.ndarray
        BGR image as a (H, W, 3) uint8 array.
    clean_threshold : float, optional
        Pixels with ratio above this value are classified as clean. Default 0.9.

    Returns
    -------
    tuple of (int, int)
        (dirty_pixel_count, clean_pixel_count).
    """
    eps = 2 * np.finfo(float).eps

    mask   = np.zeros(frame.shape[:2], dtype="uint8")
    cv2.circle(mask, (128, 128), 126, 255, -1)
    masked = cv2.bitwise_and(frame, frame, mask=mask).astype(int)

    corner_pixels = mask.size - np.sum(mask) / 255

    J = (masked[:, :, 2] - masked[:, :, 1]) / (masked[:, :, 1] - masked[:, :, 0] - eps)

    cleanliness  = J > clean_threshold
    clean_pixels = np.sum(cleanliness)
    dirty_pixels = J.size - clean_pixels - corner_pixels

    return int(dirty_pixels), int(clean_pixels)


def Mean_and_stdev_of_pillCameraImages():
    """Return fixed per-channel normalisation statistics for Colon2 CCE images.

    Statistics were estimated from a representative set of colon segment images.
    These values are used to normalise images before passing them to the neural
    network classifiers.

    Returns
    -------
    tuple of (list, list)
        (mean_per_channel, stdv_per_channel) — each a list of three floats
        in RGB channel order.
    """
    mean_per_channel = [0.55709905, 0.38253729, 0.23618910]
    stdv_per_channel = [0.18622870, 0.15077082, 0.12357164]
    return mean_per_channel, stdv_per_channel


def str_to_ms(time: str, sep: str = ":") -> int:
    """Convert a time string to milliseconds.

    Parameters
    ----------
    time : str
        Time in "HH:MM:SS" format.
    sep : str, optional
        Separator character. Default is ':'.

    Returns
    -------
    int or float
        Time in milliseconds, or numpy.nan if the input string is empty.
    """
    if time != "":
        h, m, s = [int(i) for i in time.split(sep)]
        return (((h * 60) + m) * 60 + s) * 1000
    else:
        return np.nan
