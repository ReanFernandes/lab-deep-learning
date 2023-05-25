"""
Setup global paths for the project from environment variables.
"""
import os

# CV_PATH_VOC = os.environ.get(
#     "CV_PATH_VOC", "/project/dl2023s/lmbcvtst/public/data/VOC2012")
# CV_PATH_CKPT = os.environ.get(
#     "CV_PATH_CKPT", "/project/dl2023s/lmbcvtst/public/ckpt")

# uncomment below for local dataset and model
CV_PATH_VOC = "data/VOCdevkit/VOC2012"
CV_PATH_CKPT = "ckpt"
