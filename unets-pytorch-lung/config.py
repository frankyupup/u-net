#!/usr/bin/env python
# -*- coding: UTF-8 -*-

DATASET_PATH = "../lung_split_dataset"    # 数据集的路径，可以是绝对路径, 也可以是相对路径
EPOCHS = 10                               # 模型训练的轮数
BATCH_SIZE = 2                            # 图像训练的批次大小
LEARNING_RATE = 1e-4                      # 学习率
NUM_CLASSES = 2                           # 类别数量
NAME_CLASSES = ["background", "lung"]     # 类别名称
IMAGE_SUFFIX = "tif"                      # 图像后缀
LABEL_SUFFIX = "tif"                      # 标签后缀
