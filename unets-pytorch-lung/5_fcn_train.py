# -*-coding:utf-8 -*-

import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from nets.unet_training import weights_init
from utils.callbacks import LossHistory
from utils.dataloader_medical import UnetDataset, unet_dataset_collate
from utils.utils_fit import fit_one_epoch_no_val
import os.path as osp
from nets.fcn import get_fcn8s
from config import *

# 超参数调整区域                             # 类别数
MODEL_NAME = "FCN"                         # 模型名字
SAVE_PATH = osp.join("runs", MODEL_NAME)   # 模型保存路径


def train_main(dataset_path=DATASET_PATH, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LEARNING_RATE,
               save_path=SAVE_PATH, num_classes=NUM_CLASSES):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 本机使用的设备，如果没有GPU将会自动调整为CPU 无需调整
    input_shape = [512, 512]                                                # 图片输入大小
    dice_loss = False                                                       # 医学数据集设置为True可以一定精度，如果种类较多尽量不要使用
    focal_loss = False                                                      # 设置为True可解决正负样本不均衡的问题
    cls_weights = np.ones([num_classes], np.float32)                 # 初始化模型训练的损失函数权重
    num_workers = 0
    # window下设置为0即可
    model = get_fcn8s(2)             # 设置模型输出的通道数目和模型输出的类别数目
    weights_init(model)                                                    # 模型初始化，使用kaiming模型初始化
    model_train = model.train()                                            # 将模型设置为训练模式
    model_train = model_train.to(device)                                   # 将模型转移到你的设备上面
    loss_history = LossHistory(save_path, val_loss_flag=False)             # 模型保存路径
    # 获取训练数据集，数据集初始化
    train_lines = [x.split(".")[0] for x in os.listdir(osp.join(dataset_path, "Training_Images"))]
    train_dataset = UnetDataset(train_lines, input_shape, num_classes, True, dataset_path,  IMAGE_SUFFIX, LABEL_SUFFIX)
    gen = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                     drop_last=True, collate_fn=unet_dataset_collate)
    epoch_step = len(train_lines) // batch_size
    # 获取测试数据集
    if epoch_step == 0:
        raise ValueError("无法加载数据集，请检查：\n1.是否是数据集过小。\n2.你的原图和标签数量和名称是否一致")
    optimizer = optim.Adam(model_train.parameters(), lr)                           # 设置模型的优化器
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)   # 设置学习率的优化器

    for epoch in range(epochs):
        # 不需要经过验证的训练
        fit_one_epoch_no_val(model_train, model, loss_history, optimizer, epoch, epoch_step, gen, epochs, device,
                             dice_loss, focal_loss, cls_weights, num_classes, save_path=save_path, model_save_gap=10)
        # 训练的过程中同时验证， 加入测试数据集会引来配置的麻烦，暂时只训练
        # fit_one_epoch(model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val,
        #               Epoch,
        #               cuda, dice_loss, focal_loss, cls_weights, num_classes)
        lr_scheduler.step()

if __name__ == '__main__':
    # 启动训练过程，模型将会保存在runs目录下
    train_main()

