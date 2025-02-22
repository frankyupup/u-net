# -*-coding:utf-8 -*-

import os
from PIL import Image
import os.path as osp
from tqdm import tqdm
from model_data_inference.r2unet import R2UNET_Inference
from utils.utils_metrics import compute_mIoU, show_results
from config import *

MODEL_PATH = "runs/R2Unet/r2unet_model.pth"
LABELS_SAVE_PATH = "../results/r2unet_Results"
RESULTS_SAVE_PATH = "test_results/r2unet"
# 模型验证，三个参数，分别是测试集路径，结果保存路径和实际的标签路径
def val_main(test_dir=osp.join(DATASET_PATH, "Test_Images"), result_dir=LABELS_SAVE_PATH,
             gt_dir=osp.join(DATASET_PATH, "Test_Labels"), model_path=MODEL_PATH, image_suffix = IMAGE_SUFFIX, label_suffix=LABEL_SUFFIX, result_save_path=RESULTS_SAVE_PATH, name_classes=NAME_CLASSES):
    num_classes = len(name_classes)                                             # 数据集的类别数目
    image_ids = [x.split(".")[0] for x in os.listdir(test_dir)]                 # 数据集图片名称
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    unet = R2UNET_Inference(model_path=model_path)
    print("\n{}模型加载完毕！模型的详细网络结构如下：".format(model_path))
    print(unet.net)
    print("\n开始对测试集图像进行推理")
    for image_id in tqdm(image_ids):
        image_path = os.path.join(test_dir, image_id + "." + IMAGE_SUFFIX)
        image = Image.open(image_path)
        image = unet.get_miou_png(image)
        image.save(os.path.join(result_dir, image_id + "." + LABEL_SUFFIX))
    print("\n预测完毕！预测结果保存在{}".format(result_dir))
    print("-------------------------------------------------------------------------------------")
    print("开始计算验证指标......")
    hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, result_dir, image_ids, num_classes,
                                                    name_classes, image_suffix, label_suffix)  # 执行计算mIoU的函数
    print(f"指标计算完毕，验证结果已保存在{result_save_path}！")
    show_results(result_save_path, hist, IoUs, PA_Recall, Precision, name_classes, class_names=name_classes)


if __name__ == "__main__":
    val_main()
