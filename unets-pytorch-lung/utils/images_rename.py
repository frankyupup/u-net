import os
import os.path as osp

src_folder = r"G:\Upppppdate\26\unet\unet_resnet\ori_data\TestLabels"
image_names = os.listdir(src_folder)
for image_name in image_names:
    src_path = osp.join(src_folder, image_name)
    target_path = osp.join(src_folder, image_name.replace("gt_", ""))
    os.rename(src_path, target_path)

