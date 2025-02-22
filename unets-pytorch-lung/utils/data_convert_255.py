# 数据集转换
# 将原先得255数据集转换为新得01数据集，保证代码测试
import os
import os.path as osp
import cv2
def data_convert(src_folder, target_folder):
    if osp.isdir(target_folder) == False:
        os.mkdir(target_folder)
    image_names = os.listdir(src_folder)
    for image_name in image_names:
        image_path = osp.join(src_folder, image_name)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img[img>0] = 255
        save_path = osp.join(target_folder, image_name)
        cv2.imwrite(save_path, img)
# H:\Upppppdate\322222\qq_3045834499\unets-pytorch-skin
if __name__ == '__main__':
    data_convert(src_folder=r"H:\Upppppdate\322222\qq_3045834499\heart_split_data\Training_Labels_src", target_folder=r"H:\Upppppdate\322222\qq_3045834499\heart_split_data\Training_Labels_src")
