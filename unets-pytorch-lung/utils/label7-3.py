#import os, sys
#cur_path = r'dataset/heti/labels/'  # 你的数据集路径
#labels = os.listdir(cur_path)
#for label in labels:
#    old_label = str(label)
#    new_label = label.replace('_cutout.png', '.png')
#    print(old_label, new_label)
#    os.rename(os.path.join(cur_path, old_label), os.path.join(cur_path, new_label))

import os
import random
import shutil

# 数据集路径 polyps
dataset_path = 'G:/AAA-projects/ING/4-april/800-wei/Kvasir_split_dataset'
images_path = 'G:/AAA-projects/ING/4-april/800-wei/Kvasir-SEG/images/'
labels_path = 'G:/AAA-projects/ING/4-april/800-wei/Kvasir-SEG/masks/'

images_name = os.listdir(images_path)
images_num = len(images_name)
alpha = int(images_num * 0.7)
print(images_num)

random.shuffle(images_name)
random.shuffle(images_name)
train_list = images_name[0:alpha]
#valid_list = images_name[0:alpha1]
valid_list = images_name[alpha:]

# 确认分割正确
print('train list: ', len(train_list))
print('valid list: ', len(valid_list))

# 创建train,valid和test的文件夹
# 'Training_Images',
#             ann_dir='Training_Labels'
#         img_dir='Test_Images',
#         ann_dir='Test_Labels',
train_images_path = os.path.join(dataset_path, 'Training_Images')
train_labels_path = os.path.join(dataset_path, 'Training_Labels')
if os.path.exists(train_images_path) == False:
    os.mkdir(train_images_path)
if os.path.exists(train_labels_path) == False:
    os.mkdir(train_labels_path)

valid_images_path = os.path.join(dataset_path, 'Test_Images')
valid_labels_path = os.path.join(dataset_path, 'Test_Labels')
if os.path.exists(valid_images_path) == False:
    os.mkdir(valid_images_path)
if os.path.exists(valid_labels_path) == False:
    os.mkdir(valid_labels_path)

# 拷贝影像到指定目录
for image in train_list:
    shutil.copy(os.path.join(images_path, image), os.path.join(train_images_path, image))
    # shutil.copy(os.path.join(labels_path, image).replace("jpg", "png"), os.path.join(train_labels_path, image).replace("jpg", "png"))
    shutil.copy(os.path.join(labels_path, image), os.path.join(train_labels_path, image))

for image in valid_list:
    shutil.copy(os.path.join(images_path, image), os.path.join(valid_images_path, image))
    # shutil.copy(os.path.join(labels_path, image).replace("jpg", "png"), os.path.join(valid_labels_path, image).replace("jpg", "png"))
    shutil.copy(os.path.join(labels_path, image), os.path.join(valid_labels_path, image))
