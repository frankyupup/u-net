# -*-coding:utf-8 -*-
import shutil
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys
import cv2
import torch
import os.path as osp
import uuid
import os
from PIL import Image
from model_data_inference.unet import Unet
from model_data_inference.unetpp import UnetPPX
from model_data_inference.r2unet import R2UNET_Inference
from model_data_inference.attention_unet import AttentionUnet_Inference
from model_data_inference.unet_origin import Unet_Origin_Inference
from model_data_inference.fcn import Fcn
import datetime
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WINDOW_TITLE = '二分类医学图像分割系统'
LEFT_IMAGE_PATH = "images/UI/doctor.png"
RIGHT_IMAGE_PATH = "images/UI/doctor.png"
ICON_IMAGE = "images/UI/lufei.png"
USERNAME = "123"
PASSWORD = "123"
## 登录结果 web程序 时间排序

# 获取分割区域，以及获取中心点的结果
def get_center(image_path, show_img_path):
    show_img = cv2.imread(show_img_path)
    # show_img = cv2.cvtColor(show_img, cv2.COLOR_BGR2RGB)
    img = cv2.imread(image_path)
    groundtruth = img[:, :, 0]
    h1, w1 = groundtruth.shape
    _, binary_image = cv2.threshold(groundtruth, 127, 255, cv2.THRESH_BINARY)
    contours, cnt = cv2.findContours( binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = []
    for c in range(len(contours)):
        areas.append(cv2.contourArea(contours[c]))
    max_id = areas.index(max(areas))
    X_center = []
    Y_center = []
    for i in range(len(contours)):
        if i == max_id:
            M = cv2.moments(contours[i])  # 计算第一条轮廓的各阶矩,字典形式
            center_x = int(M["m10"] / (M["m00"] + 0.00001))
            center_y = int(M["m01"] / (M["m00"] + 0.0001))
            X_center.append(center_x)
            Y_center.append(center_y)
            # print("xxxx")
            cv2.drawContours(show_img, contours, i, (0, 0, 255) , 2)  # 绘制轮廓，填充
            # m_image = cv2.circle(image, (center_x, center_y), 7, 111, -1)  # 绘制中心点
            # m_image = cv2.circle(img, (center_x, center_y), 7, (0, 0, 255), -1)  # 绘制中心点
    return show_img



class MainWindow(QTabWidget):
    # 基本配置不动，然后只动第三个界面
    def __init__(self):
        # 初始化界面
        super().__init__()
        self.setWindowTitle(WINDOW_TITLE)
        self.resize(1200, 800)
        self.setWindowIcon(QIcon(ICON_IMAGE))
        self.output_size = 480
        self.img2predict = ""
        self.origin_shape = ()
        self.is_det = False
        self.is_download = False
        self.model = Unet_Origin_Inference(model_path="runs/original_unet/original_unet_model.pth")  # 基础模型添加的位置
        # model_path = "runs/original_unet/original_unet_model.pth"
        # self.model = Unet_Origin_Inference(model_path=model_path)
        self.initUI()
        self.contour = True
        self.folder = "tmp"


    def initUI(self):
        # 图片检测子界面
        font_title = QFont('楷体', 16)
        font_main = QFont('楷体', 14)
        img_detection_widget = QWidget()
        img_detection_layout = QVBoxLayout()
        img_detection_title = QLabel("基于深度学习的肺部图像分割系统")
        img_detection_title.setFont(font_title)
        # 模型切换功能
        self.img_info_edit1_model = QComboBox()
        self.img_info_edit1_model.addItem("Original_unet")
        self.img_info_edit1_model.addItem("unet_resnet50")
        self.img_info_edit1_model.addItem("unet_vgg16")
        self.img_info_edit1_model.addItem("UnetPP")
        self.img_info_edit1_model.addItem("r2_unet")
        self.img_info_edit1_model.addItem("Attention_unet")
        self.img_info_edit1_model.addItem("fcn")

        self.img_info_edit1_model.currentIndexChanged.connect(self.indexChange)
        self.img_info_edit1_model.setFont(font_main)
        mid_img_widget = QWidget()
        mid_img_layout = QHBoxLayout()
        self.left_img = QLabel()
        self.right_img = QLabel()
        self.left_img.setPixmap(QPixmap(LEFT_IMAGE_PATH))
        self.right_img.setPixmap(QPixmap(RIGHT_IMAGE_PATH))
        self.left_img.setAlignment(Qt.AlignCenter)
        self.right_img.setAlignment(Qt.AlignCenter)
        mid_img_layout.addWidget(self.left_img)
        mid_img_layout.addStretch(0)
        mid_img_layout.addWidget(self.right_img)
        mid_img_widget.setLayout(mid_img_layout)
        up_img_button = QPushButton("上传图片")
        det_img_button = QPushButton("开始检测")
        download_img_button = QPushButton("下载分割结果")
        up_img_button.clicked.connect(self.upload_img)
        det_img_button.clicked.connect(self.detect_img)
        download_img_button.clicked.connect(self.download)
        up_img_button.setFont(font_main)
        det_img_button.setFont(font_main)
        download_img_button.setFont(font_main)
        up_img_button.setStyleSheet("QPushButton{color:white}"
                                    "QPushButton:hover{background-color: rgb(2,110,180);}"
                                    "QPushButton{background-color:rgb(48,124,208)}"
                                    "QPushButton{border:2px}"
                                    "QPushButton{border-radius:5px}"
                                    "QPushButton{padding:5px 5px}"
                                    "QPushButton{margin:5px 5px}")
        det_img_button.setStyleSheet("QPushButton{color:white}"
                                     "QPushButton:hover{background-color: rgb(2,110,180);}"
                                     "QPushButton{background-color:rgb(48,124,208)}"
                                     "QPushButton{border:2px}"
                                     "QPushButton{border-radius:5px}"
                                     "QPushButton{padding:5px 5px}"
                                     "QPushButton{margin:5px 5px}")
        download_img_button.setStyleSheet("QPushButton{color:white}"
                                          "QPushButton:hover{background-color: rgb(2,110,180);}"
                                          "QPushButton{background-color:rgb(48,124,208)}"
                                          "QPushButton{border:2px}"
                                          "QPushButton{border-radius:5px}"
                                          "QPushButton{padding:5px 5px}"
                                          "QPushButton{margin:5px 5px}")
        img_detection_layout.addWidget(img_detection_title, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(self.img_info_edit1_model)
        img_detection_layout.addWidget(mid_img_widget, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(up_img_button)
        img_detection_layout.addWidget(det_img_button)
        img_detection_layout.addWidget(download_img_button)
        img_detection_widget.setLayout(img_detection_layout)
        # 患者信息详情界面
        about_widget = QWidget()
        real_about_layout = QVBoxLayout()
        grid_widget = QWidget()
        about_layout = QGridLayout()
        name = QLabel("患者姓名")
        age = QLabel("年龄")
        time = QLabel("就诊时间")
        suggest = QLabel("诊断意见")
        self.name_edit = QLineEdit()
        self.age_edit = QLineEdit()
        self.time_edit = QDateTimeEdit(QDateTime.currentDateTime(), self)
        self.suggest_edit = QTextEdit()
        about_layout.setSpacing(10)
        about_layout.addWidget(name, 1, 0)
        about_layout.addWidget(self.name_edit, 1, 1)
        about_layout.addWidget(age, 2, 0)
        about_layout.addWidget(self.age_edit, 2, 1)
        about_layout.addWidget(time, 3, 0)
        about_layout.addWidget(self.time_edit, 3, 1)
        about_layout.addWidget(suggest, 4, 0)
        about_layout.addWidget(self.suggest_edit, 4, 1)
        go_button = QPushButton("提交")
        go_button.setFont(font_main)
        go_button.setStyleSheet("QPushButton{color:white}"
                                "QPushButton:hover{background-color: rgb(2,110,180);}"
                                "QPushButton{background-color:rgb(48,124,208)}"
                                "QPushButton{border:2px}"
                                "QPushButton{border-radius:5px}"
                                "QPushButton{padding:5px 5px}"
                                "QPushButton{margin:5px 5px}")
        go_button.clicked.connect(self.go)
        chakan_button = QPushButton("查看病例信息")
        chakan_button.setFont(font_main)
        chakan_button.setStyleSheet("QPushButton{color:white}"
                                    "QPushButton:hover{background-color: rgb(2,110,180);}"
                                    "QPushButton{background-color:rgb(48,124,208)}"
                                    "QPushButton{border:2px}"
                                    "QPushButton{border-radius:5px}"
                                    "QPushButton{padding:5px 5px}"
                                    "QPushButton{margin:5px 5px}")
        chakan_button.clicked.connect(self.chakan)
        about_title = QLabel('诊断信息填写')
        about_title.setFont(QFont('楷体', 18))
        about_title.setAlignment(Qt.AlignCenter)
        grid_widget.setFont(font_main)
        grid_widget.setLayout(about_layout)
        real_about_layout.addWidget(about_title)
        real_about_layout.addWidget(grid_widget)
        real_about_layout.addWidget(go_button)
        real_about_layout.addWidget(chakan_button)
        about_widget.setLayout(real_about_layout)

        self.left_img.setAlignment(Qt.AlignCenter)
        self.addTab(img_detection_widget, 'AI检测')
        self.addTab(about_widget, '报告填写')
        self.setTabIcon(0, QIcon('images/UI/lufei.png'))
        self.setTabIcon(1, QIcon('images/UI/lufei.png'))
        self.setTabPosition(QTabWidget.West)
        # img_detection_widget.setStyleSheet("background-color:rgb(255,250,205);")
        # about_widget.setStyleSheet("background-color:rgb(255,250,205);")

    # 图像上传功能
    def upload_img(self):
        # 选择录像文件进行读取
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.jpg *.png *.tif *.jpeg *.bmp')
        if fileName:
            # suffix = fileName.split(".")[-1]
            # save_path = osp.join("images/tmp", "tmp_upload." + suffix)
            # shutil.copy(fileName, save_path)
            im0 = cv2.imread(fileName)
            resize_scale = self.output_size / im0.shape[0]
            im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
            cv2.imwrite("images/tmp/upload_show_result.jpg", im0)
            self.img2predict = fileName
            self.origin_shape = (im0.shape[1], im0.shape[0])
            self.left_img.setPixmap(QPixmap("images/tmp/upload_show_result.jpg"))
            # todo 上传图片之后右侧的图片重置，
            self.right_img.setPixmap(QPixmap(RIGHT_IMAGE_PATH))
            self.is_det = False
            self.is_download = False
            # todo 绘制轮廓作为基础功能添加到二分类的任务中

    # 模型切换功能
    def indexChange(self):
        print(self.img_info_edit1_model.currentIndex())
        print(self.img_info_edit1_model.currentText())
        current_model = self.img_info_edit1_model.currentText()
        # 需要根据自己的训练结果来配置模型路径
        if current_model == "fcn":
            model_path = "runs/FCN/fcn_model.pth"
            self.model = Fcn(model_path=model_path)
        elif current_model == "unet_resnet50":
            model_path = "runs/unet_resnet50/unet_resnet50_model.pth" # todo 在这里修改模型的位置。
            self.model = Unet(model_path=model_path, backbone="resnet50")
        elif current_model == "unet_vgg16":
            model_path = "runs/unet_vgg16/unet_vgg16_model.pth" # todo 在这里修改模型的位置。
            self.model = Unet(model_path=model_path, backbone="vgg16")
        elif current_model == "UnetPP":
            model_path = "runs/UnetPlusPlus/unetpp_model.pth" # todo 在这里修改模型的位置。
            self.model = UnetPPX(model_path=model_path)
        elif current_model == "Attention_unet":
            model_path = "runs/Attention_Unet/attention_unet_model.pth" # todo 在这里修改模型的位置。
            self.model = AttentionUnet_Inference(model_path=model_path)
        elif current_model == "Original_unet":
            model_path = "runs/original_unet/original_unet_model.pth" # todo 在这里修改模型的位置。
            self.model = Unet_Origin_Inference(model_path=model_path)
        elif current_model == "r2_unet":
            model_path = "runs/R2Unet/r2unet_model.pth" # todo 在这里修改模型的位置。
            self.model = R2UNET_Inference(model_path=model_path)
        QMessageBox.information(self, "切换成功", "{}模型切换成功".format(self.img_info_edit1_model.currentText()))

    # 结果查看功能
    def chakan(self):
        os.startfile(osp.join(os.getcwd(), "record"))

    # 图像检测功能
    def detect_img(self):
        source = self.img2predict  # file/dir/URL/glob, 0 for webcam
        image = Image.open(source)
        result_image = self.model.detect_image_ui(image)
        result_image.save("images/tmp_result.jpg")
        pred = cv2.imread("images/tmp_result.jpg")
        im0 = cv2.resize(pred, self.origin_shape)
        # 去除图像中的椒盐噪声

        cv2.imwrite("images/tmp/single_result.jpg", im0)
        self.is_det = True
        self.right_img.setPixmap(QPixmap("images/tmp/single_result.jpg"))
        # 绘制检测出来的图像轮廓
        if self.contour:
            # 从二进制的图像中提取轮廓，在原始的图像上进行绘制，显示在左侧的结果上面
            new_img = get_center("images/tmp/single_result.jpg", "images/tmp/upload_show_result.jpg")
            cv2.imwrite("images/tmp/single_result_new.jpg", new_img)
            self.left_img.setPixmap(QPixmap("images/tmp/single_result_new.jpg"))


    # 图片和结果下载功能
    def download(self):
        if self.is_det == False:
            QMessageBox.warning(self, "无法下载", "请先进行检测")
        elif self.is_download:
            QMessageBox.information(self, "重复下载", "请勿重复下载结果")
        else:
            src_img_path = "images/tmp/single_result.jpg"
            # uuid_folder_name = str(uuid.uuid4())
            # 获取当前时间
            now = datetime.datetime.now()
            # 格式化为字符串
            time_str = now.strftime('%Y-%m-%d_%H_%M_%S')
            uuid_folder_name = str(time_str)
            os.mkdir("record/" + uuid_folder_name)
            self.folder = uuid_folder_name
            shutil.copy2(src_img_path, "record/" + self.folder)
            shutil.copy2(self.img2predict, "record/" + self.folder)
            QMessageBox.information(self, "下载完成", "图片已下载到{}".format(uuid_folder_name))
            self.is_download = True

    # 患者信息获取功能
    def go(self):
        # 获取信息
        name = str(self.name_edit.text())
        age = str(self.age_edit.text())
        time = str(self.time_edit.text())
        suggest = str(self.suggest_edit.toPlainText())
        if name == "":
            QMessageBox.warning(self, "不能为空", "请填写患者姓名")
        elif age == "":
            QMessageBox.warning(self, "不能为空", "请填写患者年龄")
        elif time == "":
            QMessageBox.warning(self, "不能为空", "请填写就诊时间")
        elif suggest == "":
            QMessageBox.warning(self, "不能为空", "请填写诊断意见")

        if self.folder == "tmp":
            QMessageBox.warning(self, "无法生成", "请先下载结果图片")
        else:
            with open("record/{}/reslt.txt".format(self.folder), "w", encoding="utf-8") as f:
                f.writelines(["姓名：{}\n".format(name), "年龄：{}\n".format(age), "就诊时间：{}\n".format(time),
                              "诊断意见：{}\n".format(suggest)])
                QMessageBox.information(self, "报告已生成", "报告已下载到{}".format(self.folder))

    # 界面关闭功能
    def closeEvent(self, event):
        reply = QMessageBox.question(self,
                                     'quit',
                                     "Are you sure?",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close()
            event.accept()
        else:
            event.ignore()


# 添加登录界面
class LoginWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        font_title = QFont('楷体', 16)
        self.setWindowTitle("识别系统登陆界面\n账号密码均为我qq，需要99调试请添加")
        self.resize(800, 600)
        mid_widget = QWidget()
        window_layout = QFormLayout()
        self.user_name = QLineEdit()
        self.u_password = QLineEdit()
        window_layout.addRow("账 号：", self.user_name)
        window_layout.addRow("密 码：", self.u_password)
        self.user_name.setEchoMode(QLineEdit.Normal)
        self.u_password.setEchoMode(QLineEdit.Password)
        mid_widget.setLayout(window_layout)
        # self.setBa
        # self.setObjectName("MainWindow")
        # self.setStyleSheet("#MainWindow{background-color:rgb(236,99,97)}")

        main_layout = QVBoxLayout()
        a = QLabel("😁😁😁😁😁😁😁😁😁😁😁😁\n欢迎使用基于YOLO11的识别系统\n 账号密码均为我QQ:3045834499"
                   "\n需要99调试请添加")
        a.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(a)
        main_layout.addWidget(mid_widget)

        login_button = QPushButton("立即登陆")
        # reg_button = QPushButton("注册用户")
        # reg_button.clicked.connect(self.reggg)
        login_button.clicked.connect(self.login)

        # main_layout.addWidget(reg_button)
        main_layout.addWidget(login_button)

        self.setLayout(main_layout)

        self.mainWindow = MainWindow()
        self.setFont(font_title)
        # self.regwindow = RegWindow()

    # mainWindow.show()

    def login(self):
        user_name = self.user_name.text()
        pwd = self.u_password.text()
        is_ok = (user_name == USERNAME) and (pwd == PASSWORD)
        # is_ok = is_correct(user_name, pwd)

        print(is_ok)
        if is_ok:
            self.mainWindow.show()
            self.close()
        else:
            QMessageBox.warning(self, "账号密码不匹配", "请输入正确的账号密码")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    # mainWindow = MainWindow()
    mainWindow = LoginWindow()
    mainWindow.show()
    sys.exit(app.exec_())
