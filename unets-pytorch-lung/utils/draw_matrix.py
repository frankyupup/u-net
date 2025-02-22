# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import confusion_matrix
#
# # 假设y_true和y_pred是你的真实标签和预测标签
# y_true = [2, 0, 2, 2, 0, 1]
# y_pred = [0, 0, 2, 2, 0, 2]
#
# # 计算混淆矩阵
# cm = confusion_matrix(y_true, y_pred)
#
# # 绘制混淆矩阵
# plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
# plt.title('Confusion Matrix')
# plt.colorbar()
# tick_marks = np.arange(len(np.unique(y_true)))
# plt.xticks(tick_marks, np.unique(y_true), rotation=45)
# plt.yticks(tick_marks, np.unique(y_true))
#
# # 循环遍历混淆矩阵的每个元素，并添加文本标签
# thresh = cm.max() / 2.
# for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#     plt.text(j, i, format(cm[i, j], 'd'),
#              horizontalalignment="center",
#              color="white" if cm[i, j] > thresh else "black")
#
# # 显示图表
# plt.tight_layout()
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# plt.show()