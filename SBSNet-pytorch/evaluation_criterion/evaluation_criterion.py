import os

import cv2
import numpy as np


# 交并比
def IoU(TP, TN, FP, FN):
    return TP / (TP + FP + FN)


# 平均交并比
def mIoU(TP, TN, FP, FN):
    return (TP / (TP + FP + FN) + TN / (TN + FN + FP)) / 2


# 像素准确率
def PA(TP, TN, FP, FN):
    return (TP + TN) / (TP + TN + FP + FN)


class evaluation:
    def __init__(self, img1, img2):
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.inputImage1 = img1
        self.inputImage2 = img2
        if self.inputImage1.shape != self.inputImage2.shape:
            self.inputImage1 = cv2.resize(self.inputImage1, (512, 512))
            self.inputImage2 = cv2.resize(self.inputImage2, (512, 512))
        for across in range(len(self.inputImage1)):
            for endlong in range(len(self.inputImage1[0])):
                if self.inputImage1[across][endlong] and self.inputImage2[across][endlong]:
                    self.TP += 1
                if self.inputImage1[across][endlong] == 0 or self.inputImage2[across][endlong] == 0:
                    self.TN += 1
                if self.inputImage1[across][endlong] and self.inputImage2[across][endlong] == 0:
                    self.FP += 1
                if self.inputImage1[across][endlong] == 0 and self.inputImage2[across][endlong]:
                    self.FN += 1

    def out(self, eva_text):
        if eva_text == 'iou' or eva_text == 1:
            return IoU(self.TP, self.TN, self.FP, self.FN)
        if eva_text == 'm_iou' or eva_text == 2:
            return mIoU(self.TP, self.TN, self.FP, self.FN)
        if eva_text == 'pa' or eva_text == 3:
            return PA(self.TP, self.TN, self.FP, self.FN)


if __name__ == '__main__':
    img1 = np.array(
        [[0, 1, 0],
         [1, 0, 0],
         [0, 1, 1]]
    )
    img2 = np.array(
        [[1, 1, 0],
         [1, 0, 1],
         [0, 1, 0]]
    )
    eva = evaluation(img1, img2)
    print(eva.out(3))
