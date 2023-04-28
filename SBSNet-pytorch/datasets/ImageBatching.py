#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import cv2
import numpy as np

'''
作用：
    对图像进行处理
属性：
    operation:默认为medians中值滤波，可以设置为blur均值滤波
    filter:滤波器的大小
'''


def batching(img_read, operation='滤波', *args):
    img_result = None
    # batching(img_read, operation='滤波', *args):
    # img_read：输入的图片
    # operation：对图像操作的类型
    # arg[0]：对图像的具体操作
    # arg[1]arg[2]：滤波器的大小
    if operation == '滤波':
        # 对图像进行均值滤波操作
        if args[0] == 1:
            img_result = cv2.blur(img_read, (args[1], args[2]))
        # 对图像进行中值滤波操作
        elif args[0] == 2:
            img_result = cv2.medianBlur(img_read, args[1])
        # 对图像进行方框滤波操作，当normalize为1或省略是进行归一化处理，当设置为0时不进行归一化处理
        elif args[0] == 3:
            img_result = cv2.boxFilter(img_read, -1, (args[1], args[2]), normalize=args[3])

    # batching(img_read, operation='滤波', *args):
    # img_read：输入的图片
    # operation：对图像操作的类型
    # arg[0]：对图像的具体操作
    # arg[1]arg[2]：滤波器的大小
    elif operation == '梯度':
        # sobel获取图像梯度
        if args[0] == 1:
            sobelx = cv2.Sobel(img_read, cv2.CV_64F, 1, 0)
            sobely = cv2.Sobel(img_read, cv2.CV_64F, 0, 1)
            # 将两个梯度合并
            img_result = cv2.addWeighted(cv2.convertScaleAbs(sobelx), 0.5, cv2.convertScaleAbs(sobely), 0.5, 0)
        # scharr获取图像梯度
        elif args[0] == 2:
            # 计算图像x轴上的梯度，当dst设置为-1时，使用的为scharr算子进行计算
            sobelx = cv2.Sobel(img_read, cv2.CV_64F, 1, 0, dst=-1)
            sobely = cv2.Sobel(img_read, cv2.CV_64F, 0, 1, dst=-1)
            # 将两个梯度合并
            img_result = cv2.addWeighted(cv2.convertScaleAbs(sobelx), 0.5, cv2.convertScaleAbs(sobely), 0.5, 0)
        # 拉普拉斯变化获取图像梯度
        elif args[0] == 3:
            laplacian = cv2.Laplacian(img_read, cv2.CV_64F)
            img_result = cv2.convertScaleAbs(laplacian)
        # Canny边缘检测，args[1]为边缘检测的梯度下限，args[2]为边缘检测梯度的上限
        elif args[0] == 4:
            img_result = cv2.Canny(img_read, args[1], args[2])

    # 图像向上向下取样
    elif operation == '金字塔':
        # 图像向下取样，变为原始图像的1/4
        if args[0] == 1:
            img_result = cv2.pyrDown(img_read)
        # 图像向下取样，变为原始图像的4倍
        elif args[0] == 2:
            img_result = cv2.pyrUp(img_read)
        # 得到拉普拉斯图像
        elif args[0] == 3:
            img_result = img_read - cv2.pyrUp(cv2.pyrDown(img_read))

    elif operation == '图像放缩':
        img_result = cv2.resize(img_read, args)

    elif operation == '去噪':
        batch = None
        k = np.ones((args[1], args[2]), np.uint8)
        # 对图像进行腐蚀操作
        if args[0] == 1:
            img_result = cv2.erode(img_read, k)
        # 对图像进行膨胀操作,args[1]为膨胀次数
        elif args[0] == 2:
            img_result = cv2.dilate(img_read, k, args[1])
        else:
            # 对图像进行闭运算，去除白色图像中的黑点
            if args[0] == 3:
                batch = cv2.MORPH_CLOSE
            # 对图像进行开运算，去除细小的白色线条和白点
            elif args[0] == 4:
                batch = cv2.MORPH_OPEN
            # 对图像进行礼帽操作，得到原始图像的白色噪点和线条
            elif args[0] == 5:
                batch = cv2.MORPH_TOPHAT
            # 对图像进行黑帽操作，得到原始图像的黑色噪点和线条
            elif args[0] == 6:
                batch = cv2.MORPH_BLACKHAT
            img_result = cv2.morphologyEx(img_read, batch, k)

    elif operation == '均衡化':
        # 对灰度图像进行直方图均衡化操作
        if args[0] == 1:
            img_result = cv2.equalizeHist(img_read)
        # 对彩色图像进行直方图均衡化操作
        elif args[0] == 2:
            (b, g, r) = cv2.split(img_read)
            bH = cv2.equalizeHist(b)
            gH = cv2.equalizeHist(g)
            rH = cv2.equalizeHist(r)
            img_result = cv2.merge((bH, gH, rH))

    elif operation == '切割':
        img_result = []
        size = img_read.shape
        for h in range(int(size[0] / args[1])):
            for w in range(int(size[1] / args[0])):
                dstImg = img_read[h * args[1]:(h + 1) * args[1], w * args[0]:(w + 1) * args[0]]
                img_result.append(dstImg)

    elif operation == '拼接':
        m = 0
        imgArray = []
        for i in range(args[2]):
            across = []
            for j in range(args[1]):
                across.append(img_read[m])
                m += 1
            imgArray.append(across)
        rows = len(imgArray)
        cols = len(imgArray[0])
        rowsAvailable = isinstance(imgArray[0], list)
        width = imgArray[0][0].shape[1]
        height = imgArray[0][0].shape[0]
        if rowsAvailable:
            for x in range(0, rows):
                for y in range(0, cols):
                    # & 判断图像与后面那个图像的形状是否一致，若一致则进行等比例放缩；否则，先resize为一致，后进行放缩
                    if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                        imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, args[0], args[0])
                    else:
                        imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                    None, args[0], args[0])
                    if imgArray[x][y].ndim > 2:
                        # & 如果是灰度图，则变成RGB图像（为了弄成一样的图像）
                        if len(imgArray[x][y].shape) == 2:
                            imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
            # & 设置零矩阵
            imageBlank = np.zeros((height, width, 3), np.uint8)
            hor = [imageBlank] * rows

            for x in range(0, rows):
                hor[x] = np.hstack(imgArray[x])
            img_result = np.vstack(hor)
        # & 如果不是一组照片，则仅仅进行放缩 or 灰度转化为RGB
        else:
            for x in range(0, rows):
                if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                    imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, args[0], args[0])
                else:
                    imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, args[0],
                                             args[0])
                if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
            img_result = np.hstack(imgArray)

    elif operation == '颜色空间转换':
        # 将BGR格式转换成RGB格式
        if args[0] == 1:
            img_result = cv2.cvtColor(img_read, cv2.COLOR_BGR2RGB)
        # 将BGR格式转换成灰度图片
        elif args[0] == 2:
            img_result = cv2.cvtColor(img_read, cv2.COLOR_BGR2GRAY)

    #
    elif operation == '分水岭':
        img_result = []
        img_black = []
        img_white = []
        kernel = np.ones((3, 3), np.uint8)
        img_read = cv2.cvtColor(img_read, cv2.COLOR_BGR2RGB)
        img_result.append(img_read)
        # 去除图像中的大块反光，用于黑图
        RemoveLight = cv2.morphologyEx(img_read, cv2.MORPH_TOPHAT, kernel)
        img_black.append(RemoveLight)
        # 对图像进行反色操作，将图像转换为白图
        blackinverse = cv2.bitwise_not(img_read)
        img_white.append(blackinverse)
        # 对黑图进行归一化操作
        blackgray = cv2.cvtColor(RemoveLight, cv2.COLOR_BGR2GRAY)
        img_black.append(blackgray)
        # 对白图进行归一化操作
        whitegray = cv2.cvtColor(blackinverse, cv2.COLOR_BGR2GRAY)
        img_white.append(whitegray)
        # 将图像进行二值化操作
        ret, blackthresh = cv2.threshold(blackgray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        img_black.append(blackthresh)
        ret, whitethresh = cv2.threshold(whitegray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        img_white.append(whitethresh)
        # # 对白图进行开运算，去除细小的白色线条和白点
        # whiteopening = cv2.morphologyEx(whitethresh, cv2.MORPH_OPEN, kernel, iterations=2)
        # img_white.append(whiteopening)
        # 对图像进行膨胀操作
        blackdilate = cv2.dilate(blackthresh, kernel, iterations=2)
        img_black.append(blackdilate)
        whitedilate = cv2.dilate(whitethresh, kernel, iterations=2)
        img_white.append(whitedilate)
        # 距离背景越远，亮度越高
        blackdist_transform = cv2.distanceTransform(blackthresh, cv2.DIST_L2, 3)
        img_black.append(blackdist_transform)
        whitedist_transform = cv2.distanceTransform(whitethresh, cv2.DIST_L2, 3)
        img_white.append(whitedist_transform)
        # 将图像中大于图像中灰度值最大的点的n倍的所有像素点设置为白色，其他像素点设置为黑色
        ret, blackthreshold = cv2.threshold(blackdist_transform, 0.4 * blackdist_transform.max(), 255, 0)
        img_black.append(blackthreshold)
        ret, whitethreshold = cv2.threshold(whitedist_transform, 0.4 * whitedist_transform.max(), 255, 0)
        img_white.append(whitethreshold)
        # 获得未知区域
        blackthreshold = np.uint8(blackthreshold)
        blackunknown = cv2.subtract(blackdilate, blackthreshold)
        img_black.append(blackunknown)
        whitethreshold = np.uint8(whitethreshold)
        whiteunknown = cv2.subtract(whitedilate, whitethreshold)
        img_white.append(whiteunknown)
        # 设置栅栏
        ret, blackmarkers1 = cv2.connectedComponents(blackthreshold)
        img_black.append(np.abs(blackmarkers1))
        ret, whitemarkers1 = cv2.connectedComponents(whitethreshold)
        img_white.append(np.abs(whitemarkers1))
        # 将所有的设置为前景的区域全部+1
        blackmarkers = blackmarkers1 + 1
        whitemarkers = whitemarkers1 + 1
        # 将所有认为是前景却未设定为前景的区域设置为0
        blackmarkers[blackunknown == 255] = 0
        img_black.append(np.abs(blackmarkers))
        whitemarkers[whiteunknown == 255] = 0
        img_white.append(np.abs(whitemarkers))
        # 使用分水岭对图像进行分割，获得边缘
        num = 1
        blackmarkers2 = cv2.watershed(img_read, blackmarkers)
        blackmarkers2 = blackmarkers2[num:blackmarkers2.shape[0] - num, num:blackmarkers2.shape[1] - num]
        blackmarkers2 = np.pad(blackmarkers2, ((num, num), (num, num)), 'constant')
        img_black.append(np.abs(blackmarkers2))
        whitemarkers2 = cv2.watershed(img_read, whitemarkers)
        whitemarkers2 = whitemarkers2[num:whitemarkers2.shape[0] - num, num:whitemarkers2.shape[1] - num]
        whitemarkers2 = np.pad(whitemarkers2, ((num, num), (num, num)), 'constant')
        img_white.append(np.abs(whitemarkers2))
        if args:
            args[0][blackmarkers2 == -1] = [0, 0, 255]
            args[0][whitemarkers2 == -1] = [0, 0, 255]
            imgresult = args[0]
        else:
            # 将边缘在原始图像中全部置为红色
            img_read[blackmarkers2 == -1] = [0, 0, 255]
            img_read[whitemarkers2 == -1] = [0, 0, 255]
            imgresult = img_read
        # print(type(img_read))
        img_result.append(imgresult)
        zeroimg = blackmarkers2.copy()
        zeroimg[blackmarkers2 != -1] = 0
        zeroimg[whitemarkers2 == -1] = -1
        img_result.append(zeroimg)
        img_result = img_result + img_black + img_white

    elif operation == '显示':
        cv2.imshow("dstImg", img_read)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img_result


'''
    作用：将对图像的所有操作全部集合起来
'''


def batchaggregate(image):
    # print(image.shape)
    # img1 = batching(image, '切割', image.shape[1] / 8, image.shape[0] / 8)
    # img2 = batching(image, '切割', image.shape[1] / 4, image.shape[0] / 4)
    # imggather1 = []
    # imggather2 = []
    # for img in img1:
    #     img3 = batching(img, '均衡化', 2)
    #     img = batching(img3, '分水岭', img)
    #     imggather1.append(img[2])
    # for img in img2:
    #     img3 = batching(img, '均衡化', 2)
    #     img = batching(img3, '分水岭', img)
    #     imggather2.append(img[2])
    # imgedge1 = batching(imggather1, '拼接', 1, 8, 8)
    # imgedge2 = batching(imggather2, '拼接', 1, 4, 4)
    # imgedge = imgedge1 - imgedge2
    # image[imgedge == -1] = [0, 0, 255]

    image = batching(image, '切割', 512, 512)
    # image = batching(image, '均衡化', 2)
    return image


'''
    作用：
        保存被处理完的图像，并且输出被保存图像的位置信息
'''


def saveImage(imgsave, path):
    cv2.imencode('.png', imgsave)[1].tofile(path)
    print(path)


class ImageBatching:
    '''
    作用：
        对图像进行批量处理·
    属性：
        sourceGieName:处理完的图像保存在该文件夹下
        root_path:字符串类型，要处理的图像的根目录
        path_list:图像所在路径保存到该列表下
        Img_read:读取图片
        Img_result:保存原图片的样本
        times:定义当前导出图片为第几张
    '''

    def __init__(self, root_path='Image'):
        self.sourceFieName = 'result'
        self.root_path = root_path
        self.path_list = []
        self.Img_read = None
        self.Img_result = None
        for temp in os.walk(self.root_path):
            temp = list(temp)
            # 判断该文件夹下是否存在可执行文件
            if len(temp[2]) != 0:
                imgpath_list = []
                for name in temp[2]:
                    # 将获取到的可执行文件名与文件路径拼合在一起加入到imgpath_list列表中
                    imgpath_list.append(os.path.join(temp[0], name))
                temp.append(imgpath_list)
                self.path_list.append(temp)
        print(self.path_list)

    '''
    作用：
        对图像进行打开和处理
    '''

    def __call__(self, savepath=None, *args, **kwargs):
        if args is None:
            # 查看输出结果所需要的文件夹是否存在，如果不存在，则创建相应文件夹
            for path in self.path_list:
                if not os.path.exists(path[0] + '/' + self.sourceFieName):
                    # 如果文件目录不存在则创建目录
                    os.makedirs(path[0] + '/' + self.sourceFieName)
        times1 = 0
        # 读取图像，遍历Imgpath_list的数量确定拥有可执行文件的文件夹个数
        for imgpath in self.path_list:
            times2 = 0
            # 遍历文件夹下的每一个可执行文件的名称
            for imgname_path in imgpath[3]:
                self.Img_read = cv2.imread(imgname_path)
                # 如果图片为空，返回错误信息，并终止程序
                if self.Img_read is None:
                    print("图片打开失败！")
                    break
                else:
                    # 对图像进行操作
                    self.Img_result = batchaggregate(self.Img_read)
                    if savepath is not None:
                        if not os.path.exists(savepath):
                            # 如果文件目录不存在则创建目录
                            os.makedirs(savepath)
                        pathname = savepath + '/' + str(times1)
                        times1 += 1
                    else:
                        if not os.path.exists(imgpath[0] + '/' + self.sourceFieName):
                            # 如果文件目录不存在则创建目录
                            os.makedirs(imgpath[0] + '/' + self.sourceFieName)
                        pathname = imgpath[0] + '/' + self.sourceFieName + '/' + str(times2)
                        times2 += 1
                    if isinstance(self.Img_result, list):
                        m = 0
                        for img in self.Img_result:
                            saveImage(img, pathname + '_' + str(m) + '.png')
                            m += 1
                    else:
                        # 对操作后的图像进行保存
                        saveImage(self.Img_result, pathname + '.png')


if __name__ == '__main__':
    img = ImageBatching('../Imageresult')
    # print(img.__doc__)
    img('../Imageresultresult')
