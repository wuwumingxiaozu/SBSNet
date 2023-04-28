# SBSNet
SBSNet: Selective branch segmentation network of coal flotation froth images论文代码

## datasets文件夹
这个文件夹下放的是数据集，泡沫图像分割使用labelimage或者labelme打标签是很麻烦的，所以，这里直接使用photoshop打的标签。
psd文件一共为三个图层。背景图层即原始图像；第二层为预处理的图层，这里也可以不要，目的是因为图像太黑了，使其看的更加清晰一点；第三个图层使用红色的画笔，直接沿着边缘进行描绘。
将打完标签后的图像直接放入到PsdImage下，运行PsdToVOC，就将数据集转换完成了。

## evaluation_criterion文件夹
判定标准，目前仅定义了IoU，mIoU，PA三种。因为分割图像的对象单一。

## Iogs文件夹
存放日志文件，查看日志文件的方法是，进入该项目内，使用tensorboard --logdir logs。

## net文件夹
存放SBSNet模型的代码，论文中提到的其他模型代码也可以直接放到这里进行调用。

## params文件夹
这里存放训练权重。

## result文件夹
训练完成的模型，需要验证其最终效果，将需验证的泡沫图像放到datasets/InferenceImages文件下。

## train_image文件夹
该目录存放训练过程中的图像，从这里可以查看训练的效果好坏。

## MyDataset.py
定义data类型

## TrainAndInference.py
用于训练和验证模型，所有的超参数均在这里调整。

## until.py
调整数据集图像大小的。里面需要设置大小在TrainAndInference.py中修改即可。
