import tqdm
from torch import optim
from torch.utils.data import DataLoader
from MyDataset import *
from evaluation_criterion.evaluation_criterion import evaluation
from net.unet.unet import *
from net.segnet.segnet import *
from net.res_unet.res_unet import *
from net.sbsnet.SBSNet import *
from net.ciscnet.ciscnet import *
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

# ------------------------------------------------------------ #
#   date：                   获取当前日期
# ------------------------------------------------------------ #
date = '20230302'

# ------------------------------------------------------------ #
#   operation：              用于决定操作，是训练还是测试
#       'train':                训练
#       'inference':            推断最终结果
# ------------------------------------------------------------ #
operation = 'inference'

# ------------------------------------------------------------ #
# 判断是否能使用gpu
# ------------------------------------------------------------ #
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------------------------------------------------------------ #
#   data_path:              加载训练数据的地址
# ------------------------------------------------------------ #
data_path = r'datasets'

# ------------------------------------------------------------ #
#   model:            选择使用的网络模型
#       'unet'          unet模型
#       'ciscnet'       ciscnet模型
#       'segnet'        segnet模型
#       'resunet'       残差网络unet模型
#       'sbsnet'        选择分支分割网络sbsnet
# ------------------------------------------------------------ #
model = 'sbsnet'

# ------------------------------------------------------------ #
#   weight_path:            训练权重地址
# ------------------------------------------------------------ #
weight_path = f'params/{model}-{date}.pth'

# ------------------------------------------------------------ #
#   分类为几种
# ------------------------------------------------------------ #
num_classes = 1 + 1  # +1是背景也为一类

# ------------------------------------------------------------ #
#   batch size:(学习率)      一次训练所选取的样本数
#   epoch:(训练轮次)         训练的次数
#   size:(训练图像大小）      默认为1024
# ------------------------------------------------------------ #
batch_size = 2
epochs = 400
image_size = (512, 512)

# ------------------------------------------------------------ #
# optimizer_type:(优化器）
#   'adam':                 自适应时刻估计方法
#   'SGD':                  随机梯度下降，需指定学习率
# learning_rate：(学习率)
#   'learning_rate':        adam默认为1e-3
# ------------------------------------------------------------ #
optimizer_type = 'adam'
learning_rate = 1e-3

# ------------------------------------------------------------ #
#   loss:(损失函数)
#       bceloss:(BCELoss)                二分类交叉熵损失
#       bce_with_logits_loss:(BCEWithLogitsLoss)   sigmoid与bce结合
#       cross_entropy_loss:(CrossEntropyLoss)
# ------------------------------------------------------------ #
loss = 'cross_entropy_loss'

# ------------------------------------------------------------ #
#   save_path:                  保存训练效果图像地址
# ------------------------------------------------------------ #
save_path = 'train_image'

if __name__ == '__main__':

    # 创建网络
    if model == 'unet':
        net = UNet(num_classes).to(device)
    elif model == 'segnet':
        net = SegNet(num_classes).to(device)
    elif model == 'ciscnet':
        net = CiscNet(num_classes).to(device)
    elif model == 'resunet':
        net = RES_UNet(num_classes).to(device)
    elif model == 'sbsnet':
        net = SBSNet(num_classes).to(device)

    # 判断训练权重是否存在，如果存在就加载
    if os.path.exists(weight_path):
        # 加载训练权重
        net.load_state_dict(torch.load(weight_path))
        print('成功加载训练权重！！！')
    else:
        print('未加载训练权重！！！')

    if operation == 'train':

        # ------------------------------------------------------------ #
        #   write:                  tensorboard日志文件
        # ------------------------------------------------------------ #
        write = SummaryWriter('logs')

        # 获取训练样本
        data_loader = DataLoader(MyDataset(data_path, imagesize=image_size), batch_size=batch_size, shuffle=True,
                                 drop_last=True)

        # 创建优化器
        if optimizer_type == 'adam':
            opt = optim.Adam(net.parameters(), lr=learning_rate)
        elif optimizer_type == 'SGD':
            # 学习率为0.01
            opt = optim.SGD(net.parameters(), lr=learning_rate)

        # 二元交叉熵损失函数
        if loss == 'bce_loss':
            loss_fun = nn.BCELoss().to(device)
        elif loss == 'cross_entropy_loss':
            loss_fun = nn.CrossEntropyLoss().to(device)
        elif loss == 'bce_with_logits_loss':
            loss_fun = nn.BCEWithLogitsLoss().to(device)

        # 循环进行训练
        for epoch in range(epochs):
            m_trainloss = 0
            m_iou = 0
            m_miou = 0
            m_pa = 0
            m = 0
            for i, (image, segment_image) in enumerate(data_loader):
                # 获取训练图像与
                image, segment_image = image.to(device), segment_image.to(device)

                # 创建网络结构
                out_image = net(image)

                # 检查损失
                train_loss = loss_fun(out_image, segment_image.long())

                # 优化器清零
                opt.zero_grad()

                # 反向传播
                train_loss.backward()

                # 更新梯度
                opt.step()

                # 获取标签图像
                _segment_image = torch.unsqueeze(segment_image[0], 0) * 255
                # 获取训练输出图像
                _out_image = torch.argmax(out_image[0], dim=0).unsqueeze(0) * 255

                # 将标签图像与训练输出图像合成一个图像
                img = torch.stack([_segment_image, _out_image], dim=0)
                # 保存训练对比图像
                save_image(img, f'{save_path}/{model}-{i}.png')
                if i % 1 == 0:
                    m_trainloss = m_trainloss + train_loss.item()
                    eva = evaluation(_segment_image[0].clone().detach().cpu().numpy(),
                                     _out_image[0].clone().detach().cpu().numpy())
                    print(
                        f'epoch:{epoch}-{i}-train_loss=>>{train_loss.item()}-IoU=>>{eva.out(1) * 100}%-mIoU=>>{eva.out(2) * 100}%-PA=>>{eva.out(3) * 100}%')
                    m_iou = m_iou + eva.out(1)
                    m_miou = m_miou + eva.out(2)
                    m_pa = m_pa + eva.out(3)
                    m = m + 1

            write.add_scalar(f'{model}_train_loss-{date}', m_trainloss / m, epoch)
            write.add_scalar(f'{model}_IoU-{date}', m_iou / m, epoch)
            write.add_scalar(f'{model}_mIoU-{date}', m_miou / m, epoch)
            write.add_scalar(f'{model}_PA-{date}', m_pa / m, epoch)
            # 保存训练权重
            if (epoch + 1) % 10 == 0:
                torch.save(net.state_dict(), weight_path)
                print('保存权重成功!')
        # 写日志文件关闭
        write.close()

    elif operation == 'inference':
        for name in os.listdir(os.path.join(data_path, 'InferenceImages')):
            img = keep_image_size_open_rgb(os.path.join(data_path, 'InferenceImages', name), size=image_size)
            img_data = transform(img).to(device)
            img_data = torch.unsqueeze(img_data, dim=0)
            net.eval()
            with torch.no_grad():
                out = net(img_data)
                out = torch.argmax(out, dim=1)
                out = torch.squeeze(out, dim=0)
                out = out.unsqueeze(dim=0)
                out = out.permute((1, 2, 0)).cpu().detach().numpy()
                cv2.imencode('.png', out * 255.0)[1].tofile(os.path.join('result', model + 'z-' + name))
