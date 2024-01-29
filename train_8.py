import os
import sys
import argparse
import logging
import itertools

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from torch.utils.tensorboard import SummaryWriter
from MobileNetv2 import MobileNet_v2
import torch
from transform_processing import transform
import torch.optim as optim
from dataset import MyDataset
from tqdm import tqdm
import math
import random
import warnings
import json
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler as lr_scheduler
from metrics import Metric
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import data_set_split, adjust_learning_rate, train_one_epoch, evaluate, plot_confusion_matrix
from matplotlib import font_manager#意思是导入一个字体管理器
font=font_manager.FontProperties(fname=r'C:\Windows\Fonts\STSONG.TTF')

#def plot(train_loss):
    #plt.figure(figsize=(5,5))#这边定义的是画布的尺寸
    #plt.plot(train_loss,label='train_loss',alpha=0.5)#label的意思是添加一个标签，会在图中显示，alpha是对比度的意思
    #plt.title('使用非线性函数和图像增强训练数据集',fontproperties=font)
    #plt.xlabel('测试/训练轮数',fontproperties=font)
    #plt.ylabel('损失',fontproperties=font)
    #plt.legend()#这一步必须有，没有不显示损失图像了
    #plt.show()
#def plot(train_accuracy):
    #plt.figure(figsize=(5,5))#这边定义的是画布的尺寸
    #plt.plot(train_accuracy, label='train_accuracy', alpha=0.5)  # label的意思是添加一个标签，会在图中显示，alpha是对比度的意思
    #plt.title('准确率与轮数图', fontproperties=font)
    #plt.xlabel('训练轮数', fontproperties=font)
    #plt.ylabel('准确率', fontproperties=font)
    #plt.legend()  # 这一步必须有，没有不显示损失图像了
    #plt.show()
def plot_roc(Recall, Precision):
    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], 'k--')  # 绘制对角线，表示随机猜测的情况
    plt.plot(Recall, Precision, label='ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

# os.environ ["CUDA_VISIBLE_DEVICES"] = '6'
# 参考 https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/examples/imagenet/main.py
parser = argparse.ArgumentParser(description='Training with pytorch')
# 需要改动的参数
parser.add_argument("--dataset_type", default='My_data', type=str, help='type of dataset')
parser.add_argument("--root_datasets", default="C://Users//20154//Desktop//彩色数据集扩充二//所有图片汇总",
                    help='Dataset directory path')
parser.add_argument("--target_data_folder", default="C://Users//20154//Desktop//数据集划分//最终数据集划分3",
                    help='target_data_folder directory path')
parser.add_argument("--Save_Confusion_Matrix_folder", default="C://Users//20154//Desktop//Mobilenetv1训练//混淆矩阵",
                    help='Dataset directory path')
parser.add_argument("--Save_weights_folder", default="C://Users//20154//Desktop//Mobilenetv1训练//权重",
                    help='target_data_folder directory path')
parser.add_argument("--json_root", default="class_indices.json", help='Save_Confusion_Matrix_folder directory path')
parser.add_argument('--net', default="MobileNetV2",
                    help="The network architecture, it can be MobileNetV1, MobileNetV2,or MobileNetV3.")
parser.add_argument('--batch_size', default=16, type=int, help='Batch size for training')
parser.add_argument('--num_classes', default=5, type=int, help='class_num for training')
parser.add_argument('--theta', default=1, type=int, help='theta for training')
parser.add_argument('--num_epochs', default=2, type=int, help='the number epochs')
# 不需要改动的参数
parser.add_argument('--balance_data', action='store_true',
                    help="Balance training data by down-sampling more frequent labels.")
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float)
parser.add_argument('--momentum', default=0.9, type=float, help='Momentum value for optim')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--resume', default=None, type=str, help='Checkpoint state_dict file to resume training from')
parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
parser.add_argument('--use_cuda', default=True, type=bool, help='Use CUDA to train model')
parser.add_argument('--checkpoint_folder', default='1_model_newspaper_hecheng_ToDesk/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--lrf', type=float, default=0.01)
parser.add_argument('--seed', default=10, type=int, help='seed for initializing training. ')
parser.add_argument('--class_label', default=[], type=list, help='seed for initializing training. ')

logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')



def main():
    #train_loss = list()
    #train_accuracy=list()
    Recall=list()
    Precision=list()
    args = parser.parse_args()
    logging.info(args)  # 打印你的args
    best_accuracy = 0.0  # 在 main 函数中定义
    early_stopping_counter = 0  # 在 main 函数中定义
    # 加了这句话，会让种子失效
    if torch.cuda.is_available() and args.use_cuda:
        torch.backends.cudnn.benchmark = True
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

        '''设置 torch.backends.cudnn.benchmark=True 将会让程序在开始时花费一点额外时间，
        为整个网络的每个卷积层搜索最适合它的卷积实现算法，进而实现网络的加速。'''
        logging.info("Use Cuda.")
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")
    tb_writer = SummaryWriter()
    if args.net == 'MobileNetV2':
        create_net = MobileNet_v2(theta=args.theta, num_classes=args.num_classes).to(device)

    train_images_path, train_images_labels, val_images_path, val_images_labels = data_set_split(args.root_datasets,
                                                                                                args.target_data_folder)

    with open(args.json_root, "r") as f:
        class_indict = json.load(f)

    for key, values in class_indict.items():
        args.class_label.append(values)

    train_dataset = MyDataset(image_path=train_images_path, image_cla=train_images_labels, transform=transform["train"])
    val_dataset = MyDataset(image_path=val_images_path, image_cla=val_images_labels, transform=transform["val"])
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=args.num_workers,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=args.num_workers,
                                             collate_fn=val_dataset.collate_fn)
    pg = [p for p in create_net.parameters() if p.requires_grad]
    '''
    调整学习率：是基于在每一个epoch调整，同时学习率的调整要在优化器参数之后更新。
    "即先优化器更新，在学习率更新"
    '''
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=1E-4)
    scheduler_plateau = ReduceLROnPlateau(optimizer, patience=3, verbose=True)  # 这是后加的，表示使用提前停止准则
    lf = adjust_learning_rate(args)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(args.num_epochs):
        # train
        mean_loss = train_one_epoch(model=create_net,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch)
        #test_loss.append(torch.tensor(mean_loss).detach().cpu())  # 后添加的
        #train_loss.append(torch.tensor(mean_loss).detach().cpu())  # 后加的
        scheduler.step()
        # 检查是否达到提前停止准则
        scheduler_plateau.step(mean_loss)#后加的
        # validate
        acc, metric = evaluate(model=create_net,  # 这边是进行混淆矩阵的调用
                               data_loader=val_loader,
                               device=device)
        # 进行准确率与召回率的定义
        # 调用 precision 方法
        precision_score = metric.precision()
        precision_value = precision_score.item()
        print("Precision:", precision_value)#加上.item就会显示出数字
        Precision.append(precision_value)
        #调用recall方法
        recall_score=metric.recall()
        recall_value = recall_score.item()
        print("Recall:", recall_value)
        Recall.append(recall_value)
        #train_accuracy.append(acc)#后加的
        # 更新最佳准确率和提前停止计数器，这些都是后加的
        if acc > best_accuracy:
            best_accuracy = acc
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        # 检查提前停止条件
        if early_stopping_counter >= 10:
            print('Early stopping triggered!')
            break
        f = plot_confusion_matrix(metric.confusion_matrix(), args.num_classes, args.class_label, [5, 5])
        f.savefig(args.Save_Confusion_Matrix_folder + '/{}-{}.png'.format("epoch", epoch))
        # f.clf()
        plt.clf()
        f.clear()  # 释放内存
        plt.close()

        logging.info("[epoch {}] accuracy: {}".format(epoch, round(acc, 5)))
        tags = ["loss", "accuracy", "learning_rate"]
        tb_writer.add_scalar(tags[0], mean_loss, epoch)
        tb_writer.add_scalar(tags[1], acc, epoch)
        tb_writer.add_scalar(tags[2], optimizer.param_groups[0]["lr"], epoch)
        if (epoch + 1) % 5 == 0:
            torch.save(create_net.state_dict(),
                       args.Save_weights_folder + "/model-{}-of-{}-{}-{}.pth".format(epoch + 1, args.num_epochs,
                                                                                     mean_loss, acc))

    #plot(train_accuracy)
    plot_roc(Recall, Precision)
if __name__ == '__main__':
    main()