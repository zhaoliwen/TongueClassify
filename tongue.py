import os

import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

dataType = 'Tongue'


class TongueDataset:
    def __init__(self, batch_size=16):
        self.batch_size = batch_size
        self.train_dataset_dir = f'./Data/{dataType}/train'
        self.test_dataset_dir = f'./Data/{dataType}/test'
        self.validation_dataset_dir = f'./Data/{dataType}/validation'

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.train_dataset = None
        self.train_dataloader = None
        self.test_dataset = None
        self.test_dataloader = None
        self.validation_dataset = None
        self.validation_dataloader = None

        self.id_to_class = dict()

    def load_train_data(self):
        self.train_dataset = torchvision.datasets.ImageFolder(self.train_dataset_dir, transform=self.transform)
        print(self.train_dataset.classes)
        print(self.train_dataset.class_to_idx)
        print(f'Train dataset size: {len(self.train_dataset)}')

        # Reverse from: label -> id, to: id -> label
        self.id_to_class = dict((val, key) for key, val in self.train_dataset.class_to_idx.items())
        print(self.id_to_class)

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size,
                                                            shuffle=True, **{'pin_memory': True})
        return self.train_dataloader

    def load_test_data(self):
        self.test_dataset = torchvision.datasets.ImageFolder(self.test_dataset_dir, transform=self.transform)
        print(f'Test dataset size: {len(self.test_dataset)}')

        self.test_dataloader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size,
                                                           **{'pin_memory': True})
        return self.test_dataloader

    def load_validation_data(self):
        self.validation_dataset = torchvision.datasets.ImageFolder(self.validation_dataset_dir,
                                                                   transform=self.transform)
        print(f'Validation dataset size: {len(self.validation_dataset)}')

        self.validation_dataloader = torch.utils.data.DataLoader(self.validation_dataset, batch_size=self.batch_size,
                                                                 **{'pin_memory': True})
        return self.validation_dataloader

    def show_sample_images(self):
        images_to_show = 6
        imgs, labels = next(iter(self.train_dataloader))
        plt.figure(figsize=(56, 56))
        for i, (img, label) in enumerate(zip(imgs[:images_to_show], labels[:images_to_show])):
            # permute交换张量维度，把原来在0维的channel移到最后一维
            img = (img.permute(1, 2, 0).numpy() + 1) / 2
            # rows * cols
            plt.subplot(2, 3, i + 1)
            plt.title(self.id_to_class.get(label.item()))
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img)

        # Show all images
        plt.show()


class TongueResnet(torch.nn.Module):
    def __init__(self, image_width=224, image_height=224, num_classifications=15,
                 enable_dropout=False, enable_bn=False):
        super().__init__()

        self.resnet = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        print(self.resnet)

        fc_features = 128
        resnet_features = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Sequential(
            torch.nn.Linear(resnet_features, fc_features),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(fc_features, num_classifications)
        )

    def forward(self, x):
        y = self.resnet(x)
        return y

    def get_name(self):
        return f'{dataType}Resnet50'

    def transfer_learning_mode(self):
        # 冻结卷积基
        for param in self.resnet.parameters():
            param.requires_grad = False
        # 解冻全连接层
        for param in self.resnet.fc.parameters():
            param.requires_grad = True

    def fine_tune_mode(self):
        # 解冻卷积基
        for param in self.resnet.parameters():
            param.requires_grad = True


class ModelTrainer():
    def __init__(self, model, loss_func, optimizer, lr_scheduler=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(self.device)
        self.model = model
        self.model = self.model.to(self.device)
        self.loss_func = loss_func
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

    def train(self, dataloader):
        # 训练模式
        self.model.train()
        # 所有批次累计损失和
        epoch_loss = 0
        # 累计预测正确的样本总数
        epoch_correct = 0

        # 循环一次数据的多个批次
        for x, y in dataloader:
            # non_blocking=True异步传输数据
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            predicted = self.model(x)
            loss = self.loss_func(predicted, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # 记录已经训练了多少个epoch并触发学习速率的衰减
            if self.lr_scheduler:
                self.lr_scheduler.step()

            # 累加
            with torch.no_grad():
                epoch_correct += (predicted.argmax(1) == y).type(torch.float).sum().item()
                epoch_loss += loss.item()

        return (epoch_loss, epoch_correct)

    def test(self, dataloader):
        # 测试模式
        self.model.eval()
        # 所有批次累计损失和
        epoch_loss = 0
        # 累计预测正确的样本总数
        epoch_correct = 0

        # 循环一次数据的多个批次
        # 测试模式，不需要梯度计算、反向传播
        with torch.no_grad():
            for x, y in dataloader:
                # non_blocking=True异步传输数据
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                predicted = self.model(x)
                loss = self.loss_func(predicted, y)

                # 累加
                epoch_correct += (predicted.argmax(1) == y).type(torch.float).sum().item()
                epoch_loss += loss.item()

        return (epoch_loss, epoch_correct)

    def validate(self, dataloader):
        total_val_data_cnt = len(dataloader.dataset)
        num_val_batch = len(dataloader)
        val_loss, val_correct = self.test(dataloader)
        # 所有批次的统计和/批次数量 = 平均损失率
        avg_val_loss = val_loss / num_val_batch
        # 预测正确的样本数/总样本数 = 平均正确率
        avg_val_accuracy = val_correct / total_val_data_cnt

        return (avg_val_loss, avg_val_accuracy)

    def fit(self, train_dataloader, test_dataloader, epoch):
        # 数据集总量
        total_train_data_cnt = len(train_dataloader.dataset)
        # 数据集批次数目
        num_train_batch = len(train_dataloader)
        # 数据集总量
        total_test_data_cnt = len(test_dataloader.dataset)
        # 数据集批次数目
        num_test_batch = len(test_dataloader)

        best_accuracy = 0.0

        # 循环全部数据
        for i in range(epoch):
            # 训练模型
            epoch_train_loss, epoch_train_correct = self.train(train_dataloader)
            # 所有批次的统计和/批次数量 = 平均损失率
            avg_train_loss = epoch_train_loss / num_train_batch
            # 预测正确的样本数/总样本数 = 平均正确率
            avg_train_accuracy = epoch_train_correct / total_train_data_cnt

            # 测试模型
            epoch_test_loss, epoch_test_correct = self.test(test_dataloader)
            # 所有批次的统计和/批次数量 = 平均损失率
            avg_test_loss = epoch_test_loss / num_test_batch
            # 预测争取的样本数/总样本数 = 平均正确率
            avg_test_accuracy = epoch_test_correct / total_test_data_cnt

            msg_template = (
                "Epoch {:2d} - Train accuracy: {:.2f}%, Train loss: {:.6f}; Test accuracy: {:.2f}%, Test loss: {:.6f}")
            print(msg_template.format(i + 1, avg_train_accuracy * 100, avg_train_loss, avg_test_accuracy * 100,
                                      avg_test_loss))

            # CheckPoint
            if avg_test_accuracy > best_accuracy:
                # 保存最佳测试模型
                best_accuracy = avg_test_accuracy
                ckpt_path = f'./{self.model.get_name()}.ckpt'
                self.save_checkpoint(i, ckpt_path)
                print(f'Save model to {ckpt_path}')

    def predict(self, x):
        # Prediction
        prediction = self.model(x.to(self.device))
        # Predicted class value using argmax
        # predicted_class = np.argmax(prediction)
        return prediction

    def save_checkpoint(self, epoch, file_path):
        # 构造CheckPoint内容
        ckpt = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch,
            # 'lr_schedule': self.lr_schedule.state_dict()
        }
        # 保存文件
        torch.save(ckpt, file_path)

    def load_checkpoint(self, file_path):
        # 加载文件
        ckpt = torch.load(file_path, weights_only=True)
        # 加载模型参数
        self.model.load_state_dict(ckpt['model'])
        # 加载优化器参数
        self.optimizer.load_state_dict(ckpt['optimizer'])
        # 设置开始的epoch
        epoch = ckpt['epoch']
        # 加载lr_scheduler
        # self.lr_schedule.load_state_dict(ckpt['lr_schedule'])
        return epoch


def train_with_resnet(including_finetune=True):
    model = TongueResnet()
    model.transfer_learning_mode()
    loss_func = torch.nn.CrossEntropyLoss()
    # 仅优化分类器参数
    optimizer = torch.optim.Adam(model.resnet.fc.parameters(), lr=0.0001)

    tongs = TongueDataset(batch_size=16)
    # 训练数据
    train_dataloader = tongs.load_train_data()
    # tongs.show_sample_images()
    # 训练数据
    test_dataloader = tongs.load_test_data()
    # 验证数据
    validation_dataloader = tongs.load_validation_data()

    # Train model and save best one
    print('Begin transfer learning...')
    trainer = ModelTrainer(model, loss_func, optimizer)
    trainer.fit(train_dataloader, test_dataloader, 5)

    if including_finetune:
        # 微调
        model.fine_tune_mode()
        # 较小的lr
        optimizer_finetune = torch.optim.Adam(model.parameters(), lr=0.00001)
        print('Begin fine tune...')
        trainer = ModelTrainer(model, loss_func, optimizer_finetune)
        trainer.fit(train_dataloader, test_dataloader, 2)

    # Load best model
    # trainer.load_checkpoint('./VegetableResnet50.ckpt')
    trainer.load_checkpoint(f'./{dataType}Resnet50.ckpt')
    avg_val_loss, avg_val_accuracy = trainer.validate(validation_dataloader)
    print(f'Validation: {avg_val_accuracy * 100}%, {avg_val_loss}')

    # Try to predict single image

    images = [
        './Data/Tongue/validation/shese_an_hong_she/1601019973062595_2024-12-09-16-47-22_0.jpg',
        './Data/Tongue/validation/shese_dan_bai_she/02087.jpg',
        './Data/Tongue/validation/shese_dan_hong_she/01069.jpg',
        './Data/Tongue/validation/shese_dan_she/01007.jpg',
        './Data/Tongue/validation/shese_dan_zi_she/02089.jpg',
        './Data/Tongue/validation/shese_hong_she/1717899939393476_2024-06-09-10-25-39_0.jpg',
        './Data/Tongue/validation/shese_jiang_she/1721954933890934_2024-07-26-08-48-53_0.jpg',
        './Data/Tongue/validation/shese_jiang_zi_she/1607143237690706_2024-03-09-10-35-53_0.jpg',
        './Data/Tongue/validation/shese_zi_hong_she/1715739328072302_2024-05-15-10-15-27_0.jpg'
    ]

    for path in images:
        img = Image.open(path)
        img_tensor = tongs.transform(img)
        img_tensor.unsqueeze_(0)
        img_tensor = img_tensor.to(trainer.device)
        prediction = trainer.predict(img_tensor)
        # numpy需要到CPU上操作
        index = prediction.to('cpu').data.numpy().argmax()
        label = tongs.id_to_class[index]
        print(label)


def predict_images():
    # 初始化模型
    model = TongueResnet()
    model.transfer_learning_mode()

    # 加载已保存的模型权重
    checkpoint_path = f'./{dataType}Resnet50.ckpt'
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # 设置设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # 加载数据集处理器
    veg = TongueDataset(batch_size=16)
    veg.load_train_data()
    veg.load_test_data()
    # 验证数据
    veg.load_validation_data()

    # 定义测试图像列表
    images = [
        './Data/Tongue/validation/shese_an_hong_she/1601019973062595_2024-12-09-16-47-22_0.jpg',
        './Data/Tongue/validation/shese_dan_bai_she/02087.jpg',
        './Data/Tongue/validation/shese_dan_hong_she/01069.jpg',
        './Data/Tongue/validation/shese_dan_she/01007.jpg',
        './Data/Tongue/validation/shese_dan_zi_she/02089.jpg',
        './Data/Tongue/validation/shese_hong_she/1717899939393476_2024-06-09-10-25-39_0.jpg',
        './Data/Tongue/validation/shese_jiang_she/1721954933890934_2024-07-26-08-48-53_0.jpg',
        './Data/Tongue/validation/shese_jiang_zi_she/1607143237690706_2024-03-09-10-35-53_0.jpg',
        './Data/Tongue/validation/shese_zi_hong_she/1715739328072302_2024-05-15-10-15-27_0.jpg'
    ]

    # 逐一测试图像
    for path in images:
        try:
            # 加载图像并进行预处理
            img = Image.open(path)
            img_tensor = veg.transform(img)
            img_tensor.unsqueeze_(0)
            img_tensor = img_tensor.to(device)
            # 模型预测
            with torch.no_grad():
                prediction = model(img_tensor)

            # 获取预测类别
            index = prediction.to('cpu').data.numpy().argmax()
            label = veg.id_to_class[index]
            # print(f"Image: {path}")
            print(f"Predicted Label: {label}")
        except Exception as e:
            print(f"Error processing image {path}: {e}")


def testCpuOrGpu():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Is CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    print(f'Running on {device}')


if __name__ == '__main__':
    testCpuOrGpu()
    train_with_resnet(True)
    predict_images()
