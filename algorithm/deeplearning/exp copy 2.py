import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import TensorDataset,DataLoader
from torch.optim import lr_scheduler
from algorithm.deeplearning.deeplearningmodel import mymodel
from algorithm.deeplearning.tools import CustomStandardScaler, CustomMinMaxScaler, EarlyStopping
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
import os
import time


class Exp:
    def __init__(self):
        super(Exp, self).__init__()
        #self.data_path = '数据/wind_dataset.csv'
        self.data_path = 'data/Aquifer_Petrignano.csv'
        self.timestep = 60  # 时间步长，就是利用多少时间窗口
        self.batch_size = 64  # 批次大小
        self.feature_size = 5
        self.hidden_size = 512  # 隐层大小
        self.output_size = 30
        self.num_layers = 2  # lstm的层数
        self.dropout = 0.3  # 丢弃率
        self.epochs = 80  # 迭代轮数
        self.learning_rate = 0.0005  # 学习率
        self.model = None
        self.model_name = 'null'  # 模型名称
        self.save_path = 'algorithm/deeplearning/result/model/{}.pth'.format(self.model_name)  # 最优模型保存路径
        self.plotsize = 500
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.scale = CustomMinMaxScaler()
        self.predata = 1  # 预测值在第几列
        self.patience = 50  # 耐心值
        self.dilation = 5
        self.out_channel = 32
        self.kernel_size = 7

        self.num_channels = [128, 64, 32]
        self.num_heads = 4
        self.time = time.time()


    # 形成训练数据，例如12345789 12-3456789
    def split_data(self, data, timestep, feature_size):
        dataX = []  # 已知特征
        dataY = []  # 预测特征
        output_size = self.output_size
        # 将整个窗口的数据保存到X中，将未来N天保存到Y中
        if feature_size == 1:
            for index in range(len(data) - timestep - output_size):
                dataX.append(data[index: index + timestep][:, self.predata])  # 添加 [:,0] 切片
                dataY.append(data[index + timestep:index + timestep + output_size][:, self.predata])  # 添加预测列
        else:
            for index in range(len(data) - timestep - output_size):
                dataX.append(data[index: index + timestep])
                dataY.append(data[index + timestep:index + timestep + output_size][:, self.predata])

        dataX = np.array(dataX)
        dataY = np.array(dataY)

        # 获取训练集大小
        train_size = int(np.round(0.7 * dataX.shape[0]))
        test_size = int(np.round(0.2 * dataX.shape[0]))
        # 划分训练集、验证集、测试集
        x_train = dataX[: train_size, :].reshape(-1, timestep, feature_size)
        y_train = dataY[: train_size].reshape(-1, output_size)
        x_val = dataX[train_size: train_size + test_size, :].reshape(-1, timestep, feature_size)
        y_val = dataY[train_size: train_size + test_size].reshape(-1, output_size)
        x_test = dataX[train_size + test_size:, :].reshape(-1, timestep, feature_size)
        y_test = dataY[train_size + test_size:].reshape(-1, output_size)

        return [x_train, y_train, x_val, y_val, x_test, y_test]


    def dataload(self):
        df = pd.read_csv(self.data_path, index_col=0)
        print(df.head(5))
        df = np.array(df)
        # 将数据进行归一化
        scaler_model = self.scale
        scaler_model.fit_transform(np.array(df[:, self.predata]).reshape(-1, 1))
        scaler = CustomMinMaxScaler()
        df_scale = scaler.fit_transform(df)
        x_train, y_train, x_val, y_val, x_test, y_test = self.split_data(df_scale, self.timestep, self.feature_size)

        # 将数据转为tensor
        x_train_tensor = torch.from_numpy(x_train).to(torch.float32).to(self.device)
        y_train_tensor = torch.from_numpy(y_train).to(torch.float32).to(self.device)
        x_val_tensor = torch.from_numpy(x_val).to(torch.float32).to(self.device)
        y_val_tensor = torch.from_numpy(y_val).to(torch.float32).to(self.device)
        x_test_tensor = torch.from_numpy(x_test).to(torch.float32).to(self.device)
        y_test_tensor = torch.from_numpy(y_test).to(torch.float32).to(self.device)

        self.x_train_tensor = x_train_tensor
        self.y_train_tensor = y_train_tensor
        self.x_test_tensor = x_test_tensor
        self.y_test_tensor = y_test_tensor
        # 形成训练数据集
        train_data = TensorDataset(x_train_tensor, y_train_tensor)
        vali_data = TensorDataset(x_val_tensor, y_val_tensor)
        test_data = TensorDataset(x_test_tensor, y_test_tensor)

        # 将数据加载成迭代器
        train_loader = torch.utils.data.DataLoader(train_data,
                                                   self.batch_size,
                                                   False)
        vali_loader = torch.utils.data.DataLoader(vali_data,
                                                  self.batch_size,
                                                  False)
        test_loader = torch.utils.data.DataLoader(test_data,
                                                  self.batch_size,
                                                  False)
        return train_loader, vali_loader, test_loader

    def setmodel(self, modelname):
        model_params = {
            "LSTM": {
                "model_name": "LSTM",
                "model": mymodel.LSTM(self.feature_size, self.hidden_size, self.num_layers, self.output_size)
            },
            "GRU": {
                "model_name": "GRU",
                "model": mymodel.GRU(self.feature_size, self.hidden_size, self.num_layers, self.output_size)
            },
            "CNNGRU": {
                "model_name": "CNN-GRU",
                "model": mymodel.CNNGRU(self.feature_size, self.out_channel, self.hidden_size, self.num_layers,
                                        self.kernel_size, self.output_size)
            },
            "DAR": {
                "model_name": "DAR",
                "model": mymodel.DAR(self.feature_size, self.num_channels, self.dilation,
                                     self.kernel_size, self.output_size)
            },
            "TCN": {
                "model_name": "TCN",
                "model": mymodel.TCN(self.feature_size, self.output_size, self.num_channels, self.kernel_size,
                                     self.dropout)
            },
        }

        if modelname in model_params:
            params = model_params[modelname]  # 模型字典的字典
            self.model_name = params["model_name"]
            self.model = params["model"].to(self.device)
            self.save_path = 'algorithm/deeplearning/result/model/{}.pth'.format(self.model_name)
        else:
            raise ValueError("Unsupported modelname: {}".format(modelname))
        self.save_path = 'algorithm/deeplearning/result/model/{}.pth'.format(self.model_name)

    # 模型训练
    def train(self):
        try:
            early_stopping = EarlyStopping(patience=self.patience, verbose=True)
            model = self.model
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
            loss_function = nn.MSELoss().to(self.device)
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs // 2)

            train_loader, vali_loader, test_loader = self.dataload()
            self.loss_save1 = []
            self.loss_save2 = []
            self.loss_save3 = []
            self.time = time.time()
            
            # Progress tracking
            total_batches = len(train_loader)

            for epoch in range(self.epochs):
                model.train()
                running_loss = 0
                
                for batch_idx, (x_train, y_train) in enumerate(train_loader):
                    try:
                        optimizer.zero_grad()
                        y_train_pred = model(x_train)
                        loss = loss_function(y_train_pred, y_train)
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item()
                        
                    except Exception as e:
                        print(f"Error during batch {batch_idx+1}: {str(e)}")
                        # Clean up CUDA memory if using GPU
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                        
                train_loss = running_loss / len(train_loader)
                self.loss_save1.append(train_loss)
                
                # Validation
                model.eval()
                with torch.no_grad():
                    running_loss1 = 0
                    for x_val, y_val in vali_loader:
                        y_val_pred = model(x_val)
                        loss = loss_function(y_val_pred, y_val)
                        running_loss1 += loss.item()
                    vali_loss = np.average(running_loss1)
                    self.loss_save2.append(vali_loss)
                    
                    # Testing
                    running_loss2 = 0
                    for x_test, y_test in test_loader:
                        y_test_pred = model(x_test)
                        loss = loss_function(y_test_pred, y_test)
                        running_loss2 += loss.item()
                    test_loss = np.average(running_loss2)
                    self.loss_save3.append(test_loss)
                    
                print(f"Epoch [{epoch+1}/{self.epochs}] - Train Loss: {train_loss:.3f}, Val Loss: {vali_loss:.3f}, Test Loss: {test_loss:.3f}")

                early_stopping(vali_loss, self.model, self.save_path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
                    
                if early_stopping.counter >= 5:
                    scheduler.step()
                    last_lr = str(scheduler.get_last_lr())
                    print(f'Learning rate changed to: {last_lr}')
                    
        except Exception as e:
            print(f"Training error: {str(e)}")
            raise
        finally:
            # Clean up resources
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print('Finished Training')
            self.time = time.time() - self.time
            
        return model

    def paint(self):
        model = self.model
        model.load_state_dict(torch.load(self.save_path))
        model.eval()
        plot_size = self.plotsize
        scaler = self.scale
        x_train_tensor = self.x_train_tensor
        y_train_tensor = self.y_train_tensor
        x_test_tensor = self.x_test_tensor
        y_test_tensor = self.y_test_tensor

        train_predicted = scaler.inverse_transform(model(x_train_tensor).detach().cpu().numpy())
        train_actual = scaler.inverse_transform(y_train_tensor.detach().cpu().numpy())
        train_predicted_display = train_predicted[:plot_size]
        train_actual_display = train_actual[: plot_size]

        y_test_pred = scaler.inverse_transform(model(x_test_tensor).detach().cpu().numpy())
        y_test_actual = torch.tensor(scaler.inverse_transform(y_test_tensor.detach().cpu().numpy()))
        y_test_pred_display = y_test_pred[: plot_size]  # ytest预测值显示
        y_test_actual_display = y_test_actual[: plot_size]  # ytest真实值显示

        # 均方误差
        sme = mean_squared_error(y_test_actual[:, -1], y_test_pred[:, -1])
        # 均方根误差
        srme = mean_squared_error(y_test_actual[:, -1], y_test_pred[:, -1])
        # 平均绝对误差
        mae = mean_absolute_error(y_test_actual[:, -1], y_test_pred[:, -1])
        # 平均绝对百分比误差
        mape = torch.mean(torch.abs((y_test_actual[:, -1] - y_test_pred[:, -1]) / y_test_actual[:, -1]))
        # 对称平均绝对百分比误差
        smape = torch.mean(torch.abs(y_test_actual[:, -1] - y_test_pred[:, -1]) / torch.abs(
            (y_test_actual[:, -1] + y_test_pred[:, -1]) / 2))
        r_squared = r2_score(y_test_actual[:, -1], y_test_pred[:, -1])
        print(f'sme:{sme}')
        # print(srme)
        # print(mae)
        print(f'mape{mape.item()}')
        # print(smape)
        print(f'r2{r_squared}')

        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
        plt.rcParams.update({'font.size': 14})
        fig, axs = plt.subplots(2, 1, figsize=(9, 7))
        axs[0].set_title("模型训练结果")
        sns.lineplot(data=y_test_pred_display[:, -1], label='预测值', ax=axs[0])
        sns.lineplot(data=y_test_actual_display[:, -1], label='真实值', ax=axs[0])
        axs[0].legend()

        train_losses = np.array(self.loss_save1)
        val_losses = np.array(self.loss_save2)
        test_losses = np.array(self.loss_save3)

        axs[1].set_title("损失函数值")
        sns.lineplot(data=train_losses, label='训练集损失', ax=axs[1])
        sns.lineplot(data=val_losses, label='验证集损失', ax=axs[1])
        sns.lineplot(data=test_losses, label='测试集损失', ax=axs[1])
        min_train_loss = min(train_losses)  #最小值
        min_train_loss_idx = np.argmin(train_losses)  #最小值索引
        min_val_loss = min(val_losses)
        min_val_loss_idx = np.argmin(val_losses)

        axs[1].scatter(min_train_loss_idx, min_train_loss, color='r', marker='o',
                       label=f'训练集最小损失值 ({min_train_loss:.4f})')
        axs[1].scatter(min_val_loss_idx, min_val_loss, color='r', marker='o',
                       label=f'验证集最小损失值 ({min_val_loss:.4f})')
        axs[1].text(min_val_loss_idx - 0.5, min_val_loss, f'验证集最小损失值 ({min_val_loss:.4f})',
                    fontsize=12, verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.6, edgecolor='black'))

        hyperparameter_text = f'num_channels: {self.num_channels}\n' \
                              f'output size: {self.output_size}\n' \
                              f'kernel size: {self.kernel_size}\n' \
                              f'timestep: {self.timestep}\n' \
                              f'epochs: {self.epochs}\n' \
                              f'dilation: {self.dilation}'

        #axs[1].text(0.4, 0, hyperparameter_text, transform=axs[1].transAxes,
        #fontsize=12, verticalalignment='center', bbox=dict(facecolor='none', edgecolor='gray'))

        axs[1].legend()
        fig.subplots_adjust(hspace=0.5)

        #保留模型标注参数
        if not os.path.exists('%s %d %.3f %.3f %.3f' % (self.save_path, self.time, sme, mape, r_squared)):
            os.rename(self.save_path, '%s %d %.3f %.3f %.3f' % (self.save_path, self.time, sme, mape, r_squared))

        else:
            os.rename(self.save_path, '%s +1 %d %.3f %.3f %.3f' % (self.save_path, self.time, sme, mape, r_squared))

        #保存绘图
        if not os.path.exists('algorithm/deeplearning/result/pic/%s %.3f %.3f %.3f.png' % (self.model_name, sme, mape.item(), r_squared)):
            plt.savefig('algorithm/deeplearning/result/pic/%s %.3f %.3f %.3f.png' % (self.model_name, sme, mape.item(), r_squared))

        else:
            plt.savefig('algorithm/deeplearning/result/pic/+1%s %.3f %.3f %.3f.png' % (self.model_name, sme, mape.item(), r_squared))

        y_test = [np.array(y_test_actual), y_test_pred]

        #保存loss值和预测值
        if not os.path.exists('algorithm/deeplearning/result/pre&loss/%s %.3f.npy' % (self.model_name, r_squared)):
            np.save('algorithm/deeplearning/result/pre&loss/%s %.3f.npy' % (self.model_name, r_squared), y_test)
            np.save('algorithm/deeplearning/result/loss/%s %.3f.npy' % (self.model_name, r_squared), [train_losses, val_losses, test_losses])
        else:
            np.save('algorithm/deeplearning/result/pre&loss/+1%s %.3f.npy' % (self.model_name, r_squared), y_test)
            np.save('algorithm/deeplearning/result/loss/+1%s %.3f.npy' % (self.model_name, r_squared), [train_losses, val_losses, test_losses])

        return 'algorithm/deeplearning/result/pic/%s %.3f %.3f %.3f.png' % (self.model_name, sme, mape.item(), r_squared)
