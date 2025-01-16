import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import torch
import seaborn as sns


class CustomStandardScaler:
    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, data):
        return self.scaler.fit_transform(data)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class CustomMinMaxScaler:
    def __init__(self):
        self.scaler = MinMaxScaler()

    def fit_transform(self, data):
        return self.scaler.fit_transform(data)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


# 绘制结果
def paint(exp):
    model = exp.model
    model.load_state_dict(torch.load(exp.save_path))
    model.eval()
    plot_size = exp.plotsize
    scaler = exp.scale
    x_train_tensor = exp.x_train_tensor
    y_train_tensor = exp.y_train_tensor
    x_test_tensor = exp.x_test_tensor
    y_test_tensor = exp.y_test_tensor

    train_predicted = scaler.inverse_transform(model(x_train_tensor).detach().cpu().numpy())
    train_actual = scaler.inverse_transform(y_train_tensor.detach().cpu().numpy())
    train_predicted_display = train_predicted[:plot_size]
    train_actual_display = train_actual[: plot_size]

    y_test_pred = scaler.inverse_transform(model(x_test_tensor).detach().cpu().numpy())
    y_test_actual = torch.tensor(scaler.inverse_transform(y_test_tensor.detach().cpu().numpy()))
    y_test_pred_display = y_test_pred[: plot_size]  # ytest预测值显示
    y_test_actual_display = y_test_actual[: plot_size]  # ytest真实值显示

    # 均方误差
    sme = mean_squared_error(y_test_actual, y_test_pred, squared=False)
    # 均方根误差
    srme = mean_squared_error(y_test_actual, y_test_pred)
    # 平均绝对误差
    mae = mean_absolute_error(y_test_actual, y_test_pred)
    # 平均绝对百分比误差
    mape = torch.mean(torch.abs((y_test_actual - y_test_pred) / y_test_actual))
    # 对称平均绝对百分比误差
    smape = torch.mean(torch.abs(y_test_actual - y_test_pred) / torch.abs((y_test_actual + y_test_pred) / 2))
    r_squared = r2_score(y_test_actual, y_test_pred)
    print(sme)
    # print(srme)
    # print(mae)
    print(mape.item())
    # print(smape)
    print(r_squared)

    if exp.output_size == 1:
        plt.figure(figsize=(18, 5))
        ax1 = plt.subplot(1, 2, 1)
        ax1.set_title("train")
        ax1.plot(train_predicted_display, "b", label="Predicted")
        ax1.plot(train_actual_display, "r", label='Actual')
        ax1.legend()

        # 绘制预测值和实际值的曲线
        ax2 = plt.subplot(1, 2, 2)
        ax2.set_title("test")
        ax2.plot(y_test_pred_display, "b", label='Predicted')
        ax2.plot(y_test_actual_display, "r", label='Actual')
        ax2.legend()
        plt.tight_layout()
        plt.savefig('图片/%s %.3f %.3f %.3f.png' % (exp.model_name, sme, mape.item(), r_squared))

    else:
        plt.figure(figsize=(15, 5))
        plt.plot(y_test_pred_display[:, -1], "b", label='Predicted')
        plt.plot(y_test_actual_display[:, -1], "r", label='Actual')
        plt.legend()
        plt.savefig('图片/multipre_%s %.3f %.3f %.3f.png' % (exp.model_name, sme, mape.item(), r_squared))


def paintlosses(model_name, train_losses, test_losses):
    plt.figure(figsize=(12, 8))
    plt.title("loss")
    plt.plot(train_losses, "b", label='trainloss')
    plt.plot(test_losses, "r", label='testloss')
    plt.legend()
    # plt.show()
    plt.savefig('图片/{}loss.png'.format(model_name))


def attendmaxmin(data, timestap, predata):
    maxcolumn = np.zeros(data.shape[0])
    mincolumn = np.zeros(data.shape[0])
    for i in range(data.shape[0] - timestap + 1):
        maxcolumn[i] = np.max(data[i:i + timestap, predata])
        mincolumn[i] = np.min(data[i:i + timestap, predata])
    for i in range(timestap - 1):
        maxcolumn[-i] = np.max(data[-timestap:-1, predata])
        mincolumn[-i] = np.min(data[-timestap:-1, predata])
    data = np.hstack((data, maxcolumn.reshape(-1, 1)))
    data = np.hstack((data, mincolumn.reshape(-1, 1)))
    return data
