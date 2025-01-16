from exp import Exp
import os

print('finished import')


def check_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"文件夹 '{directory}' 不存在，已成功创建。")
    else:
        print(f"文件夹 '{directory}' 已存在。")


# 指定要检查的目录路径
directory_paths = ["result/pic", "result/model", "result/pre&loss", 'result/loss']
for path in directory_paths:
    # 调用函数检查并创建目录
    check_directory(path)

exp = Exp()
#exp2 = Exp2()
for _ in range(1):

    exp.setmodel("GRU")
    exp.train()
    exp.paint()
    exp.setmodel("TCN")
    exp.train()
    exp.paint()
    exp.setmodel("LSTM")
    exp.train()
    exp.paint()
    exp.setmodel("CNNGRU")
    exp.train()
    exp.paint()
'''    

    exp.setmodel("DAR")
    exp.train()
    exp.paint()


'''
