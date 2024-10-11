
from drawTool import plotFigureDA,plotFigureTarget,plotFigureSegDA,plotFigureSegDATarget
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from IPython import display

def save_pickle(data, file_name):
    f = open(file_name, "wb")
    pickle.dump(data, f)
    f.close()
def load_pickle(file_name):
    f = open(file_name, "rb+")
    data = pickle.load(f)
    f.close()
    return data


if __name__ == '__main__':
    for ii in range(6):
        attr_name = f'dist_{ii}'
    rootpath='../DistPic/savedata'
    all_files = os.listdir(rootpath)
    pkl_files = [f for f in all_files if f.endswith('.pkl')]
    pkl_data = []

    for pkl_file in pkl_files:
        # 构建完整的文件路径
        file_path = os.path.join(rootpath, pkl_file)

        # 读取Pickle文件
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            pkl_data.append(data)
            # print('aa',len(data))
            # print(data)
    x = np.linspace(0, len(pkl_data[0]['dist_0']),len(pkl_data[0]['dist_0']))  # 横坐标数据点序列
    print(len(pkl_data[0]['dist_0']),x)
    # 创建6个子图

    # plt.figure(figsize=(10, 6))  # 设置图表尺寸
    color=['b','g','r','c','m','y']
    for num in range(len(pkl_data)):
        fig, axs = plt.subplots(2, 3, figsize=(20, 10))  # 创建2x3的子图

        for i in range(6):
            ax = axs[i // 3][i % 3]  # 获取当前子图
            ax.plot(x, pkl_data[num]['dist_%d' % i], label='dist_%d' % i, color=color[i])
            # plt.plot(x, pkl_data[num]['dist_%d'%i], label='dist_%d'%i, color=color[i])
        # 添加图例、标题和坐标轴标签
            ax.legend()  # 添加图例
            ax.set_title("Nearest prototype dist for 6 classe")  # 图表标题
            ax.set_xlabel("Data Points")  # 横坐标标签
            ax.set_ylabel("Distances")  # 纵坐标标签
            # ax.set_tight_layout()
        plt.savefig('../DistPic/savepic/dist_%d.png' % num)
        plt.clf()
        display.clear_output(wait=True)
        display.display(plt.gcf())