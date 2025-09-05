import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
plt.style.use('ggplot')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 平滑处理，类似tensorboard的smoothing函数。
def smooth(read_path, save_path, file_name, x='Step', y='Value', weight=0.95):

    data = pd.read_csv(read_path + file_name)
    scalar = data[y].values
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val

    save = pd.DataFrame({x: data[x].values, y: smoothed})
    save.to_csv(save_path + 'smooth_'+ file_name)

#smooth(read_path='/home/gzz/MQL_mujoco/data/log3/', save_path='/home/gzz/MQL_mujoco/data/log3/', file_name='run-.-tag-episode_reward(2).csv')

#smooth(read_path='/home/gzz/下载/', save_path='/home/gzz/MQL_mujoco/data/log3/', file_name='run-MQL_e-noise_0.3-tag-episode_reward(2).csv')
#smooth(read_path='/home/gzz/下载/', save_path='/home/gzz/MQL_mujoco/data/log3/', file_name='run-MQL_3.11G-tag-episode_reward.csv')
#smooth(read_path='/home/gzz/下载/', save_path='/home/gzz/MQL_mujoco/data/log3/', file_name='run-LAF-MQL_2.10-tag-episode_reward(2).csv')
#smooth(read_path='/home/gzz/下载/', save_path='/home/gzz/MQL_mujoco/data/log3/', file_name='run-LAF-MQL_3.15-tag-episode_reward.csv')
df1 = pd.read_csv('/home/gzz/MQL_mujoco/data/log3/smooth_run-MQL_e-noise_0.3-tag-episode_reward(1).csv')
#df2 = pd.read_csv('/home/gzz/MQL_mujoco/data/log3/smooth_run-MQL_3.11G-tag-episode_reward.csv')
df3 = pd.read_csv('/home/gzz/MQL_mujoco/data/log3/smooth_run-LAF-MQL_2.10-tag-episode_reward.csv')
#df4 = pd.read_csv('/home/gzz/MQL_mujoco/data/log3/smooth_run-LAF-MQL_3.15-tag-episode_reward.csv')
df =df1.append(df3)
df.index = range(len(df))
print(df)
plt.figure(figsize=(15, 10))
sns.lineplot(x='Step', y='Value',hue='Alg', data=df)
#sns.relplot(x='Step', y='Value',kind ='line',ci ='sd',hue='Alg',data=df)
plt.show()




'''
#!/usr/bin/python
# -*- coding: utf-8 -*-
# Time: 2021-3-19
# Author: ZYunfei
# File func: draw func

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
import os
myfont=FontProperties(fname=r'/home/gzz/下载/simsun.ttc')
sns.set(font=myfont.get_name())

class Painter:
    def __init__(self, load_csv, load_dir=None):
        if not load_csv:
            self.data = pd.DataFrame(columns=['Method','episode', 'episode reward'])
        else:
            self.load_dir = load_dir
            if os.path.exists(self.load_dir):
                print("==正在读取{}。".format(self.load_dir))
                self.data = pd.read_csv(self.load_dir).iloc[:,1:] # csv文件第一列是index，不用取。
                print("==读取完毕。")
            else:
                print("==不存在{}下的文件，Painter已经自动创建该csv。".format(self.load_dir))
                self.data = pd.DataFrame(columns=['Method','episode', 'episode reward'])
        self.xlabel = None
        self.ylabel = None
        self.title = None
        self.hue_order = None

    def setXlabel(self,label): self.xlabel = label

    def setYlabel(self, label): self.ylabel = label

    def setTitle(self, label): self.title = label

    def setHueOrder(self,order):
        """设置成['name1','name2'...]形式"""
        self.hue_order = order

    def addData(self, dataSeries, method, smooth = True):
        if smooth:
            dataSeries = self.smooth(dataSeries)
        size = len(dataSeries)
        for i in range(size):
            dataToAppend = {'episode reward':dataSeries[i],'episode':i+1,'Method':method}
            self.data = self.data.append(dataToAppend,ignore_index = True)

    def drawFigure(self):
        sns.set_theme(style="darkgrid")
        sns.set_style(rc={"linewidth": 1})
        print("==正在绘图...")
        sns.relplot(data = self.data, kind = "line", x='Step', y='Value')
                    #x = "episode", y = "episode reward")
                   # hue= "Method", hue_order=None)
        plt.title(self.title,fontsize = 12)
        plt.xlabel(self.xlabel)
        plt.ylabel(self.ylabel)
        print("==绘图完毕！")
        plt.show()

    def saveData(self, save_dir):
        self.data.to_csv(save_dir)
        print("==已将数据保存到路径{}下!".format(save_dir))

    def addCsv(self, add_load_dir):
        """将另一个csv文件合并到load_dir的csv文件里。"""
        add_csv = pd.read_csv(add_load_dir).iloc[:,1:]
        self.data = pd.concat([self.data, add_csv],axis=0,ignore_index=True)

    def deleteData(self,delete_data_name):
        """删除某个method的数据，删除之后需要手动保存，不会自动保存。"""
        self.data = self.data[~self.data['Method'].isin([delete_data_name])]
        print("==已删除{}下对应数据!".format(delete_data_name))

    def smoothData(self, smooth_method_name,N):
        """对某个方法下的reward进行MA滤波，N为MA滤波阶数。"""
        begin_index = -1
        mode = -1  # mode为-1表示还没搜索到初始索引， mode为1表示正在搜索末尾索引。
        for i in range(len(self.data)):
            #if self.data.iloc[i]['Method'] == smooth_method_name and mode == -1:
             #   begin_index = i
             #   mode = 1
              #  continue
            if mode == 1 and self.data.iloc[i]['episode'] == 1:
                self.data.iloc[begin_index:i,0] = self.smooth(
                    self.data.iloc[begin_index:i,0],N = N
                )
                begin_index = -1
                mode = -1
                if self.data.iloc[i]['Method'] == smooth_method_name:
                    begin_index = i
                    mode = 1
            if mode == 1 and i == len(self.data) - 1:
                self.data.iloc[begin_index:,0]= self.smooth(
                    self.data.iloc[begin_index:,0], N=N
                )
        print("==对{}数据{}次平滑完成!".format(smooth_method_name,N))

    @staticmethod
    def smooth(data,N=5):
        n = (N - 1) // 2
        res = np.zeros(len(data))
        for i in range(len(data)):
            if i <= n - 1:
                res[i] = sum(data[0:2 * i+1]) / (2 * i + 1)
            elif i < len(data) - n:
                res[i] = sum(data[i - n:i + n +1]) / (2 * n + 1)
            else:
                temp = len(data) - i
                res[i] = sum(data[-temp * 2 + 1:]) / (2 * temp - 1)
        return res



if __name__ == "__main__":
    painter = Painter(load_csv=True,load_dir='/home/gzz/下载/run-e-noise_0.1-tag-episode_reward.csv')
    painter = Painter(load_csv=True, load_dir='/home/gzz/下载/run-log1-tag-episode_reward(1).csv')
    painter.smoothData('MQL',33)
    painter.smoothData('LAF-MQL', 33)
    painter.drawFigure()'''

