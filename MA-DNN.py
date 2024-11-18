#!/usr/bin/env python
# coding: utf-8

# In[1]:


import scipy.io as scio
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import xlrd
import openpyxl
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import tensorflow as tf
from tensorflow import keras


# In[2]:


df1 = pd.read_excel('水平主余震PGA.xlsx')
df1_1 = df1[['主余震','Fault_type','Earthquake Magnitude','Joyner-Boore Dist. (km)','ClstD (km)','Depth to Top Of Fault Rupture Model','Rx','Vs30 (m/s) selected for analysis','Fault Rupture Width (km)','HypD (km)']]
input1_1 = np.array(df1_1)
df1_output = df1.iloc[:,12]
output1_1 = np.array(df1_output)
CLST_1 = df1_1 .iloc[:, 4]
VS30_1 = df1_1 .iloc[:, 7]
def smooth_log(x, epsilon=1e-8):
    epsilon_added = np.where(x == 0, epsilon, 0)  # 在 x = 0 的位置加上 epsilon
    return np.log(x + epsilon_added)

CLST_1_log = CLST_1
# 对数据进行平滑取对数转换
log_CLST_1 = smooth_log(CLST_1_log)

def smooth_log(x, epsilon=1e-8):
    epsilon_added = np.where(x == 0, epsilon, 0)  # 在 x = 0 的位置加上 epsilon
    return np.log(x + epsilon_added)

VS30_1_log = VS30_1
# 对数据进行平滑取对数转换
log_VS30_1 = smooth_log(VS30_1_log)

def smooth_log(x, epsilon=1e-8):
    epsilon_added = np.where(x == 0, epsilon, 0)  # 在 x = 0 的位置加上 epsilon
    return np.log(x + epsilon_added)

data1_log = output1_1 
# 对数据进行平滑取对数转换
log_df1 = smooth_log(data1_log)

VS_1 = np.array(log_VS30_1)
CD_1 = np.array(log_CLST_1)
Vs30_4 = VS_1.reshape((16861, 1))
CD_4 = CD_1.reshape((16861, 1))
output_11 = log_df1
output_111 = log_df1
zhuyu_1 = input1_1[:,0]
Fault_type_1 = input1_1[:,1]
Earthquake_1 = input1_1[:,2]
RJB_11= input1_1[:,3]
CLSTD_1 = CD_4
Depth_1 = input1_1[:,5]
Rx_1 = input1_1[:,6]
Vs30_1 = Vs30_4 
FaultWidth = input1_1[:,8]
HYPE_11 = input1_1[:,9]


zhuyu_11 = np.array([zhuyu_1])
zhuyu_4 = zhuyu_11.reshape(16861, 1)
Fault_type_11 = np.array([Fault_type_1])
Fault_type_5 = Fault_type_11.reshape(16861, 1)
Earthquake_11 = np.array([Earthquake_1])
Earthquake_5 = Earthquake_11.reshape(16861, 1)
RJB_111 = np.array([RJB_11])
RJB_5 = RJB_111.reshape(16861, 1)
CLSTD_5 = CLSTD_1
Rx_11 = np.array([Rx_1])
Rx_5 = Rx_11.reshape(16861, 1)
Depth_11 = np.array([Depth_1])
Depth_5 = Depth_11.reshape(16861, 1)
FaultWidth_11 = np.array([FaultWidth])
FaultWidth_5 =FaultWidth_11.reshape(16861, 1)
HYPE_111 = np.array([HYPE_11])
HYPE_5 =HYPE_111.reshape(16861, 1)
Vs30_5 =Vs30_1 

input_11 = np.hstack((zhuyu_4,Fault_type_5,Earthquake_5,RJB_5,CLSTD_5, Depth_5,Rx_5,Vs30_5,FaultWidth_5,HYPE_5)) 
output_1111 = np.array([output_111])
output_11= output_1111.reshape(16861, 1)
print(input_11.shape)
print(output_11.shape)
scaler = MinMaxScaler(feature_range=(0,1))  #进行归一化处理
input_1 = scaler.fit_transform(input_11) 
output_1 = scaler.fit_transform(output_11)
train_x_1, test_x1, train_y_1, test_y1 = train_test_split(input_1, output_1, test_size=0.2 ,random_state=42)
train_x = train_x_1
test_x =test_x1
train_y = train_y_1
test_y = test_y1


# In[4]:


df_zinput_11= pd.DataFrame(input_1)

df_zinput_11.to_excel('2归一化输入79.xlsx', index=False)


# In[3]:


scaler = MinMaxScaler(feature_range=(0,1))  #进行归一化处理
input_1 = scaler.fit_transform(input_11) 
output_1 = scaler.fit_transform(output_11)

# 打印归一化前的 train_y 范围
print(f"Original train_y range: {scaler.data_min_[0]} to {scaler.data_max_[0]}")


# In[4]:


import numpy as np
from tensorflow.keras.models import load_model
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 加载训练好的模型
model_path = '水平主余震PGA-613.h5'
trained_model = load_model(model_path)

# 数据准备（确保 train_x 和 train_y 已定义）
# train_x: 训练输入特征，train_y: 训练目标值
X, y = train_x, train_y

# 定义目标值的最小值和最大值，用于反归一化
target_min = -14.4255
target_max = 0.57001

# 定义反归一化函数
def inverse_transform(scaled_values, min_val, max_val):
    """将归一化数据反归一化为原始尺度"""
    return scaled_values * (max_val - min_val) + min_val

# 定义 10 折交叉验证
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# 存储各折的验证结果
r2_scores = []
rmse_scores = []
mae_scores = []

# 开始 10 折交叉验证
fold = 1
for train_idx, val_idx in kfold.split(X):
    print(f"正在处理第 {fold} 折...")

    # 获取验证集
    X_val = X[val_idx]
    y_val = y[val_idx]

    # 模型预测
    y_pred = trained_model.predict(X_val)

    # 反归一化预测值和真实值
    y_pred_original = inverse_transform(y_pred, target_min, target_max)
    y_val_original = inverse_transform(y_val, target_min, target_max)

    # 打印当前折的反归一化范围
    print(f"Fold {fold} y_pred_original range: {y_pred_original.min()} to {y_pred_original.max()}")
    print(f"Fold {fold} y_val_original range: {y_val_original.min()} to {y_val_original.max()}")

    # 计算性能指标
    r2 = r2_score(y_val_original, y_pred_original)
    rmse = np.sqrt(mean_squared_error(y_val_original, y_pred_original))
    mae = mean_absolute_error(y_val_original, y_pred_original)

    # 保存结果
    r2_scores.append(r2)
    rmse_scores.append(rmse)
    mae_scores.append(mae)

    print(f"Fold {fold} Results: R^2 = {r2:.4f}, RMSE = {rmse:.4f}, MAE = {mae:.4f}")
    fold += 1

# 打印交叉验证结果汇总
print("\n交叉验证结果汇总：")
print(f"平均 R^2: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
print(f"平均 RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
print(f"平均 MAE: {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")


# In[4]:


from sklearn.model_selection import KFold
from tensorflow import keras
import numpy as np

# 假设 train_x, train_y, test_x, test_y 已经定义

# 加载已经保存的模型
model = keras.models.load_model('水平主余震PGA-613.h5')

# 创建 KFold 交叉验证对象
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# 用于存储每次验证的结果
cv_train_losses = []
cv_val_losses = []

# 交叉验证循环
for train_index, val_index in kfold.split(train_x):
    # 拆分训练集和验证集
    train_fold_x, val_fold_x = train_x[train_index], train_x[val_index]
    train_fold_y, val_fold_y = train_y[train_index], train_y[val_index]

    # 训练模型
    history = model.fit(train_fold_x, train_fold_y, batch_size=128, epochs=100, validation_data=(val_fold_x, val_fold_y))

    # 记录训练误差和验证误差
    cv_train_losses.append(history.history['loss'][-1])
    cv_val_losses.append(history.history['val_loss'][-1])

# 输出交叉验证结果
print(f'交叉验证训练损失: {cv_train_losses}')
print(f'交叉验证验证损失: {cv_val_losses}')

# 计算平均训练误差和验证误差
mean_train_loss = np.mean(cv_train_losses)
mean_val_loss = np.mean(cv_val_losses)

print(f'平均训练损失: {mean_train_loss}')
print(f'平均验证损失: {mean_val_loss}')

# 比较训练误差和验证误差
if mean_train_loss < mean_val_loss:
    print("模型可能存在过拟合现象")
else:
    print("模型表现良好，未出现明显过拟合")


# In[ ]:


model = tf.keras.models.load_model('水平主余震PGA-613.h5')
import matplotlib.pyplot as plt

# 假设 train_x, train_y, test_x, test_y 已经定义并准备好

# ...（您的模型定义和编译代码保持不变）

history = model.fit(train_x, train_y, batch_size=128, epochs=170, validation_data=(test_x, test_y))

# 提取训练和验证损失
train_loss = history.history['loss']
val_loss = history.history['val_loss']

# 绘制学习曲线
plt.figure(figsize=(10, 6))
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Learning Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# 保存模


# In[ ]:





# In[ ]:





# In[7]:


from tensorflow import keras
from tensorflow.keras import regularizers

model = keras.Sequential()

model.add(keras.layers.Dense(256, activation='relu'))
model.add(keras.layers.Dropout(0.2))  # 添加一个 dropout 层，丢弃率为 0.2
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(1))

model.compile(optimizer=keras.optimizers.Adam(0.001), loss='mean_squared_error')

history = model.fit(train_x, train_y, batch_size=128, epochs=170, validation_data=(test_x, test_y))

model.save('水平主余震PGA-613.h5')


# In[8]:


model = tf.keras.models.load_model('水平主余震PGA-613.h5')
trainPredict_0 = model.predict(train_x)  #用训练好的模型进行预测
testPredict_0 = model.predict(test_x)
predtrain=scaler.inverse_transform(trainPredict_0)
TrainY=scaler.inverse_transform(train_y)
predtest=scaler.inverse_transform(testPredict_0)
TestY=scaler.inverse_transform(test_y)
import matplotlib.pyplot as plt

# 输入x和y坐标的值
x = TrainY
y = predtrain

plt.scatter(x, y)

plt.plot([min(x), max(x)], [min(x), max(x)], color='red')  # 添加红线

plt.title('train_Scatter plot')
plt.show()

import matplotlib.pyplot as plt

# 输入x和y坐标的值
x = TestY
y = predtest

plt.scatter(x, y)

plt.plot([min(x), max(x)], [min(x), max(x)], color='red')  # 添加红线

plt.title('train_Scatter plot')
plt.show()


# In[9]:


scaler = MinMaxScaler(feature_range=(0,1))  #进行归一化处理
input_1 = scaler.fit_transform(input_11) 
output_1 = scaler.fit_transform(output_11)
xunlian_y_0 = model.predict(input_1)
zhenshi_y_1=scaler.inverse_transform(output_1)
xunlian_y_1 =scaler.inverse_transform(xunlian_y_0)
junzhi=np.exp(xunlian_y_1 )
mean = np.mean(junzhi)
zcc_1 = zhenshi_y_1-xunlian_y_1
variance = np.var(zcc_1)
print("Mean:", mean)
print("Variance:", variance)


# In[ ]:




