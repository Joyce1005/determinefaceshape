#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 基本庫
import numpy as np  # 用於多維陣列、矩陣運算
import pandas as pd  # 用於從Excel讀取和處理資料
import seaborn as sns  # 數據視覺化
import matplotlib.pyplot as plt  # 繪製2D圖表

# 機器學習與數據處理
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score  # 評估模型表現
from sklearn.model_selection import train_test_split  # 分割數據集
from sklearn.preprocessing import LabelEncoder, MinMaxScaler  # 數據標籤處理與標準化

# 深度學習框架 (使用 TensorFlow 的 Keras API)
from tensorflow.keras.models import Sequential  # 用於建構模型
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization  # 神經網路層
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler  # 訓練過程中的回調函數
from tensorflow.keras.optimizers import Adam  # 優化器


# In[2]:


# 設置資料路徑
directory_path = 'D:\\專題\\all\\model_other\\1_15_angle_pi(完全一致 增加鏡射).xlsx'

# 讀取檔案，使用指定位置讀取數據
try:
    df = pd.read_excel(directory_path, header=None)  # 不使用標頭，讀取原始數據
    # 指定特徵和標籤的列範圍
    df_features = df.iloc[:, 4:19]  # E到S列（特徵角度）
    df_label = df.iloc[:, 3]        # D列（判斷的種類標籤）
except Exception as e:
    print(f"讀取文件時發生錯誤: {e}")

# 檢查數據異常
if df_features.isnull().values.any() or df_label.isnull().values.any():
    print("數據中存在缺失值，請檢查數據。")
else:
    # 檢查數據集是否平衡
    print(df_label.value_counts())


# In[3]:


#設定學習率調度器函數
def scheduler(epoch, lr):
  if epoch < 10:
    return lr
  else:
    return lr * 0.90 #每10輪減少10%的學習率


# In[4]:


#定義學習率調度器
lr_scheduler = LearningRateScheduler(scheduler)

#初始學習率
optimizer = Adam(learning_rate=0.001)

#載入和預處理數據
x = df_features  #特徵
y = df_label  #目標

#文字標籤轉成數值標籤
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

#特徵標準化
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)
x_train = x_scaled
y_train = y_encoded

# 建立神經網絡模型
model = Sequential([
    Dense(128, input_dim=15, activation='relu'),  #隱藏層 1
    BatchNormalization(),              #批標準化層
    Dropout(0.3),                  #避免過擬合的 Dropout

    Dense(64, activation='relu'),          #隱藏層 2
    BatchNormalization(),              #批標準化層
    Dropout(0.3),                  #避免過擬合的 Dropout

    Dense(32, activation='relu'),          #隱藏層 2
    BatchNormalization(),              #批標準化層
    Dropout(0.3),                  #避免過擬合的

    Dense(16, activation='relu'),          #隱藏層 2
    BatchNormalization(),              #批標準化層
    Dropout(0.3),                  #避免過擬合的

    Dense(8, activation='relu'),          #隱藏層 2
    BatchNormalization(),              #批標準化層
    Dropout(0.3),                  #避免過擬合的

    Dense(3, activation='softmax')         #輸出層
])

from sklearn.utils import class_weight

# 計算類別權重
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',  # 使各類別的權重平衡
    classes=np.unique(y_train),  # 傳入唯一的類別
    y=y_train  # 傳入訓練數據的標籤
)

# 轉換為字典形式傳入模型
class_weights_dict = dict(enumerate(class_weights))

# 編譯模型
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 設定早期停止回調
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 訓練模型，移除 validation_data
history = model.fit(x_train, y_train, epochs=100, batch_size=16, callbacks=[early_stopping], class_weight=class_weights_dict)# 加入計算好的類別權重

# 保存訓練好的模型
model.save('D:\\專題\\all\\model_other\\model_A.keras')

import joblib  # 用於保存和加載模型和預處理器

# 保存 scaler
joblib.dump(scaler, 'D:\\專題\\all\\model_other\\scaler.pkl')

# 保存 label_encoder
joblib.dump(label_encoder, 'D:\\專題\\all\\model_other\\label_encoder.pkl')


# In[5]:


from tensorflow.keras.models import load_model #載入訓練好的模型
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report #評估模型


# In[6]:


# 讀取測試資料並進行預測
directory_path_test = 'D:\\專題\\all\\model_other\\1_15_angle_pi(不一致).xlsx'
try:
    df = pd.read_excel(directory_path_test, header=None)  # 不使用標頭
    df_features = df.iloc[:, 4:19]  # E到S列（特徵角度）
    df_label = df.iloc[:, 3]        # D列（判斷的種類標籤）
except Exception as e:
    print(f"讀取文件時發生錯誤: {e}")
# 檢查數據異常
if df_features.isnull().values.any() or df_label.isnull().values.any():
    print("數據中存在缺失值，請檢查數據。")
else:
    # 檢查數據集是否平衡
    print(df_label.value_counts())


# In[7]:


#測試資料
x_B = df_features  #測試資料特徵
y_B = df_label  #測試資料目標

#文字標籤轉成數值標籤
y_B_encoded = label_encoder.transform(y_B)

#特徵標準化
x_B_scaled = scaler.transform(x_B)

#載入訓練好的模型
model_A_loaded = load_model('D:\\專題\\all\\model_other\\model_A.keras')

# 在測試資料上進行預測
y_B_pred = model_A_loaded.predict(x_B_scaled)
y_B_pred_classes = np.argmax(y_B_pred, axis=1)

# 計算評估指標
accuracy = accuracy_score(y_B_encoded, y_B_pred_classes)
precision = precision_score(y_B_encoded, y_B_pred_classes, average='weighted')
recall = recall_score(y_B_encoded, y_B_pred_classes, average='weighted')
f1 = f1_score(y_B_encoded, y_B_pred_classes, average='weighted')

print(f"在測試資料上的準確率: {accuracy}")
print(f"在測試資料上的精確率: {precision}")
print(f"在測試資料上的召回率: {recall}")
print(f"在測試資料上的F1分數: {f1}")

# 印出分類報告和混淆矩陣
print("\n分類報告:")
print(classification_report(y_B_encoded, y_B_pred_classes, target_names=label_encoder.classes_))

print("\n混淆矩陣:")
print(confusion_matrix(y_B_encoded, y_B_pred_classes))


# In[ ]:




