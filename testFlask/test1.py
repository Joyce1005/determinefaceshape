#!/usr/bin/env python
# coding: utf-8

# # 點擊圖片回傳

# +判斷眼距

# In[1]:


import os
import subprocess
import pandas as pd
import re
import math
import openpyxl
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import shutil
from flask import Flask, request, render_template, redirect, url_for, send_from_directory, jsonify
from PIL import Image
from werkzeug.utils import secure_filename
from flask_wtf.csrf import CSRFProtect
import uuid
import tensorflow as tf


# In[2]:


def getOpenface(directory_path, filename_jpg):
    filename_csv = filename_jpg.replace('.jpg', '.csv')
    file_path = os.path.join(directory_path, filename_csv)
    df = pd.read_csv(file_path)
    return df

# 用於檢查上傳的檔案是否具有允許的副檔名
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def run_openface(input_path, output_path):
    # OpenFace 的命令行指令
    command = [
        'C:\\Users\\user\\Downloads\\OpenFace_2.2.0_win_x64\\OpenFace_2.2.0_win_x64\\FaceLandmarkImg.exe',  # OpenFace 的執行檔路徑
        '-f', input_path,                 # 輸入文件（圖片或影片）
        '-out_dir', output_path           # 輸出文件的資料夾
    ]

    try:
        # 使用 subprocess 來執行命令
        subprocess.run(command, check=True)
        #print("OpenFace 執行成功")
    except subprocess.CalledProcessError as e:
        print(f"執行失敗: {e}")


# 定義兩點距離
def dis(df, p1, p2):
    return ((df.iat[0,p1+296] - df.iat[0,p2+296])**2 + (df.iat[0,p1+364] - df.iat[0,p2+364])**2)**(0.5)    


# 定義計算弧度
def calAngle(df, p1, p2, p3):  # 點由左到右
    a = dis(df, p1, p2)
    b = dis(df, p2, p3)
    c = dis(df, p1, p3)

    cosAngle = (a**2 + b**2 - c**2) / (2*a*b)
    return math.acos(cosAngle)*(180/math.pi)*math.pi/180 #弧度


# 新增額頭三點至文件中
def AddForeheadPt(directory_path, coords):
    # 確認 coords 至少有 3 個點
    if len(coords) < 3:
        raise ValueError("coords 需要至少 3 個點")
    
    for filename_csv in os.listdir(directory_path):
        if filename_csv.endswith('.csv'):
            file_path = os.path.join(directory_path, filename_csv)

            # 嘗試讀取 CSV 文件
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                print(f"讀取 CSV 文件時發生錯誤: {file_path}, {str(e)}")
                continue

            # 新增欄位到最後一欄
            try:
                df['x_68'] = coords[0]['x']
                df['y_68'] = coords[0]['y']

                df['x_69'] = coords[2]['x']
                df['y_69'] = coords[2]['y']

                df['x_70'] = coords[1]['x']
                df['y_70'] = coords[1]['y']
            except IndexError as e:
                print(f"新增欄位時發生錯誤，coords 資料不足: {str(e)}")
                continue
            except Exception as e:
                print(f"修改 DataFrame 時發生錯誤: {str(e)}")
                continue

            # 保存修改後的 CSV 文件
            try:
                df.to_csv(file_path, index=False)
                #print(f"已成功更新文件: {file_path}")
                return file_path
            except Exception as e:
                print(f"保存 CSV 文件時發生錯誤: {file_path}, {str(e)}")
                continue

                
# 生成弧度檔案
def BuildRadianFile(file_path):
    os.chdir('D:\\專題\\all\\output') 
    wb = openpyxl.Workbook()    # 建立空白的 Excel 活頁簿物件
    s1 = wb.create_sheet('angle')

    # 目錄下的所有文件
    fn_angle = {}

    # 讀取 .csv 文件
    df = pd.read_csv(file_path)

    # 將15個角度加入angleArr
    angleArr = []
    for i in range(0, 15):
        angleArr.append(calAngle(df, i, i+1, i+2))

    s1.append([file_path]+angleArr) # 寫入 .csv 文件

    wb.save('radianFile.xlsx')  # 儲存檔案
    #print("Success! (弧度檔案)")
    
    
# get下巴形狀
def getJaw():
    #讀取訓練數據，要留(才不會爆炸)
    train_df = pd.read_excel('D:\\專題\\all\\model_other\\1_15_angle_pi(完全一致 增加鏡射).xlsx', header=None)

    #提取訓練數據的特徵和標籤，要留
    x_train = train_df.iloc[:, 4:19].values
    y_train = train_df.iloc[:, 3].values

    #數據預處理，要留
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    #特徵標準化，要留
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    #載入模型，要留且改路徑(附件1)
    model = tf.keras.models.load_model('D:\\專題\\all\\model_other\\model_A.keras')

    #讀取數據 !!! 注意我 !!! 這一行可以指定要讀取的文件路徑
    df = pd.read_excel('D:\\專題\\all\\output\\radianFile.xlsx', sheet_name='angle', header=None)

    #提取應用數據，要留
    data = df.iloc[0, :16].values
    filename_jpg = data[0]
    features = data[1:]

    #應用數據處理，要留
    features = np.array(features).reshape(1, -1)  # 將features轉為2D數組
    features = scaler.transform(features)  # 使用訓練時的縮放器進行數據縮放

    #產生預測結果，要留
    predictions = model.predict(features)

    #判斷結果-找出最大機率的索引，要留
    predicted_class = np.argmax(predictions, axis=1)

    #將數值標籤轉回文字標籤，要留
    label_encoder.classes_ = np.array(['圓', '尖', '方'])  # 根據你的類別進行設置
    predicted_label = label_encoder.inverse_transform(predicted_class)
    predicted_probabilities = predictions[0]

    #輸出結果，要進入判斷程序
    #print(f"檔名: {filename_jpg}")
    #print(f"預測類別: {predicted_label[0]}")
    #print("各類別預測機率:")
    #for label, probability in zip(label_encoder.classes_, predicted_probabilities):
        #print(f"{label}: {probability:.4f}")
    return predicted_label[0] #回傳下巴形狀


# In[3]:


#final判斷臉型
def getFaceshape(file_path, jaw):
    # 讀取 .csv 文件
    df = pd.read_csv(file_path)
    
    
    ###################tmpPrint#####################
    print("x8: ",df.iat[0,304], "x51: ", df.iat[0,347], "x27: ", df.iat[0,323])
    


    mark = df.iat[0,393]    #基準線p29
    lip_mark = (df.iat[0,420] + df.iat[0,422])/2   #基準線56-58

    j = 16
    
    #find 基準線(上)
    for i in range(364,372):
        if ((df.iat[0,i] + df.iat[0,i+j])/2 > mark):
            mark = i%364   #基準線p29下,第一條線左邊點在excel的index(y),求x -> y-68
            mark2 = (i+j)%364
            break
        j-=2

    #df.iat[0, i] = p0_y, p0_x = df.iat[0, i-68], p16_y -> df.iat[0, j], p16_x -> df.iat[0,j-68]

    s = 2
    #find 基準線(下)
    for t in range(371, 364,-1):
        if ((df.iat[0,t] + df.iat[0,t+s])/2 < lip_mark):
            lip_mark = (t + 1)%364  #基準線56-58上,第一條線左邊點在excel的index(y)
            lip_mark2 = (s + t - 1)%364
            break
        s+=2

    key1 = {} #判斷1

    #find 額頭長度 -> max_width1
    max_width1 = ((df.iat[0,713] - df.iat[0, 711])**2 + (df.iat[0,714] - df.iat[0, 712])**2)**(0.5)   
    key1.update({1:max_width1})

    #find 臉頰MAX寬 -> max_width2
    max_width2 = dis(df, mark, 16-mark)
    #print(max_width2)
    for i in range(mark+1, lip_mark):
        j = 16-i
        #print(dis(i,j))
        if dis(df, i, j) > max_width2:
            max_width2 = dis(df, i, j)
    key1.update({2:max_width2})

    #find 下巴MAX寬 -> max_width3
    max_width3 = dis(df, lip_mark, 16-lip_mark)
    #print(max_width3)
    for i in range(lip_mark+1, 8):
        j = 16-i
        #print(dis(i,j))
        if dis(df, i, j) > max_width3:
            max_width3 = dis(df, i, j)
    key1.update({3:max_width3})

    key1_sorted = sorted(key1.items(),key=lambda item:item[1],reverse=True)   

    height = ((df.iat[0,372] - df.iat[0, 716])**2 + (df.iat[0,304] - df.iat[0, 715])**2)**(0.5)   #p70-p8 

    
    #判斷 2 ->  判斷長寬比 明顯(1)or不明顯(2:黃金比例, 3:不明顯)
    #-----明顯------
    if height/max_width2 > 1.48:
        key2 = 1
    #-----不明顯-----
    elif height/max_width2 > 1.4:
            key2 = 2
    else:
        key2 = 3

    #print((max_width1 - max_width3)/max_width3)

    change = 0.5

    #start判斷
    if key1_sorted[0][0] == 1: #額頭最長
        if key2 == 1: #長寬比明顯
            faceshape = "長臉"
        else:
            if max_width1 > max_width3: #額頭>下顎
                faceshape = "長臉" if jaw == '圓' else "心型臉"
                if (max_width1 - max_width3)/max_width3 <= change:
                    faceshape = "長臉" if jaw == '尖' else "方臉"
            else:
                faceshape = "長臉" if jaw == '尖' else "方臉"
    elif key1_sorted[0][0] == 2:#臉頰最長
        if key2 == 1: #長寬比明顯
            if (max_width1 - max_width3)/max_width3 <= change and (max_width1 - max_width3)/max_width3 >= -1*change: #額頭=下顎
                faceshape = "菱形臉" if jaw == '尖' else "長臉"
            else:
                faceshape = '長臉'
        else: #長寬比不明顯
            if (max_width1 - max_width3)/max_width3 <= change and (max_width1 - max_width3)/max_width3 >= -1*change: #額頭=下顎
                faceshape = "鵝蛋臉" if jaw == '尖' else "方臉" if jaw == '方' else "圓臉"
            elif max_width1 > max_width3: #額頭>下顎
                faceshape = "鵝蛋臉" if jaw == '尖' else "圓臉"
            else: #額頭<下巴
                faceshape = "方臉"
    else:#下顎最長
        faceshape = "長臉" if key2 == 1 else "方臉"
    
    #########################  output  ############################
    
    if key1_sorted[0][0] == 1: 
        print("額頭最長")
    elif key1_sorted[0][0] == 2:
        print("臉頰最長")
    else:
        print("下顎最長")
      
    #print(key1_sorted)
    if key2 == 1:
        print("長寬比明顯")
    else:
        if key2 == 2:
            print("黃金比例")
        print("長寬比不明顯")
    print(jaw)
    
    ###########################   distance of eyes   ############################   (-1:略短, 0:正常，黃金比例, 1:略長)
    
    lenLeye = dis(df, 36, 39) #左眼長度
    lenReye = dis(df, 42, 45) #右眼長度
    lenEye = (lenLeye + lenReye) / 2 #眼睛平均長度
    
    disEyes = dis(df, 39, 42) #眼距(p39和p42的距離)
    
    if disEyes < lenEye:
        ansDisEyes = -1
    elif disEyes == lenEye:
        ansDisEyes = 0
    else:
        ansDisEyes = 1
        
    print("眼睛長度：",lenEye,"左眼：",lenLeye, "右眼：", lenReye," 眼距：", disEyes, "眼距/眼長：", disEyes/lenEye)
    print("x1: ", df.iat[0,332], ",y1: ", df.iat[0,400])
    print("x2: ", df.iat[0,335], ",y2: ", df.iat[0,403])
    print("x3: ", df.iat[0,338], ",y3: ", df.iat[0,406])
    print("x4: ", df.iat[0,341], ",y4: ", df.iat[0,409])

    #############################################################################

    
    return faceshape, key2, key1_sorted, ansDisEyes #key2==2(黃金比例)


# In[ ]:


## from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import os
import shutil
from flask_wtf.csrf import CSRFProtect
import requests

numToface = ['額頭','臉頰','下顎']
app = Flask(__name__)
app.config['SECRET_KEY'] = '5k4g4ji3ap75j0 wu62k7au4a83'  # 設定密鑰
csrf = CSRFProtect(app)

# 設定上傳圖片的資料夾
UPLOAD_FOLDER = 'C:\\Users\\user\\Desktop\\testFlask\\static\\uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 清空指定資料夾
def clear_output_folder(folder_path):
    if os.path.exists(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

# 允許的圖片格式
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 在啟動應用時，檢查並創建 uploads 目錄
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/index')
def index_main():
    #return render_template('index7-3.html')
    #return render_template('index7-4.html') #加入navbar
    #return render_template('index.html') #導到主頁
    return render_template('home2.html') #導到主頁

@app.route('/')
def index():
    #return render_template('index7-3.html')
    #return render_template('index7-4.html') #加入navbar
    #return render_template('index7-6.html') #加入 陰影、圓角、背景顏色堆疊
    return render_template('index7-7.html')  #test 加入觸控功能


@app.route('/upload', methods=['POST'])
def upload_image():
    output_folder = "D:\\專題\\all\\output"
    clear_output_folder(output_folder)
    
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}),400

    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}),400

    if file and allowed_file(file.filename):
        
        # 生成一個唯一的文件名，使用 UUID
        filename = str(uuid.uuid4()) + ".jpg"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        #filename = secure_filename(file.filename)
        #filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        #file.save(filepath)
        
        #跑openface
        run_openface(filepath, output_folder)
        
        #取出額頭接近點
        df_Openface = getOpenface(output_folder, filename)
        print(type(df_Openface))
        tmpForehead = [[df_Openface.iat[0,313]-5, df_Openface.iat[0,383]],[df_Openface.iat[0,323], df_Openface.iat[0,383]-(df_Openface.iat[0,397]-df_Openface.iat[0,391])],[df_Openface.iat[0,322]+5, df_Openface.iat[0,388]]]
        
        print(tmpForehead)
        return jsonify({'image_path': f'/static/uploads/{filename}', 'filepath': filepath, 'forehead':tmpForehead})
    
    return jsonify({'error': 'File type not allowed'}),400

@app.route('/take_photo')
def take_photo():
    output_folder = "D:\\專題\\all\\output"
    clear_output_folder(output_folder)
    return render_template('take_photo.html')

@app.route('/coords', methods=['POST'])
def receive_coords():
    #print("print ", df_Openface)
    print("here")
    data = request.json
    if not data:
        return jsonify({'error': 'Invalid JSON data'}), 400  # 確保有處理無效的 JSON 請求
    coords = data['coords']
    #print("Before coords : ",coords)
    print("Received coordinates:", coords)


    coords = sorted(coords, key=lambda coord: coord['x'])

    print("After coords : ",coords)
    image_path = data['image_path']
    
    # 下載圖片
    response = requests.get(image_path)

    # 確認下載成功
    if response.status_code == 200:
        # 將圖片保存到本地，這裡將文件名定義為 "photo.jpg"
        with open("D:\\專題\\all\\input\\photo.jpg", "wb") as f:
            f.write(response.content)
        #print("圖片下載成功！")
    else:
        print(f"下載失敗，狀態碼：{response.status_code}")
    
    input_file = "D:\\專題\\all\\input\\photo.jpg"
    output_dir = "D:\\專題\\all\\output"

    try:
        #run_openface(input_file, output_dir)
        file_path = AddForeheadPt(output_dir, coords)
        BuildRadianFile(file_path)
        jaw = getJaw()
        faceshape, goldenFace, face_lenth_sorted, disOfEyes = getFaceshape(file_path, jaw)
    except Exception as e:
        print(f"發生錯誤: {e}")
        return jsonify({'status': 'error', 'message': '處理失敗'})
    return jsonify({'status': 'success', 'faceshape': faceshape, 'goldenFace': goldenFace, 'maxLenth':numToface[face_lenth_sorted[0][0]-1], 'midLenth':numToface[face_lenth_sorted[1][0]-1], 'minLenth':numToface[face_lenth_sorted[2][0]-1], 'jaw':jaw, 'disOfEyes':disOfEyes})

if __name__ == '__main__':
    #app.run(debug=True, use_reloader=False)
    app.run(host='0.0.0.0', port=5000, debug=True)


# In[ ]:





# In[ ]:




