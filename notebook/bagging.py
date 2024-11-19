from os import path, listdir, makedirs
import sys
sys.path.append('../') # 查找模組，是將上層目錄 (../) 添加到 Python 的模組搜索路徑中。
from utils.data_io import (
    read_data_from_dataset,
    ReccurentPredictingGenerator
)
import matplotlib.pyplot as plt
from keras.models import load_model
from tqdm import tqdm # 顯示進度條，看到任務的完成進度。
import numpy as np
from sklearn.metrics import mean_squared_error as mse

# 目錄設置
target = 'sru'
print(target)
write_out_dir = '../reports/result'
write_result_out_dir = path.join(write_out_dir, 'bagging', target)

# load dataset (載入數據集)
data_dir_path = path.join('..', 'dataset', 'target', target)
X_train, y_train, X_test, y_test = \
    read_data_from_dataset(data_dir_path)

period = (len(y_train) + len(y_test)) // 30
RPG = ReccurentPredictingGenerator(X_test, batch_size=1, timesteps=period) # 將batch_size設為1，逐條處理測試數據，記錄每筆數據的預測結果。
prediction = [] # 用於存儲每個模型的預測值

# 加載模型並生成預測
for path_model in tqdm(listdir(path.join(write_result_out_dir, 'model'))):
    file_path = path.join(write_result_out_dir, 'model', path_model)
    best_model = load_model(file_path)
    y_test_pred = best_model.predict_generator(RPG) # 獲取模型對測試數據的預測值。
    prediction.append(y_test_pred)

# 計算預測結果的準確度，並保存所有的預測數據。集成所有預測結果。
prediction = np.array(prediction) # 轉換為NumPy陣列，形狀為 (模型數量, 測試數據大小, 預測特徵數=1)。 # 單變量預測結果
print(f'prediction.shape: {prediction.shape}')
list_score = [] # 存儲每次計算的均方誤差（MSE）。
size_test = prediction.shape[1] # 測試數據的樣本數量。
y_test = y_test[-size_test:] # 將 y_test 修正為與 size_test 長度相同，確保測試標籤與預測數據對齊。
for i_prediction in range(prediction.shape[0]): # [:1]若僅只包含第0個模型的預測結果。
    pred = np.mean(prediction[:i_prediction + 1], axis=0) # 取第0個模型到第i_prediction個模型的預測，並取平均值作為集成預測結果，即集成預測值。
    print(f'y_test.shape: {y_test.shape}, pred.shape: {pred.shape}')
    accuracy = mse(y_test, pred.flatten()) # 與 y_test 的形狀一致
    list_score.append(accuracy)

np.save( path.join(write_result_out_dir, target), prediction) # 將所有的預測結果保存為NumPy檔案(.npy)

plt.rcParams['font.size'] = 25 # 設定字體大小
plt.figure(figsize=(15, 7)) # 建立圖表
print(f'總共有 {len(list_score)} 個預測的準確率。')
print(f'list_score: {list_score}')
plt.plot(list_score) # 繪製折線圖
plt.xlabel('the number of subsets / -') # 為X軸添加標籤，表示子集數量。
plt.ylabel('MSE / -') # 為Y軸添加標籤，表示均方誤差（MSE）。
plt.savefig( path.join(write_result_out_dir, target, 'bagging_sru') ) # 保存圖表

print('おしまい')