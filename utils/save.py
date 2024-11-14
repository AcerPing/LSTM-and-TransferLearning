from os import path

from sklearn.metrics import mean_squared_error as mse
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.size"] = 13


def save_lr_curve(H, out_dir: str, f_name=None):
    """save learning curve in deep learning

    Args:
        model : trained model (keras)
        out_dir (str): directory path for saving
    """
    f_name = 'learning_curve' if not f_name else f_name # 檔名
    plt.figure(figsize=(18, 5)) # 建立圖表
    plt.rcParams["font.size"] = 18 # 字體大小為 18。
    plt.plot(H.history['loss']) # 繪製訓練損失曲線
    plt.plot(H.history['val_loss']) # 繪製驗證損失曲線
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'valid'], loc='upper right')
    plt.savefig(path.join(out_dir, f'{f_name}.png'))


def save_prediction_plot(y_test_time: np.array, y_pred_test_time: np.array, out_dir: str):
    """save prediction plot for tareget varibale

    Args:
        y_test_time (np.array): observed data for target variable # 實際值
        y_pred_test_time (np.array): predicted data for target variable # 預測值
        out_dir (str): directory path for saving # 保存目錄
    """
    plt.figure(figsize=(18, 5)) # 設定圖表大小
    plt.rcParams["font.size"] = 18 # 設置字體大小
    plt.plot([i for i in range(1, 1 + len(y_pred_test_time))], y_pred_test_time, 'r', label="predicted") # 繪製預測數據的紅色折線圖
    plt.plot([i for i in range(1, 1 + len(y_test_time))], y_test_time, 'b', label="measured", lw=1, alpha=0.3) # 繪製實際數據的藍色折線圖
    plt.ylim(0, 1) # 設置y軸的顯示範圍為0到1。
    plt.xlim(0, len(y_test_time)) # 設置x軸範圍，從0到實際數據的長度。
    plt.ylabel('Value') # 設置y軸標籤。
    plt.xlabel('Time') # 設置x軸標籤。
    plt.legend(loc="best") # 顯示圖例，並將圖例放在最佳位置（由 Matplotlib 自動確定）。
    plt.savefig(path.join(out_dir, 'prediction.png')) # 保存圖像


def save_yy_plot(y_test_time: np.array, y_pred_test_time: np.array, out_dir: str):
    """save yy plot for target variable

    Args:
        y_test_time (np.array): observed data for target variable # 實際值
        y_pred_test_time (np.array): predicted data for target variable # 預測值
        out_dir (str): directory path for saving # 保存目錄
    """
    plt.figure(figsize=(10, 10)) # 設定圖表大小
    plt.rcParams["font.size"] = 18 # 設置字體大小
    plt.plot(y_test_time, y_pred_test_time, 'b.') # 繪製預測值與實際值的散點圖，使用藍色點標記。
    diagonal = np.linspace(0, 1, 10000) # 繪製對角線
    plt.plot(diagonal, diagonal, 'r-') # 繪製一條紅色的對角線，表示理想狀況下的預測值與實際值相等。散點越接近這條線，表示預測越準確。
    plt.xlim(0, 1) # 設定x軸的範圍為0到1。
    plt.ylim(0, 1) # 設定y軸的範圍為0到1。
    plt.xlabel('Observed') # 設置x軸標籤。
    plt.ylabel('Predicted') # 設置y軸標籤。
    plt.savefig(path.join(out_dir, 'yy_plot.png')) # 保存圖像


# def save_yy_plot(y_test_time: np.array, y_pred_test_time: np.array, out_dir: str): # ! Duplicate 與上段重複
#     """save yy plot for target variable

#     Args:
#         y_test_time (np.array): observed data for target variable
#         y_pred_test_time (np.array): predicted data for target variable
#         out_dir (str): directory path for saving
#     """
#     plt.figure(figsize=(10, 10))
#     plt.rcParams["font.size"] = 18
#     plt.plot(y_test_time, y_pred_test_time, 'b.')
#     diagonal = np.linspace(0, 1, 10000)
#     plt.plot(diagonal, diagonal, 'r-')
#     plt.xlim(0, 1)
#     plt.ylim(0, 1)
#     plt.xlabel('Observed')
#     plt.ylabel('Predicted')
#     plt.savefig(path.join(out_dir, 'yy_plot.png'))


def save_mse(y_test_time: np.array, y_pred_test_time: np.array, out_dir: str, model=None):
    """save mean squared error for tareget variable

    Args:
        y_test_time (np.array): observed data for target variable
        y_pred_test_time (np.array): predicted data for target variable
        out_dir (str): directory path for saving
        model : trained model (keras)
    """
    accuracy = mse(y_test_time, y_pred_test_time) # 計算均方誤差
    with open(path.join(out_dir, 'log.txt'), 'w') as f: # 寫入文件
        f.write('accuracy : {:.6f}\n'.format(accuracy))
        f.write('=' * 65 + '\n')
        if model:
            model.summary(print_fn=lambda x: f.write(x + '\n')) # 將模型摘要資訊寫入文件。
    return accuracy


# TODO: 需要額外畫製曲線 -- 殘差圖（Residual Plot）、誤差直方圖（Error Histogram）
