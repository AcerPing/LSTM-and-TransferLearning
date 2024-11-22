from os import listdir, path
import pickle

from dtw import dtw # 計算動態時間規劃距離（Dynamic Time Warping）。
from sklearn.metrics import mean_squared_error as mse
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import rcParams
from matplotlib import font_manager
import matplotlib.pyplot as plt
# import seaborn as sns


def dataset_idx_vs_improvement(out_dir, train_mode, diff_cal=None): # feature_make
    '''
    比較各數據與目標資料集之間的相似性，
    本程式碼通過 動態時間規劃 和 曼哈頓距離 測量資料序列之間的相似程度。
    結果中數值越小，代表序列越相似；數值越大，代表差異越大。
    '''
    # matplotlib 設定
    # sns.set(font_scale=1.4, font="Times New Roman") # 設定 Seaborn 圖表中的字體和字體大小。
    # sns.set_style("ticks", {'font.family':'serif', 'font.serif':'Times New Roman'}) # 設定 Seaborn 圖表的樣式和字體。
    plt.rcParams["xtick.direction"] = "in" # 控制Matplotlib圖表的x軸刻度線方向。"in"：刻度線指向圖表內部。
    plt.rcParams["ytick.direction"] = "in" # 控制Matplotlib圖表的y軸刻度線方向。"in"：刻度線指向圖表內部。

    # matplotlib 字體設定
    # matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP' # 設置日文字體，例如 Noto Sans CJK JP
    # 設定中文字體，例如 Noto Sans CJK 字體為默認字體
    font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    chinese_font = font_manager.FontProperties(fname=font_path)
    rcParams['font.family'] = ['Noto Sans CJK SC', chinese_font.get_name()] + rcParams['font.family']
    rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題
    rcParams['font.sans-serif'] = ['Noto Sans CJK SC'] + rcParams['font.sans-serif'] # 其他備選字體

    # load dataset (載入數據集)
    source_path = './dataset/source/'
    data_dict = {}
    for d_name in listdir(source_path):
        with open(path.join(source_path, d_name, 'y_test.pkl'), 'rb') as f:
            data = pickle.load(f)
        data_dict[d_name] = data # 數據集以字典形式存儲，鍵為數據集名稱，值為數據。
    
    target_path = './dataset/target/'
    for d_name in listdir(target_path):
        with open(path.join(target_path, d_name, 'y_test.pkl'), 'rb') as f:
            data = pickle.load(f)
        data_dict[d_name] = data # 數據集以字典形式存儲，鍵為數據集名稱，值為數據。

    # feature make # ? 未定義feature_make函數。
    # for key in data_dict.keys():
    #     data_dict[key] = feature_make(data_dict[key]) # NameError: name 'feature_make' is not defined

    # calculate difference
    dataset_index = pd.DataFrame(columns=['sru', 'debutanizer'], index=listdir(source_path))

    # 計算數據集的相似性指數, 比較各數據集與目標資料集的相似性。
    for key, value in data_dict.items():
        if key == 'sru' or key == 'debutanizer': continue
        print(f'比較{key}的數據相似程度。')
        if not diff_cal:
            print('please select how to calculate difference between dataset indices')
        if diff_cal == 'DTW': # Dynamic Time Warping 衡量兩個時間序列相似度的演算法，透過非線性地調整時間軸，對齊兩個序列。
            manhattan_distance = lambda x, y: np.abs(x - y) # 設定曼哈頓距離作為 DTW 的距離測量方法 # -- "歐幾里得距離" vs. "曼哈頓距離"
            dataset_index.at[key, 'sru'] = dtw(data_dict['sru'], data_dict[key], manhattan_distance).distance #-- "distance：DTW 距離值" vs. "path：最佳對齊路徑"
            dataset_index.at[key, 'debutanizer'] = dtw(data_dict['debutanizer'], data_dict[key], manhattan_distance).distance
        elif diff_cal == 'mse': # !X 均方誤差需要兩個數據的形狀完全一致才能計算。
            dataset_index.at[key, 'sru'] = mse(data_dict['sru'], data_dict[key])
            dataset_index.at[key, 'debutanizer'] = mse(data_dict['debutanizer'], data_dict[key])
        else:
            print('please select how to calculate difference between dataset indices')
        
    # plot dataset_idx vs improvement
    base_out_dir = path.join(out_dir, train_mode)
    mse_df = pd.read_csv(path.join(base_out_dir, 'mse.csv'), index_col=0)
    improvement_df = pd.read_csv(path.join(base_out_dir, 'improvement.csv'), index_col=0)
    print(dataset_index)
    print(improvement_df)
    dataset_index.to_csv(path.join(base_out_dir, f'{diff_cal}特徵非類似度.csv'), index=True)

    # 比較有無遷移學習(転移学習あり 與 転移学習なし)的結果差異。X 軸：特徵非類似度，表示來源與目標資料集的相似性程度；Y 軸：MSE，表示預測效果的準確性。
    plt.figure(figsize=(20, 10))  # # 調整畫布大小以避免擠壓。 整體畫布寬度 12 英吋，高度 6 英吋
    for idx, target in enumerate(listdir(target_path)):
        plt.subplot(1, 2, idx + 1) # 創建 1 行 2 列的子圖
        x, y = [], []
        for source in listdir(source_path):
            x.append(dataset_index.at[source, target]*10**7) # x：來源資料集與目標資料集的特徵非類似度。特徵非類似度被乘以 10^7 進行放大，以更好地顯示數據。
            y.append(mse_df.at[target, source]) # y：來源資料集與目標資料集的MSE。
            
        plt.plot(x, y, 'b.', markersize=20, label='with transfer', color='blue') # 藍色點表示每個來源資料集的特徵非類似度與MSE，表示使用遷移學習的結果。
        # 辨識每個資料點對應的來源資料集
        for (i,j,k) in zip(x,y,listdir(source_path)):
            plt.annotate(k, xy=(i, j),  xytext=(5, 5), textcoords="offset points", fontsize=8, color='black', arrowprops=dict(arrowstyle='-', color='gray'))
        x_range = max(x) - min(x) # 計算特徵非類似度的範圍，用於繪製基線。
        x_min = min(x) - 0.1 * x_range
        x_max = max(x) + 0.1 * x_range
        plt.plot([x_min, x_max], [mse_df.at[target, 'base'] for _ in range(2)], linestyle='dashed', label='without transfer', color='black') # 無遷移學習情況下的MSE，繪製為虛線。
        plt.xlabel('特徵非類似度 / -', fontweight='bold') # 設定X軸名稱。
        plt.ylabel('MSE / -', fontweight='bold') # 設定Y軸名稱。
        target = target.capitalize() if target == 'debutanizer' else target.upper()
        plt.title(f'{target}')
        plt.legend(loc='best') # 顯示標籤資訊。
    plt.tight_layout() # ：自動調整子圖間距，避免重疊。
    plt.savefig( path.join(base_out_dir, '特徵相似性與MSE的關係圖'), bbox_inches='tight' ) # 保存圖表

    # plot dataset_idx rank vs improvement (繪製相似性排序與改進程度的關係圖)
    plt.figure(figsize=(20, 10))  # # 調整畫布大小以避免擠壓。 整體畫布寬度 12 英吋，高度 6 英吋
    for idx, target in enumerate(listdir(target_path)):
        plt.subplot(1, 2, idx + 1)
        sources_sorted = dataset_index[target].sort_values().keys().tolist()  # 獲取排序後的 source 名稱列表
        improvement_list = [improvement_df.at[target, source] for source in sources_sorted]  
        n = len(improvement_list)
        plt.plot(range(1, n + 1), improvement_list, 'b', label='with transfer')
        plt.plot(range(1, n + 1), [0 for _ in range(len(improvement_list))], 'r', label='without transfer', linestyle='dashed') 
        # 在每個點上標註數據集名稱
        for rank, (improvement, source) in enumerate(zip(improvement_list, sources_sorted), start=1):
            plt.annotate(
                source,  # 要標註的文字
                xy=(rank, improvement),  # 點的座標
                xytext=(5, 5),  # 偏移量
                textcoords='offset points',  # 偏移基於點的座標系
                fontsize=8,  # 字體大小
                color='black'  # 文字顏色
            )
        plt.xlabel('Feature Similarity Rank / -', fontweight='bold')
        plt.ylabel('Improvement / %', fontweight='bold')
        plt.yticks([i*50 for i in range(-3,4)])
        plt.legend(loc='best')
        target = target.capitalize() if target == 'debutanizer' else target.upper()
        plt.title(f'({"ab"[idx]}) {target}')
    plt.tight_layout()
    plt.savefig( path.join(base_out_dir, '特徵相似性與MSE改進程度的關係圖'), bbox_inches='tight' ) # 保存圖表
