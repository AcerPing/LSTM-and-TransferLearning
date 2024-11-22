from os import makedirs, path, listdir
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def comparison(out_dir, train_mode): 
    '''
    比較針對不同條件使用和不使用遷移學習訓練的模型的均方誤差（MSE）值、顯示不同數據集上的遷移學習表現差異。
    '''

    # make output base directory (建立輸出目錄)
    base_out_dir = path.join(out_dir, train_mode)
    makedirs(base_out_dir, exist_ok=True)

    source_relative_path = path.join(out_dir, 'pre-train')
    source_dir = [f for f in listdir(source_relative_path) if path.isdir(path.join(source_relative_path, f))]
    target_relative_path = path.join(out_dir, 'transfer-learning')
    target_dir = [f for f in listdir(target_relative_path) if path.isdir(path.join(target_relative_path, f))]
    
    no_tl, tl = [], []
    for target in target_dir:
        
        # fetch results without transfer learning (收集未使用遷移學習訓練的模型的MSE值。)
        with open(path.join(out_dir, 'without-transfer-learning', target, 'log.txt'), 'r') as f: # 打開目錄中未使用遷移學習訓練的模型的 log.txt 檔。
            base_mse = float(f.readlines()[0].split(':')[1].lstrip(' ')) # 提取MSE值，並將其轉換為浮點數。
            no_tl.append(base_mse)
        print(f'{target}: ({base_mse})')
        
        # fetch results as row(1×sources) with transfer learning (獲取遷移學習的MSE值)
        row = []
        for source in source_dir:
            with open(path.join(target_relative_path, target, source, 'log.txt'), 'r') as f: # 打開每個相應的log.txt檔。
                mse = float(f.readlines()[0].split(':')[1].lstrip(' ')) # 提取MSE值，並將其轉換為浮點數。
                print('{}:{:.1f} ({})'.format(source, (1 - mse / base_mse) * 100, mse)) # 計算相對改進(MSE改進百分比)：（1 - mse / base_mse） * 100 （百分比）。
                row.append(mse)
        print()
        row.append(base_mse)
        tl.append(row)
    print('※ MSE value in () \n')
    
    # 將結果保存為 DataFrame
    tl = pd.DataFrame(np.array(tl), columns=source_dir+['base'], index=target_dir)
    tl.to_csv(path.join(base_out_dir, 'mse.csv'), index=True)

    # 計算轉移學習（Transfer Learning）的改進率，並可視化結果。
    metrics_map = (1 - tl.divide(no_tl, axis=0)) * 100 # 計算MSE的改進比例（減少了多少比例的 MSE）。
    metrics_map.to_csv(path.join(base_out_dir, 'improvement.csv'), index=True) # 保存結果
    
    # 繪製熱圖
    sns.set(font_scale=2.0) # 設定字體大小
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.heatmap(metrics_map.T, annot=True, fmt="1.1f", cmap="Blues",
                cbar_kws={'label': 'mse decreasing rate after transfer [%]'}, annot_kws={"size": 20}) # 繪製熱圖（Heatmap），來源模型變為橫軸，目標模型變為縱軸。
    plt.xticks(rotation=20) # 旋轉橫軸標籤20度。
    plt.yticks(rotation=20) # 旋轉縱軸標籤20度。
    plt.tight_layout() # 自動調整圖形中子圖或元素之間的間距，以防止重疊。
    ax.set_ylim(metrics_map.T.shape[0], 0) # 調整縱軸範圍，避免圖形被裁剪。
    plt.savefig(path.join(base_out_dir, 'metrics.png')) # 保存圖像
    
    # 根據改進率排序並保存排名
    debutanizer = sorted(metrics_map.iloc[0, :].to_dict().items(), key=lambda x: x[1], reverse=True) # 結果是一個按改進率降序排列的 (source_model, improvement) 的列表。
    debutanizer = np.array([data[0] for data in debutanizer]).reshape(-1, 1) # 將排序結果轉換為一列的 2D 陣列。
    sru = sorted(metrics_map.iloc[1, :].to_dict().items(), key=lambda x: x[1], reverse=True)
    sru = np.array([data[0] for data in sru]).reshape(-1, 1)
    rank = pd.DataFrame(np.concatenate([debutanizer, sru],axis=1), columns=['debutanizer', 'sru'])
    rank.to_csv(path.join(base_out_dir, 'rank.csv'), index=False)

    '''
    mse.csv：記錄了每個來源模型在不同目標模型上的 MSE 值以及基線模型的 MSE。
    improvement.csv：包含每個來源模型對不同目標模型的MSE改進率（百分比）。
    metrics.png：熱圖，可視化改進率。
    rank.csv：列出來源模型的改進率排名。
    '''
