import random
import argparse
import json
import math
from os import path, getcwd, makedirs, environ, listdir

import tensorflow as tf
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau # --, EarlyStopping
from recombinator.optimal_block_length import optimal_block_length
from recombinator.block_bootstrap import circular_block_bootstrap # 用於對具有時間依賴性的數據進行重抽樣，在不打破數據時間依賴性的情況下生成新的數據。採用的是循環抽樣的方式，這意味著當抽樣到序列尾端時，可以回到序列開頭繼續抽樣。

from utils.model import build_model
from utils.data_io import (
    read_data_from_dataset,
    ReccurentTrainingGenerator,
    ReccurentPredictingGenerator
)
from utils.save import save_lr_curve, save_prediction_plot, save_yy_plot, save_mse
from utils.device import limit_gpu_memory


def parse_arguments():
    ap = argparse.ArgumentParser(
        description='Time-Series Regression by LSTM through transfer learning') # 表示該程式的用途是透過遷移學習使用 LSTM 進行時間序列回歸分析。
    # for dataset path
    ap.add_argument('--out-dir', '-o', default='result',
                    type=str, help='path for output directory') # 指定輸出目錄的路徑，預設值為 result。
    # for model
    ap.add_argument('--seed', type=int, default=1234,
                    help='seed value for random value, (default : 1234)') # 確保隨機操作（如資料分割、模型初始化等）在每次執行中一致，方便實驗重現性。
    ap.add_argument('--train-ratio', default=0.8, type=float,
                    help='percentage of train data to be loaded (default : 0.8)') # 指定訓練集比例為 0.8（即 80%）。數據集會依據此比例分割為訓練集和測試集或驗證集。
    ap.add_argument('--time-window', default=1000, type=int,
                    help='length of time to capture at once (default : 1000)') # 設定時間窗口為 1000。這可能代表模型在一次處理過程中觀察的資料長度或時間範圍。
    # for training
    ap.add_argument('--train-mode', '-m', default='pre-train', type=str,
                    help='"pre-train", "transfer-learning", "without-transfer-learning", \
                            "bagging", "noise-injection", "score" (default : pre-train)') # 設定模式
    ap.add_argument('--gpu', action='store_true',
                    help='whether to do calculations on gpu machines (default : False)') # 是否啟用GPU加速
    ap.add_argument('--nb-epochs', '-e', default=1, type=int,
                    help='training epochs for the model (default : 1)') # 設定訓練的epoch。（epoch是完整地使用所有訓練數據訓練模型的一次過程。）
    ap.add_argument('--nb-batch', default=20, type=int,
                    help='number of batches in training (default : 20)') # 設定訓練過程中的批次數量，預設為 20。 批次大小（batch size） = 總訓練樣本數量 ÷ 批次數量（nb-batch）
    ap.add_argument('--nb-subset', default=10, type=int,
                    help='number of data subset in bootstrapping (default : 10)') # 在bootstrapping中設定資料子集的數量。
    ap.add_argument('--noise-var', default=0.0001, type=float,
                    help='variance of noise in noise injection (default : 0.0001)') # 在噪聲注入中設定噪聲的變異數。
    ap.add_argument('--valid-ratio', default=0.2, type=float,
                    help='ratio of validation data in train data (default : 0.2)') # 在訓練資料中設定驗證資料的比例。
    ap.add_argument('--freeze', action='store_true', 
                    help='whether to freeze transferred weights in transfer learning (default : False)') # 在遷移學習中凍結已轉移的權重。
    # for output
    ap.add_argument('--train-verbose', default=1, type=int,
                    help='whether to show the learning process (default : 1)') # 設定訓練過程中的輸出詳盡程度。
    args = vars(ap.parse_args())
    return args
    

def seed_every_thing(seed=1234): # 確保各種隨機操作（如資料分割、模型初始化等）在每次執行中產生相同的結果，從而提高實驗的可重現性。
    environ['PYTHONHASHSEED'] = str(seed) # 設定Python的雜湊隨機種子，確保Python的雜湊行為在每次執行時保持一致。
    np.random.seed(seed) # 設定NumPy的隨機種子，確保 NumPy 產生的隨機數在每次執行時相同。
    random.seed(seed) # 設定Python標準庫的隨機種子，確保Python標準庫中的隨機數生成器在每次執行時產生相同的結果。
    tf.random.set_random_seed(seed) # 設定TensorFlow的隨機種子，確保TensorFlow產生的隨機數在每次執行時一致。


def save_arguments(args, out_dir): # 旨在將參數字典 args 以 JSON 格式保存到指定的輸出目錄 out_dir 中
    path_arguments = path.join(out_dir, 'params.json')
    if not path.exists(path_arguments):
        with open(path_arguments, mode="w") as f:
            json.dump(args, f, indent=4)


def make_callbacks(file_path, save_csv=True):
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.000001) # 降低學習率，以促進模型更好地收斂。 # --verbose=1,
    model_checkpoint = ModelCheckpoint(filepath=file_path, monitor='val_loss', save_best_only=True) # 保存最佳模型。 # -- save_weights_only = True,
    if not save_csv:
        return [reduce_lr, model_checkpoint]
    csv_logger = CSVLogger(path.join(path.dirname(file_path), 'epoch_log.csv')) # 將每個訓練週期的損失和評估指標記錄到 CSV 文件中
    # early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True) # -- 建議可以增加EarlyStopping
    return [reduce_lr, model_checkpoint, csv_logger] # -- early_stopping
 

def main():

    # make analysis environment
    limit_gpu_memory() # 限制GPU記憶體使用量，避免因為分配過多而造成系統不穩定。
    args = parse_arguments() # 解析參數
    seed_every_thing(args["seed"]) # 設定隨機種子，在每次運行時產生一致的結果。
    write_out_dir = path.normpath(path.join(getcwd(), 'reports', args["out_dir"])) # 輸出文件的存放路徑
    makedirs(write_out_dir, exist_ok=True)
    
    print('-' * 140)
    
    print(f'train_mode: {args["train_mode"]}')
    if args["train_mode"] == 'pre-train': # 以預訓練模式執行模型訓練。
        
        for source in listdir('dataset/source'): # 逐個處理來源數據集

            # skip source dataset without pickle file
            data_dir_path = path.join('dataset', 'source', source)
            if not path.exists(f'{data_dir_path}/X_train.pkl'): continue
            
            # make output directory
            write_result_out_dir = path.join(write_out_dir, args["train_mode"], source)
            makedirs(write_result_out_dir, exist_ok=True)
            
            # load dataset
            X_train, y_train, X_test, y_test = \
                read_data_from_dataset(data_dir_path) # 讀取'X_train', 'y_train', 'X_test', 'y_test'資料
            period = (len(y_train) + len(y_test)) // 30 # period：表示時間步數（time steps），即模型在每次輸入中考慮的過去觀測值的數量。
            X_train = np.concatenate((X_train, X_test), axis=0)  # > no need for test data when pre-training
            y_train = np.concatenate((y_train, y_test), axis=0)  # > no need for test data when pre-training
            print(f'切分比例: {args["valid_ratio"]}')
            X_train, X_valid, y_train, y_valid =  \
                train_test_split(X_train, y_train, test_size=args["valid_ratio"], shuffle=False) # 不隨機打亂數據 (shuffle=False)
            print(f'\nSource dataset : {source}')
            print(f'\nX_train : {X_train.shape[0]}')
            print(f'\nX_valid : {X_valid.shape[0]}')
            
            # construct the model
            file_path = path.join(write_result_out_dir, 'best_model.hdf5') # 指定模型的保存路徑
            callbacks = make_callbacks(file_path) # 在訓練過程中保存最佳模型
            input_shape = (period, X_train.shape[1]) # (timesteps, features)，period表示時間步數，X_train.shape[1]為欄位特徵。
            model = build_model(input_shape, args["gpu"], write_result_out_dir)
            
            # train the model
            bsize = len(y_train) // args["nb_batch"] # 計算批次大小batch_size
            RTG = ReccurentTrainingGenerator(X_train, y_train, batch_size=bsize, timesteps=period, delay=1) # 創建訓練數據
            RVG = ReccurentTrainingGenerator(X_valid, y_valid, batch_size=bsize, timesteps=period, delay=1) # 創建驗證數據
            H = model.fit_generator(RTG, validation_data=RVG, epochs=args["nb_epochs"], verbose=1, callbacks=callbacks) # 訓練模型
            save_lr_curve(H, write_result_out_dir) # 保存每個epoch的學習曲線

            # clear memory up (清理記憶體並保存參數)
            keras.backend.clear_session() # 清理記憶體，釋放模型佔用的資源。
            save_arguments(args, write_result_out_dir) # 保存訓練參數 (args) 到結果輸出目錄中。
            print('\n' * 2 + '-' * 140 + '\n' * 2)
    
    elif args["train_mode"] == 'transfer-learning':
        
        for target in listdir('dataset/target'):
        
            # skip target in the absence of pickle file
            if not path.exists(f'dataset/target/{target}/X_train.pkl'): continue

            for source in listdir(f'{write_out_dir}/pre-train'):
                
                # make output directory
                write_result_out_dir = path.join(write_out_dir, args["train_mode"], target, source)
                pre_model_path = f'{write_out_dir}/pre-train/{source}/best_model.hdf5'
                if not path.exists(pre_model_path): continue
                makedirs(write_result_out_dir, exist_ok=True)
                    
                # load dataset
                data_dir_path = f'dataset/target/{target}'
                X_train, y_train, X_test, y_test = \
                    read_data_from_dataset(data_dir_path)
                period = (len(y_train) + len(y_test)) // 30
                X_train, X_valid, y_train, y_valid = \
                    train_test_split(X_train, y_train, test_size=args["valid_ratio"], shuffle=False)
                print(f'\nTarget dataset : {target}')
                print(f'\nX_train : {X_train.shape[0]}')
                print(f'\nX_valid : {X_valid.shape[0]}')
                print(f'\nX_test : {X_test.shape[0]}')
                
                # construct the model
                pre_model = load_model(pre_model_path)
                file_path = path.join(write_result_out_dir, 'transferred_best_model.hdf5')
                callbacks = make_callbacks(file_path)
                input_shape = (period, X_train.shape[1])
                model = build_model(input_shape, args["gpu"], write_result_out_dir, pre_model=pre_model, freeze=args["freeze"])
        
                # train the model
                bsize = len(y_train) // args["nb_batch"]
                RTG = ReccurentTrainingGenerator(X_train, y_train, batch_size=bsize, timesteps=period, delay=1)
                RVG = ReccurentTrainingGenerator(X_valid, y_valid, batch_size=bsize, timesteps=period, delay=1)
                H = model.fit_generator(RTG, validation_data=RVG, epochs=args["nb_epochs"], verbose=1, callbacks=callbacks)
                save_lr_curve(H, write_result_out_dir)
                
                # prediction
                best_model = load_model(file_path)
                RPG = ReccurentPredictingGenerator(X_test, batch_size=1, timesteps=period)
                y_test_pred = best_model.predict_generator(RPG)

                # save log for the model
                y_test = y_test[-len(y_test_pred):]
                save_prediction_plot(y_test, y_test_pred, write_result_out_dir)
                save_yy_plot(y_test, y_test_pred, write_result_out_dir)
                mse_score = save_mse(y_test, y_test_pred, write_result_out_dir, model=best_model)
                args["mse"] = mse_score
                save_arguments(args, write_result_out_dir)
                keras.backend.clear_session()
                print('\n' * 2 + '-' * 140 + '\n' * 2)
    
    elif args["train_mode"] == 'without-transfer-learning':

        for target in listdir('dataset/target'):
        
            # make output directory
            write_result_out_dir = path.join(write_out_dir, args["train_mode"], target)
            makedirs(write_result_out_dir, exist_ok=True)

            # load dataset
            data_dir_path = path.join('dataset', 'target', target)
            X_train, y_train, X_test, y_test = \
                read_data_from_dataset(data_dir_path)
            period = (len(y_train) + len(y_test)) // 30
            X_train, X_valid, y_train, y_valid =  \
                train_test_split(X_train, y_train, test_size=args["valid_ratio"], shuffle=False)
            print(f'\nTarget dataset : {target}')
            print(f'\nX_train : {X_train.shape[0]}')
            print(f'\nX_valid : {X_valid.shape[0]}')
            print(f'\nX_test : {X_test.shape[0]}')
            
            # construct the model
            file_path = path.join(write_result_out_dir, 'best_model.hdf5')
            callbacks = make_callbacks(file_path)
            input_shape = (period, X_train.shape[1])
            model = build_model(input_shape, args["gpu"], write_result_out_dir)
            
            # train the model
            bsize = len(y_train) // args["nb_batch"]
            RTG = ReccurentTrainingGenerator(X_train, y_train, batch_size=bsize, timesteps=period, delay=1)
            RVG = ReccurentTrainingGenerator(X_valid, y_valid, batch_size=bsize, timesteps=period, delay=1)
            H = model.fit_generator(RTG, validation_data=RVG, epochs=args["nb_epochs"], verbose=1, callbacks=callbacks)
            save_lr_curve(H, write_result_out_dir)

            # prediction
            best_model = load_model(file_path)
            RPG = ReccurentPredictingGenerator(X_test, batch_size=1, timesteps=period)
            y_test_pred = best_model.predict_generator(RPG)

            # save log for the model
            y_test = y_test[-len(y_test_pred):]
            save_prediction_plot(y_test, y_test_pred, write_result_out_dir)
            save_yy_plot(y_test, y_test_pred, write_result_out_dir)
            mse_score = save_mse(y_test, y_test_pred, write_result_out_dir, model=best_model)
            args["mse"] = mse_score
            save_arguments(args, write_result_out_dir)

            # clear memory up
            keras.backend.clear_session()
            print('\n' * 2 + '-' * 140 + '\n' * 2)

    elif args["train_mode"] == 'bagging':
    
        for target in listdir('dataset/target'):
            
            # make output directory
            write_result_out_dir = path.join(write_out_dir, args["train_mode"], target)
            makedirs(write_result_out_dir, exist_ok=True)

            # load dataset
            data_dir_path = path.join('dataset', 'target', target)
            X_train, y_train, X_test, y_test = \
                read_data_from_dataset(data_dir_path)
            period = (len(y_train) + len(y_test)) // 30

            # make subsets
            b_star = optimal_block_length(y_train)
            b_star_cb = math.ceil(b_star[0].b_star_cb)
            print(f'optimal block length for circular bootstrap = {b_star_cb}')
            subsets_y_train = circular_block_bootstrap(y_train, block_length=b_star_cb,
                                                       replications=args["nb_subset"], replace=True)
            subsets_X_train = []
            for i in range(X_train.shape[1]):
                np.random.seed(0)
                X_cb = circular_block_bootstrap(X_train[:, i], block_length=b_star_cb,
                                                replications=args["nb_subset"], replace=True)
                subsets_X_train.append(X_cb)
            subsets_X_train = np.array(subsets_X_train)
            subsets_X_train = subsets_X_train.transpose(1, 2, 0)

            # train the model for each subset
            model_dir = path.join(write_result_out_dir, 'model')
            makedirs(model_dir, exist_ok=True)
            for i_subset, (i_X_train, i_y_train) in enumerate(zip(subsets_X_train, subsets_y_train)):
                
                i_X_train, i_X_valid, i_y_train, i_y_valid = \
                    train_test_split(i_X_train, i_y_train, test_size=args["valid_ratio"], shuffle=False)
                
                # construct the model
                file_path = path.join(model_dir, f'best_model_{i_subset}.hdf5')
                callbacks = make_callbacks(file_path, save_csv=False)
                input_shape = (period, i_X_train.shape[1])  # x_train.shape[2] is number of variable
                model = build_model(input_shape, args["gpu"], write_result_out_dir, savefig=False)

                # train the model
                bsize = len(i_y_train) // args["nb_batch"]
                RTG = ReccurentTrainingGenerator(i_X_train, i_y_train, batch_size=bsize, timesteps=period, delay=1)
                RVG = ReccurentTrainingGenerator(i_X_valid, i_y_valid, batch_size=bsize, timesteps=period, delay=1)
                H = model.fit_generator(RTG, validation_data=RVG, epochs=args["nb_epochs"], verbose=1, callbacks=callbacks)
            
            keras.backend.clear_session()
            print('\n' * 2 + '-' * 140 + '\n' * 2)

    elif args["train_mode"] == 'noise-injection':

        for target in listdir('dataset/target'):
            
            # make output directory
            write_result_out_dir = path.join(write_out_dir, args["train_mode"], target)
            makedirs(write_result_out_dir, exist_ok=True)

            # load dataset
            data_dir_path = path.join('dataset', 'target', target)
            X_train, y_train, X_test, y_test = \
                read_data_from_dataset(data_dir_path)
            period = (len(y_train) + len(y_test)) // 30
            X_train, X_valid, y_train, y_valid =  \
                train_test_split(X_train, y_train, test_size=args["valid_ratio"], shuffle=False)
            print(f'\nTarget dataset : {target}')
            print(f'\nX_train : {X_train.shape}')
            print(f'\nX_valid : {X_valid.shape}')
            print(f'\nX_test : {X_test.shape[0]}')

            # construct the model
            file_path = path.join(write_result_out_dir, 'best_model.hdf5')
            callbacks = make_callbacks(file_path)
            input_shape = (period, X_train.shape[1])
            model = build_model(input_shape, args["gpu"], write_result_out_dir, noise=args["noise_var"])

            # train the model
            bsize = len(y_train) // args["nb_batch"]
            RTG = ReccurentTrainingGenerator(X_train, y_train, batch_size=bsize, timesteps=period, delay=1)
            RVG = ReccurentTrainingGenerator(X_valid, y_valid, batch_size=bsize, timesteps=period, delay=1)
            H = model.fit_generator(RTG, validation_data=RVG, epochs=args["nb_epochs"], verbose=1, callbacks=callbacks)
            save_lr_curve(H, write_result_out_dir)

            # prediction
            best_model = load_model(file_path)
            RPG = ReccurentPredictingGenerator(X_test, batch_size=1, timesteps=period)
            y_test_pred = best_model.predict_generator(RPG)

            # save log for the model
            y_test = y_test[-len(y_test_pred):]
            save_prediction_plot(y_test, y_test_pred, write_result_out_dir)
            save_yy_plot(y_test, y_test_pred, write_result_out_dir)
            mse_score = save_mse(y_test, y_test_pred, write_result_out_dir, model=best_model)
            args["mse"] = mse_score
            save_arguments(args, write_result_out_dir)

            # clear memory up
            keras.backend.clear_session()
            print('\n' * 2 + '-' * 140 + '\n' * 2)


if __name__ == '__main__':
    main()
