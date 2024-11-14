from keras.layers import Input, Dense, BatchNormalization
from keras.layers.wrappers import TimeDistributed
from keras.layers.noise import GaussianNoise
from keras.models import Model
from keras.utils import plot_model
from keras.optimizers import Adam
from keras import initializers, regularizers
import numpy as np


def build_model(input_shape: tuple, # 模型的輸入形狀(timesteps, features)
                gpu, # 若為True則使用CUDA加速的LSTM層（CuDNNLSTM）
                write_result_out_dir,
                pre_model=None, # 若有傳入預訓練模型，則可以從中載入權重。
                freeze=False, # 若為True，會將部分層設為不可訓練，用於遷移學習。
                noise=None, # 若設定此參數，會加入一層高斯噪聲層，模擬數據變異。
                verbose=True,
                savefig=True):

    if gpu:
        from keras.layers import CuDNNLSTM as LSTM
    else:
        from keras.layers import LSTM

    # construct the model
    input_layer = Input(input_shape)
    
    print(f'是否加入噪聲: {noise}')
    if noise:
        noise_input = GaussianNoise(np.sqrt(noise))(input_layer) # 加入高斯噪聲層，用於模擬數據的隨機變異。
        dense = TimeDistributed(
            Dense(
                10,
                kernel_regularizer=regularizers.l2(0.01), # 正則化
                kernel_initializer=initializers.glorot_uniform(seed=0),
                bias_initializer=initializers.Zeros()
            )
        )(noise_input)

    else:
        dense = TimeDistributed(
            Dense(
                10,
                kernel_regularizer=regularizers.l2(0.01), # 正則化
                kernel_initializer=initializers.glorot_uniform(seed=0),
                bias_initializer=initializers.Zeros()
            )
        )(input_layer)

    lstm1 = LSTM(
        60,
        return_sequences=True,
        kernel_regularizer=regularizers.l2(0.01), # 正則化
        kernel_initializer=initializers.glorot_uniform(seed=0),
        recurrent_initializer=initializers.Orthogonal(seed=0),
        bias_initializer=initializers.Zeros()
    )(dense)
    lstm1 = BatchNormalization()(lstm1) # 正規化，穩定訓練過程。

    lstm2 = LSTM(
        60,
        return_sequences=False,
        kernel_regularizer=regularizers.l2(0.01), # 正則化
        kernel_initializer=initializers.glorot_uniform(seed=0),
        recurrent_initializer=initializers.Orthogonal(seed=0),
        bias_initializer=initializers.Zeros()
    )(lstm1)
    lstm2 = BatchNormalization()(lstm2) # 正規化，穩定訓練過程。

    output_layer = Dense(
        1,
        activation='sigmoid', # 激活函數為sigmoid，適合輸出一個範圍在0到1之間的預測結果。
        kernel_regularizer=regularizers.l2(0.01), # 正則化
        kernel_initializer=initializers.glorot_uniform(seed=0),
        bias_initializer=initializers.Zeros()
    )(lstm2)

    model = Model(inputs=input_layer, outputs=output_layer) # 建立模型
    if savefig:
        plot_model(model, to_file=f'{write_result_out_dir}/architecture.png', show_shapes=True, show_layer_names=False)
    
    # transfer weights from pre-trained model (遷移學習：載入預訓練模型權重)
    if pre_model:
        for i in range(2, len(model.layers) - 1):
            model.layers[i].set_weights(pre_model.layers[i].get_weights()) # 將對應層的權重從預訓練模型載入。
            if freeze: # 若 freeze=True，則將這些層設置為不可訓練（即權重不會在訓練中更新），這樣可以保持預訓練權重不變。
                model.layers[i].trainable = False
            
    model.compile(optimizer=Adam(), loss='mse', metrics=['accuracy']) # metrics是模型訓練過程中用來監控模型性能的指標、評估模型的訓練效果。 # --'mse','mae','mape','msle' 
    if verbose: print(model.summary())

    return model
    
'''
GaussianNoise 高斯噪聲層
好處：
1.) 防止過擬合：模型會學習到更多數據的變異性，而不是過度擬合訓練數據的特定模式。
2.) 增加泛化能力：模型在訓練時遇到更多不同的數據輸入，從而提高模型在測試數據上的性能。
3.) 模擬數據噪聲：幫助模型在面對真實世界的噪聲數據時表現得更好。
'''
