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
        noise_input = GaussianNoise(np.sqrt(noise))(input_layer) # 加入高斯噪聲層，用於模擬數據的隨機變異。屬於正則化技術，而非數據擴充（Data Augmentation）。np.sqrt(noise) 表示噪聲的標準差。
        dense = TimeDistributed(
            Dense(
                10,
                kernel_regularizer=regularizers.l2(0.01), # 正則化，減少模型的過度擬合。
                kernel_initializer=initializers.glorot_uniform(seed=0), # 使用 Glorot 均勻初始化方法對權重進行初始化，有助於提高模型的收斂速度和穩定性，並且有效減少梯度消失或梯度爆炸問題。
                bias_initializer=initializers.Zeros() # 將偏置初始化為 0。
            )
        )(noise_input)

    else:
        dense = TimeDistributed(
            Dense( 
                10, # 輸出單元數為10的全連接層。
                kernel_regularizer=regularizers.l2(0.01), # 正則化，減少模型的過度擬合。
                kernel_initializer=initializers.glorot_uniform(seed=0), # 使用 Glorot 均勻初始化方法對權重進行初始化，有助於提高模型的收斂速度和穩定性。
                bias_initializer=initializers.Zeros() # 將偏置初始化為 0。
            )
        )(input_layer)

    lstm1 = LSTM(
        60,
        return_sequences=True,
        kernel_regularizer=regularizers.l2(0.01), # 正則化，減少模型的過度擬合。
        kernel_initializer=initializers.glorot_uniform(seed=0), # 使用 Glorot 均勻初始化方法對權重進行初始化，有助於提高模型的收斂速度和穩定性。
        recurrent_initializer=initializers.Orthogonal(seed=0), # 將 LSTM 的遞歸權重初始化為正交矩陣，以促進梯度穩定。
        bias_initializer=initializers.Zeros() # 將偏置初始化為 0。
    )(dense)
    lstm1 = BatchNormalization()(lstm1) # 正規化，穩定訓練過程、加速收斂，並提高模型的泛化能力。

    lstm2 = LSTM(
        60,
        return_sequences=False,
        kernel_regularizer=regularizers.l2(0.01), # 正則化，減少模型的過度擬合。
        kernel_initializer=initializers.glorot_uniform(seed=0), # 使用 Glorot 均勻初始化方法對權重進行初始化，有助於提高模型的收斂速度和穩定性。
        recurrent_initializer=initializers.Orthogonal(seed=0), # 將 LSTM 的遞歸權重初始化為正交矩陣，以促進梯度穩定。
        bias_initializer=initializers.Zeros() # 將偏置初始化為 0。
    )(lstm1)
    lstm2 = BatchNormalization()(lstm2) # 正規化，穩定訓練過程、加速收斂，並提高模型的泛化能力。

    output_layer = Dense(
        1,
        activation='sigmoid', # 激活函數為sigmoid，適合輸出一個範圍在0到1之間的預測結果。
        kernel_regularizer=regularizers.l2(0.01), # 正則化，減少模型的過度擬合。
        kernel_initializer=initializers.glorot_uniform(seed=0), # 使用 Glorot 均勻初始化方法對權重進行初始化，有助於提高模型的收斂速度和穩定性。
        bias_initializer=initializers.Zeros() # 將偏置初始化為 0。
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

L2 正則化 
目的：減少過度擬合，也就是讓模型不過於貼合訓練數據，使它能對新數據有更好的表現。
原理：L2 正則化會讓模型的權重（即每個輸入對預測結果的影響程度）保持較小，以便更簡單、更平滑地適應數據。
作法：L2 正則化通過在損失函數中加入一項與權重的平方和成比例的懲罰項來抑制模型的複雜度，從而減少過度擬合風險。
如何實現：它會把每個權重值的平方乘上一個小係數（如 0.01），然後把這些結果加到損失函數中。這會讓模型更偏好小權重，從而減少過度擬合的風險。

Orthogonal 初始化
目的：初始化模型權重，使模型在一開始訓練時更穩定。
原理：Orthogonal 初始化會讓模型的初始權重彼此間保持「正交」，也就是說，它們的方向完全不相關。這樣能確保訊號在傳遞過程中不會過於減弱或變得過強，這對於 RNN 等時間序列模型特別有幫助。
如何實現：在模型開始訓練前，為權重分配一組特殊的初始值，使它們的方向是彼此「垂直」的（數學上稱之為「正交矩陣」）。

Glorot 均勻初始化方法 是什麼？
目的：確保模型一開始的權重不會讓數據信號在傳遞過程中爆炸或消失。
原理：Glorot 均勻初始化（也叫 Xavier 初始化）會根據輸入與輸出的神經元數量，選擇一個適當的範圍，從中均勻分布地選取權重值。這樣可以保持訊號穩定，避免模型在訓練初期就出現梯度爆炸或消失的問題。
如何實現：模型的權重會隨機從 [-limit, limit] 的範圍中選取，其中 limit 是基於該層的輸入和輸出單元數計算得來的。

BatchNormalization (批次正規化) 是什麼？
目的：加速訓練，並讓模型更穩定。
原理：批次正規化會在每一層的輸出上進行「標準化」，也就是把輸出調整成一個更穩定的範圍（通常平均值為 0，標準差為 1）。這樣做可以讓模型更快找到最佳解，減少訓練過程中的波動。
如何實現：模型會根據當前的批次數據，計算出每層輸出的平均值和標準差，然後把每個輸出都調整到該範圍內。這樣做可以平衡輸出的大小，提升模型的穩定性和收斂速度。
'''
