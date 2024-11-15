from os import path, listdir
import json

def read_arguments(out_dir): 
    """從指定目錄讀取 'params.json' 檔案，並返回解析後的參數字典。

    Args:
        out_dir (str): 包含 'params.json' 檔案的目錄路徑。

    Returns:
        dict: 解析後的參數字典。
    """
    path_arguments = path.join(out_dir, 'params.json')
    if path.exists(path_arguments):
        with open(path_arguments, mode="r") as f:
            args = json.load(f)
        return args

out_dir = r'./reports/result/pre-train/traffic'
args = read_arguments(out_dir)
args
print(args)
