from matplotlib import rcParams
from matplotlib import font_manager
import matplotlib.pyplot as plt

# 設定 Noto Sans CJK 字體為默認字體
# 動態加載字體路徑
font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
chinese_font = font_manager.FontProperties(fname=font_path)

# 設置全局字體
rcParams['font.sans-serif'] = ['Noto Sans CJK SC']
rcParams['font.family'] = chinese_font.get_name()
rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題
print(f"font.family: {rcParams['font.family']}, font.sans-serif:{rcParams['font.sans-serif']}")

# 測試繪圖
plt.figure(figsize=(8, 6))
plt.plot([1, 2, 3], [4, 5, 6], label='數據線')
plt.xlabel('特徵非類似度 / -', fontweight='bold') # 設定X軸名稱。
plt.ylabel('MSE / -', fontweight='bold') # 設定Y軸名稱。
plt.title('特徵相似性與MSE的關係圖')
plt.legend()
plt.savefig('test_output.png', bbox_inches='tight', dpi=300)
# plt.show()