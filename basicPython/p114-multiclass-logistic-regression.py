import numpy as np
from sklearn.unils import shuffle
M = 2     # 入力データの次元
K = 3     # クラス数
n = 100   # クラスごとのデータ数
N = n * K # 全データ数
x1 = np.random.randn(n,M)