import numpy as np
rng = np.random.RandomState(123)
d = 2     # データの次元
N = 10    # 各パターンのデータ数
mean = 5  # ニューロンが発火するデータの平均値
x1 = rng.randn(N,d) + np.array([0,0])        # randnは標準正規分布のランダム数発生
x2 = rng.randn(N,d) + np.array([mean,mean])  # 平均:0 標準偏差:1
# print(x1)
# file = open('InputText.txt', 'w')
# for i in x1.tolist():
#    for j in i:
#        file.write(str(j)+"\n")
# file.close()
x=np.concatenate((x1,x2),axis=0) # 行方向に結合