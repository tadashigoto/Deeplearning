# 3.3 単純パーセプトロン (P.80)
import numpy as np
rng = np.random.RandomState(123)
d = 2     # データの次元
N = 10    # 各パターンのデータ数
mean = 5  # ニューロンが発火するデータの平均値
x1 = rng.randn(N,d) + np.array([0,0])        # randnは標準正規分布のランダム数発生,平均:0 標準偏差:1
x2 = rng.randn(N,d) + np.array([mean,mean])  # x1,x2は2次元(10x2)の乱数,x2は各要素にそれぞれ5を加算
                                             # np.array([x,x])は1x2のベクトルなのでx1,x2の各行の要素に加えられる
x=np.concatenate((x1,x2),axis=0)             # 行方向に結合
# 生成したデータをパーセプトロンで分類する
w=np.zeros(d)
b = 0 # -θ:閾値の補数
def y(x):
    return step(np.dot(w,x)+b)
def step(x):
    return 1 * (x>0)
# 先頭10件は非発火、後半10件なら発火する
def t(i):
    if i < N:
        return 0
    else:
        return 1
while True:
    classified = True
    for i in range(N*2):
        delta_w = (t(i) - y(x[i])) * x[i]
        delta_b = (t(i)) - y(x[i])
        w += delta_w
        b += delta_b
        classified *= all(delta_w ==0) * (delta_b == 0)
    if classified:
        break
print(w,b)