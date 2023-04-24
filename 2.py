# 2. Посчитать коэффициент линейной регрессии при заработной плате (zp), используя градиентный спуск (без intercept).
# zp = [35, 45, 190, 200, 40, 70, 54, 150, 120, 110],
# ks = [401, 574, 874, 919, 459, 739, 653, 902, 746, 832]. 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

zp = np.array([35, 45, 190, 200, 40, 70, 54, 150, 120, 110])
ks = np.array([401, 574, 874, 919, 459, 739, 653, 902, 746, 832])

def _mse(b, x, y):
    return np.sum((b*x-y)**2)/len(x)

b=(np.mean(zp * ks) - np.mean(zp) * np.mean(ks)) / (np.mean(zp**2) - np.mean(zp) ** 2)
print('коэффициент b = ', b)

print('Среднеквадратическая функция _mse = ', _mse(2.62, zp, ks))

# производная функции потерь:

def _mse_p(b,x,y):
    return (2/len(x))*np.sum((b*x-y)*x)

# введем параметр - скорость обучения. C его помощью можно регулировать скорость подбора коэффициента b:

alpha=1e-06
b=0.1
mse_max=_mse(b,zp,ks)
i_max=1
b_max=b
for i in range(10000):
    b-=alpha*_mse_p(b,zp,ks)
    if i%100==0:
        print(f'Итерация #{i}, b={b}, mse={_mse(b, zp,ks)}')
    if _mse(b,zp,ks)>mse_max:
        print(f'Итерация #{i_max}, b={b_max}, mse={mse_max},\nДостигнут максимум.')
        break
    else:
        mse_max=_mse(b,zp,ks)
        i_max=i
        b_max=b
print('b_max = ', b_max)