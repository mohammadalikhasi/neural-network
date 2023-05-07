import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as ms

df = pd.read_csv('trainhouse.csv')



inputs = np.array(df.loc[:,['area','bedrooms','bathrooms','stories','parking']])
expected = np.array(df.loc[:,'price'])


weights = np.array([20000,500000,500000,500000,500000])
bias = 1000000

epoch = 0
MSE = []
while epoch < 10000:
    output = np.dot(inputs,weights) + bias
    mse = output - expected
    mse_amount = 0
    if epoch < 200:
        for i in mse:
            mse_amount += i**2
        MSE.append(mse_amount)
    for i in range(5):
        weight_change = 0
        for j in range(len(mse)):
            weight_change += 2*mse[j]*inputs[j][i]
        
        weights[i] -= 0.000000000001*weight_change
        
            
    epoch+=1

print(f'final weights are : {weights}')
print(f'this model has {abs(np.sum(mse))} error on train')
df2 = pd.read_csv('test.csv')
inputs = np.array(df2.loc[:,['area','bedrooms','bathrooms','stories','parking']])
expected = np.array(df2.loc[:,'price'])
output = np.dot(inputs,weights) + bias
mse = output - expected


print(f'this model has {np.sum(mse)} error on test')

plt.plot(MSE)
plt.show()