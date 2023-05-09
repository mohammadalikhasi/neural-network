import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as ms

df = pd.read_csv('trainhouse.csv')
#git


inputs = np.array(df.loc[:,['area','bedrooms','bathrooms','stories','parking']])
expected = np.array(df.loc[:,'price'])

# at first we divide our dataset to 5 batches
batches_input = [inputs[0:80,:],inputs[80:160,:],inputs[160:240,:],inputs[240:320,:],inputs[320:,:]]
batches_expected = [expected[0:80],expected[80:160],expected[160:240],expected[240:320],expected[320:]]

weights = np.array([20000,500000,500000,500000,500000])
bias = 1000000

epoch = 0
MSE = []
bach_select = 0


while epoch < 10000:
    # in every epoch we choose a batch and we run gradiant descent on it
    bach_select %= len(batches_input)
    output = np.dot(batches_input[bach_select],weights) + bias
    mse = output - expected[bach_select]
    mse_amount = 0
    if epoch < 200:
        for i in mse:
            mse_amount += i**2
        MSE.append(mse_amount)
    for i in range(5):
        weight_change = 0
        for j in range(len(mse)):
            weight_change += 2*mse[j]*batches_input[bach_select][j][i]
        weights[i] -= 0.000000000001*weight_change
        
    bach_select += 1
    epoch+=1

print(f'final weights are : {weights}')
print(f'this model has {(np.sum(mse))} error on train')
df2 = pd.read_csv('test.csv')
inputs = np.array(df2.loc[:,['area','bedrooms','bathrooms','stories','parking']])
expected = np.array(df2.loc[:,'price'])
output = np.dot(inputs,weights) + bias
mse = output - expected


print(f'this model has {np.sum(mse)} error on test')

plt.plot(MSE)
plt.show()
