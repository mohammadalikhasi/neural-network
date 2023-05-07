import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as ms
import random as rd
df = pd.read_csv('trainhouse.csv')



inputs = np.array(df.loc[:,['area','bedrooms','bathrooms','stories','parking']])
weighted_input = np.array(1.1 *df.loc[:,['area','bedrooms','bathrooms','stories','parking']])
expected = np.array(df.loc[:,'price'])
expected_weighted = np.array(1.1 * df.loc[:,'price'])

weights = np.array([20000,500000,500000,500000,500000])
bias = 1000000

epoch = 0
MSE = []
list_of_labels =['area','bedrooms','bathrooms','stories','parking']
# normilizing input our normilization method is min-max
for i in range(5):
    max_number = max(np.array(df.loc[:,list_of_labels[i]]))
    min_number = min(np.array(df.loc[:,list_of_labels[i]]))
    for j in range(0,len(expected)):
        weighted_input[j][i] = (float)((inputs[j][i] - min_number) /(max_number - min_number))

# normilizing output our normilization method is min-max
max_number = max(np.array(df.loc[:,'price']))
min_number = min(np.array(df.loc[:,'price']))
for j in range(0,len(expected)):
    expected_weighted[j] = (float)((expected[j] - min_number) /(max_number - min_number))

inputs = weighted_input
expected = expected_weighted
while epoch < 1000:
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
print(f'this model has {(np.sum(mse))} error on train')
df2 = pd.read_csv('test.csv')
inputs = np.array(df2.loc[:,['area','bedrooms','bathrooms','stories','parking']])
expected = np.array(df2.loc[:,'price'])
output = np.dot(inputs,weights) + bias
mse = output - expected


print(f'this model has {np.sum(mse)} error on test')

plt.plot(MSE)
plt.show()