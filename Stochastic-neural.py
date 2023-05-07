import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as ms
import random as rd

# using pandas library for extracting data from excel to a datagram
df = pd.read_csv('trainhouse.csv')



inputs = np.array(df.loc[:,['area','bedrooms','bathrooms','stories','parking']])
expected = np.array(df.loc[:,'price'])


weights = np.array([20000,500000,500000,500000,500000])
bias = 1000000

epoch = 0
MSE = []
while epoch < 10000:
    # we produce a random number for choosing our record for schotastic training
    random_number =int(rd.random() * len(inputs)) - 1
    output = np.dot(inputs[random_number],weights) + bias
    mse = output - expected[random_number]
    if epoch < 200:
        MSE.append(mse ** 2)
    for i in range(5):
        weight_change = 0
        weight_change += 2*mse*inputs[random_number][i]
        
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