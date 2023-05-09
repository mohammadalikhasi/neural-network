import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as ms

# using pandas library for extracting data from excel to a datagram
df = pd.read_csv('trainhouse.csv')
# test

# we use pandas array for our input and output
inputs = np.array(df.loc[:,['area','bedrooms','bathrooms','stories','parking']])
expected = np.array(df.loc[:,'price'])

#initial weigths and bias
weights = np.array([20000,500000,500000,500000,500000])
bias = 1000000

epoch = 0
MSE = []

while epoch < 10000:
    # we use numpy.dot method which is matrix multiplation to calculate outputs
    output = np.dot(inputs,weights) + bias
    mse = output - expected
    mse_amount = 0
    # we use a loop to calculate (output - expected)**2 and store it for plot
    if epoch < 200:
        for i in mse:
            mse_amount += i**2
        MSE.append(mse_amount)
    for i in range(5):
        # this part we use backpropagation 
        weight_change = 0
        for j in range(len(mse)):
            weight_change += 2*mse[j]*inputs[j][i]
        
        #because our data is too large we use a tiny learning rate to slower convergence
        weights[i] -= 0.000000000001*weight_change
        
            
    epoch+=1

print(f'final weights are : {weights}')
print(f'this model has {abs(np.sum(mse))} error on train')
# now we calculate the result on test data and return MSE
df2 = pd.read_csv('test.csv')
inputs = np.array(df2.loc[:,['area','bedrooms','bathrooms','stories','parking']])
expected = np.array(df2.loc[:,'price'])
output = np.dot(inputs,weights) + bias
mse = output - expected


print(f'this model has {np.sum(mse)} error on test')
# we use matlibplot for showing plots
plt.plot(MSE)
plt.show()
