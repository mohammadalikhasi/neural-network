import numpy as np
import pandas as pd

df = pd.read_csv('trainhouse.csv')



inputs = np.array(df.loc[:,['area','bedrooms','bathrooms','stories','parking']])
expected = np.array(df.loc[:,'price'])


weights = np.array([10000,1000000,1000000,3000000,2000000])
bias = 1000000

print(inputs[0][1])
epoch = 0

while epoch < 100000:
    output = np.dot(inputs,weights) + bias
    mse = output - expected
    print(weights)
    for i in range(5):
        weight_change = 0
        for j in range(len(mse)):
            weight_change += 2*mse[j]*inputs[j][i]
        if (i == 0):
            weights[i] -= 0.00000000001*weight_change
        else:
             weights[i] -= 0.0000001*weight_change
    epoch+=1

print(output)
print(np.sum(mse))