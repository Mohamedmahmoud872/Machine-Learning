import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# reading data from the file
data = pd.read_csv('I:\\Faculty\\4_Year-1_Term\\Machine Learning\\Assignment1\\house_data.csv')

# applying feature scaling
data.sqft_living = (data.sqft_living - data.sqft_living.mean()) / data.sqft_living.std()

# split data into 80% training and 20% testing
X_train = data.loc[:(len(data) * 80 / 100), ["sqft_living"]]
y_train = data.loc[:(len(data) * 80 / 100), ["price"]]
X_train.insert(0, '1', 1)

X_test = data.loc[(len(data) * 80 / 100):, ["sqft_living"]]
y_test = data.loc[(len(data) * 80 / 100):, ["price"]]


# convert to matrices
X_train = np.matrix(X_train.values)
y_train = np.matrix(y_train.values)
init_theta = np.matrix(np.array([0, 0]))


# error function
def erroe_fun(X_train, y_train, theta):
    temp = np.power(((X_train * theta.T) - y_train), 2)
    return np.sum(temp) / (2 * len(X_train))


# gradient descent function
def gradient_descent_fun(X_train, y_train, theta, alpha, iterations):
    temp1 = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iterations)

    for i in range(iterations):
        error = (X_train * theta.T) - y_train
        for j in range(parameters):
            temp2 = np.multiply(error, X_train[:,j])
            temp1[0, j] = theta[0, j] - ((alpha / len(X_train)) * np.sum(temp2))

        theta = temp1
        cost[i] = erroe_fun(X_train, y_train, theta)

    return theta, cost


# different values for learning rate
#alpha = 0.003
#alpha = 0.001
#alpha = 0.03
#alpha = 0.01
#alpha = 0.3
alpha = 0.1
i = 100

# A)
# perform linear regression on the data set
best_theta, all_error = gradient_descent_fun(X_train, y_train, init_theta, alpha, i)


# B)
print('Error in all iterations  = ', all_error)
print("==============================================================")
print('Best_theta = ', best_theta)

h = best_theta[0, 0] + best_theta[0, 1] * X_test


# C)
print("predicted price = ", h, "Actual price = ", y_test)

# D) when learning rate is too small the accuracy is bad
# but when we start increase the learning rate the accuracy
# becomes better until learning rate is equal to 0.1








