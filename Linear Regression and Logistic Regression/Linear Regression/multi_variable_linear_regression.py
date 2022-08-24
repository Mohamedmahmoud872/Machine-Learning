import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# reading data from the file
data = pd.read_csv('I:\\Faculty\\4_Year-1_Term\\Machine Learning\\Assignment1\\house_data.csv')


# applying feature scaling
data.grade = (data.grade - data.grade.mean()) / data.grade.std()
data.bathrooms = (data.bathrooms - data.bathrooms.mean()) / data.bathrooms.std()
data.lat = (data.lat - data.lat.mean()) / data.lat.std()
data.sqft_living = (data.sqft_living - data.sqft_living.mean()) / data.sqft_living.std()
data.view = (data.view - data.view.mean()) / data.view.std()


# split data into 80% training and 20% testing
X_train = data.loc[:(len(data) * 80 / 100), ["grade", "bathrooms", "lat", "sqft_living", "view"]]
y_train = data.loc[:(len(data) * 80 / 100), ["price"]]
X_train.insert(0, '1', 1)

X_test_grade = data.loc[(len(data) * 80 / 100):, ["grade"]]
X_test_bathrooms = data.loc[(len(data) * 80 / 100):, ["bathrooms"]]
X_test_lat = data.loc[(len(data) * 80 / 100):, ["lat"]]
X_test_sqft_living = data.loc[(len(data) * 80 / 100):, ["sqft_living"]]
X_test_view = data.loc[(len(data) * 80 / 100):, ["view"]]
y_test = data.loc[(len(data) * 80 / 100):, ["price"]]


# convert to matrices
X_train = np.matrix(X_train.values)
y_train = np.matrix(y_train.values)
init_theta = np.matrix(np.array([0, 0, 0, 0, 0, 0]))

X_test_grade = np.matrix(X_test_grade.values)
X_test_bathrooms = np.matrix(X_test_bathrooms.values)
X_test_lat = np.matrix(X_test_lat.values)
X_test_sqft_living = np.matrix(X_test_sqft_living.values)
X_test_view = np.matrix(X_test_view.values)


# error function
def error_fun(X_train, y_train, theta):
    temp = np.power(((X_train * theta.T) - y_train), 2)
    return np.sum(temp) / (2 * len(X_train))


# gradient descent function
def gradient_descent_fun(X_train, y_train, theta, alpha, itrations):
    temp1 = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(itrations)

    for i in range(itrations):
        error = (X_train * theta.T) - y_train

        for j in range(parameters):
            temp2 = np.multiply(error, X_train[:,j])
            temp1[0,j] = theta[0,j] - ((alpha / len(X_train)) * np.sum(temp2))

        theta = temp1
        cost[i] = error_fun(X_train, y_train, theta)

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
print('Best Theta = ', best_theta)


h = best_theta[0,0] + best_theta[0,1]*X_test_grade + best_theta[0,2]*X_test_bathrooms + best_theta[0,3]*X_test_lat + \
    best_theta[0,4]*X_test_sqft_living + best_theta[0,5]*X_test_view

# C)
print("predicted price = ", h)
print("\n")
print("Actual price = ", y_test)

# D) when learning rate is too small the accuracy is bad
# but when we start increase the learning rate the accuracy
# becomes better until learning rate is equal to 0.1





