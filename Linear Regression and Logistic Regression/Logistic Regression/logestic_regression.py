import pandas as pd
import numpy as np

dataset = pd.read_csv('heart.csv')

x = dataset[['trestbps','chol','thalach', 'oldpeak']].values
y = dataset['target'].values

class Logistic_Regression:
    def __init__(self, learning_rate=0.001, iterations_number=1000):
        self.learning_rate = learning_rate
        self.iterations_number = iterations_number

    def predict_func(self, x):
        pred_y = self.theta[0] + np.dot(x, self.theta[1:])
        z = pred_y
        g_z = 1 / (1 + np.e ** (-z))
        return [1 if i > 0.5 else 0 for i in g_z]

    def fit(self, x, y):
        self.losses = []
        self.theta = np.zeros((1 + x.shape[1]))
        n = x.shape[0]

        for i in range(self.iterations_number):
            y_pred = self.theta[0] + np.dot(x, self.theta[1:])

            z = y_pred
            g_z = 1 / (1 + np.e ** (-z))

            cost = (-y * np.log(g_z) - (1 - y) * np.log(1 - g_z)) / n
            self.losses.append(cost)
            d_theta1 = (1 / n) * np.dot(x.T, (g_z - y))
            d_theta0 = (1 / n) * np.sum(g_z - y)

            self.theta[1:] = self.theta[1:] - self.learning_rate * d_theta1
            self.theta[0] = self.theta[0] - self.learning_rate * d_theta0
        print("Cost: ",end="\n")
        print(cost)
        return self



def scale_func(x):
    x_scaled = x - np.mean(x, axis=0)
    x_scaled = x_scaled / np.std(x_scaled, axis=0)
    return x_scaled

x_sd= scale_func(x)
model = Logistic_Regression()
model.fit(x_sd, y)
pred_y = model.predict_func(x_sd)
print("Theta 0= ", model.theta[0])
print("Theta 1= ", model.theta[1])
print("Theta 2= ", model.theta[2])

print("Prediction: " ,pred_y)
