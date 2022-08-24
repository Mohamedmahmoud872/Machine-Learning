import numpy as np
import matplotlib.pyplot as plot
import pandas as pd

#My Functions

#load the data from csv file
def load_data(filename, feature, target='target'):
    data = pd.read_csv(filename)
    x = data[feature]
    y = data[target]
    return data,x,y

#split the train and test data
def split_data(x, y, ratio):
    ran = np.random.rand(len(x)) < ratio
    return x[ran], y[ran], x[~ran], y[~ran]

def check_data(x, y, w):
    correct = 0
    #convert y to an array
    y = y.tolist()
    for i, curr in enumerate(x):
        if y[i] == predict_value(curr, w):
            correct += 1
    return correct

#Predict the value 1 or -1
def predict_value(x, w):
    if np.dot(x, w) < 0:
        return -1
    else:
        return 1

#scale function
def normalize_value(x):
    mean = np.mean(x, axis=0)
    standard_diviation = np.std(x, axis=0)
    score = x - mean
    score = score / standard_diviation
    return score

#convert the target (y) to be 1 or -1
def convert_target(y):
    return y * 2 - 1

def calculate_svm(x, y, alpha, lmbda, epochs):
    #create a epmty vector w
    w = np.zeros(x.shape[1])
    # converts y to an array
    y = y.tolist()

    #list to store the costs
    cost_log = []
    for t in range(epochs):
        cost = 0
        for i, current in enumerate(x):
            expression = y[i] * np.dot(current, w)
            if expression >= 1:
                w -= alpha * 2 * lmbda * w
            else:
                w += alpha * (np.dot(current, y[i]) - 2 * lmbda * w)
                cost += 1 - expression

        cost_log.append(cost)
    return w, cost_log



def visualize_features(x, features, y):
    plot.scatter(x[features[0]], x[features[1]], marker='o', c=y)
    plot.xlabel(features[0])
    plot.ylabel(features[1])
    plot.show()


# try a different combination of data to get the best result
def visualize(x, y):
    visualize_features(x, ['sex', 'exang'], y)
    visualize_features(x, ['sex', 'ca'], y)
    visualize_features(x, ['exang', 'ca'], y)
    visualize_features(x, ['fbs', 'exang'], y)


def plot_stat(history, c, label):
    plot.ylabel(label)
    plot.xlabel('Iterations')
    plot.plot(history, c=c, label=label)
    plot.legend()
    plot.show()

#Main
features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca',
            'thal']
target = 'target'
data,x,y = load_data('heart.csv', features, target)

#test differant features in training data
#train_features = ['age', 'chol']
train_features = ['age', 'thalach', 'chol']
#train_features = ['sex', 'exang', 'ca']

y = convert_target(y)
x = normalize_value(data[train_features])

# svm attributes
x = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
bst = -1
learning_rate = 0.009
lamda = 0.001
epochs = 50
average = 0
iterations_num = 100
cnt = 0
best_result = []

for i in range(iterations_num):
    x_train, y_train, x_test, y_test = split_data(x, y, 0.8)
    w, cost_history = calculate_svm(x_train, y_train, learning_rate, lamda, epochs)
    curr = check_data(x_test, y_test, w) / x_test.shape[0]
    average += curr
    if curr > bst:
        cnt += 1
        best_result = cost_history
        bst = curr

plot_stat(best_result, 'red', 'Costs')
print("The Average is ", average / iterations_num)
print("The Best result is ", bst)