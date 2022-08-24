from tensorflow.python.keras import datasets
from tensorflow.keras.optimizers import RMSprop
from keras.utils.np_utils import to_categorical
from keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.models import Sequential
from keras.callbacks import ReduceLROnPlateau
import numpy as np
from sklearn.model_selection import KFold # to split data into train and test
#Functions
def load_data():
    #Load Data Train And Test
    (trainX, trainY), (testX, testY) = mnist.load_data()
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))

    #Convert Vector to Binary
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)

    return trainX, trainY, testX, testY
def normalization(train, test):
    # convert datatype of pixels to float and normalize each image to range 0-1
    train_normalized = train.astype('float32') / 255.0
    test_normalized = test.astype('float32') / 255.0
    return train_normalized, test_normalized
def add_layers(layers_num, pooling_num, dense_num):
    layers = Sequential()
    t = 1
    for i in range(layers_num - 1):
        if i == 0:
            #First Layer
            layers.add(Conv2D(filters=16, kernel_size=(5, 5), padding='Same', activation='relu', input_shape=(28, 28, 1)))
        else:
            layers.add(Conv2D(filters=16, kernel_size=(5, 5), padding='Same', activation='relu'))

        if t == pooling_num:
            layers.add(MaxPool2D(pool_size=(2, 2)))
            layers.add(Dropout(0.2))
            t = 1
        if (i == layers_num - 1) and (t != pooling_num):
            layers.add(MaxPool2D(pool_size=(2, 2)))
            # Reduce The Overfitting
            layers.add(Dropout(0.2))
        t += 1

    layers.add(Flatten())

    for i in range(dense_num):
        layers.add(Dense(256, activation="relu"))
        # Reduce The Overfitting
        layers.add(Dropout(0.5))

    layers.add(Dense(10, activation="softmax"))
    optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
    layers.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    return layers
def Validate(data_X, data_Y, fold_num, layers_number, epochs, batch_size, pooling_layers, dense_layers):
    accuracy_list = list()
    optimal_accuracy = 0

    n = 1

    kfold = KFold(n_splits=fold_num, shuffle=True, random_state=1)

    for train_ix, test_ix in kfold.split(data_X):
        x_train, y_train, x_test, y_test = data_X[train_ix], data_Y[train_ix], data_X[test_ix], data_Y[test_ix]

        # add model layers and fit the model
        model = add_layers(layers_number, pooling_layers, dense_layers)  # compile and build CNN model
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2)

        # Print data accuracy
        o, accuracy = model.evaluate(x_test, y_test, verbose=0)
        print('\n==========================================================================')
        print("accuracy of (", n ,') : ', (accuracy * 100.0))
        print('============================================================================\n')
        n += 1

        if accuracy * 100 > optimal_accuracy:
            optimal_accuracy = accuracy * 100
        accuracy_list.append(accuracy)

    print("The Optimal Accuracy: ", optimal_accuracy)
    return accuracy_list
def build_architecture(layers_num=6, pooling_num=1, k_fold=3, dense_num=3, epochs=20, batch_size=86):
    x_train, y_train, x_test, y_test = load_data()
    x_train, x_test = normalization(x_train, x_test)

    results = Validate(x_train, y_train, k_fold, layers_num, epochs, batch_size, pooling_num, dense_num)
    best_architecture = np.max(results)
    print("Architecture's The Optimal Accuracy : ", best_architecture*100)
#Main
if __name__ == '__main__':
    print("Architecture 1:")
    print("---------------")
    build_architecture(5, 2, 3, 4, 3, 86)
    print("=================================")
    print("Architecture 2:")
    print("---------------")
    build_architecture(6, 2, 3, 3, 4, 86)
    print("=================================")
    print("Architecture 3:")
    print("---------------")
    build_architecture(3, 2, 3, 2, 4, 86)
    print("=================================")
    print("Architecture 4:")
    print("---------------")
    build_architecture(2, 2, 3, 2, 4, 86)