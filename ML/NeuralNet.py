#####################################################################################################################
#   Assignment 2, Neural Network Programming
#   This is a starter code in Python 3.6 for a 2-hidden-layer neural network.
#   You need to have numpy and pandas installed before running this code.
#   Below are the meaning of symbols:
#   train - training dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   train - test dataset - can be a link to a URL or a local file
#         - you can assume the last column will the label column
#   h1 - number of neurons in the first hidden layer
#   h2 - number of neurons in the second hidden layer
#   X - vector of features for each instance
#   y - output for each instance
#   w01, delta01, X01 - weights, updates and outputs for connection from layer 0 (input) to layer 1 (first hidden)
#   w12, delata12, X12 - weights, updates and outputs for connection from layer 1 (first hidden) to layer 2 (second hidden)
#   w23, delta23, X23 - weights, updates and outputs for connection from layer 2 (second hidden) to layer 3 (output layer)
#
#   You need to complete all TODO marked sections
#   You are free to modify this code in any way you want, but need to mention it in the README file.
#
#####################################################################################################################
from copy import deepcopy

from sklearn import preprocessing
import numpy as np
import pandas as pd


class NeuralNet:

    def __init__(self, train, header = True, h1 = 4, h2 = 2):
        np.random.seed(1)
        # train refers to the training dataset
        # test refers to the testing dataset
        # h1 and h2 represent the number of nodes in 1st and 2nd hidden layers

        raw_input = pd.read_csv(train)
        # TODO: Remember to implement the preprocess method
        whole_dataset = self.preprocess(raw_input)
        samples = len(whole_dataset.index)
        train_dataset = whole_dataset[:int(samples*0.75)]
        self.test_dataset = whole_dataset[int(samples*0.75):-1]
        ncols = len(train_dataset.columns)
        nrows = len(train_dataset.index)
        self.X = train_dataset.iloc[:, 0:(ncols -1)].values.reshape(nrows, ncols-1)
        self.y = train_dataset.iloc[:, (ncols-1)].values.reshape(nrows, 1)
        #
        # Find number of input and output layers from the dataset
        #
        input_layer_size = len(self.X[0])
        if not isinstance(self.y[0], np.ndarray):
            output_layer_size = 1
        else:
            output_layer_size = len(self.y[0])

        #self.activation = 'sigmoid'
        # assign random weights to matrices in network
        # number of weights connecting layers = (no. of nodes in previous layer) x (no. of nodes in following layer)
        self.w01 = 2 * np.random.random((input_layer_size, h1)) - 1
        self.X01 = self.X
        self.delta01 = np.zeros((input_layer_size, h1))
        self.w12 = 2 * np.random.random((h1, h2)) - 1
        self.X12 = np.zeros((len(self.X), h1))
        self.delta12 = np.zeros((h1, h2))
        self.w23 = 2 * np.random.random((h2, output_layer_size)) - 1
        self.X23 = np.zeros((len(self.X), h2))
        self.delta23 = np.zeros((h2, output_layer_size))
        self.deltaOut = np.zeros((output_layer_size, 1))
    #
    # TODO: I have coded the sigmoid activation function, you need to do the same for tanh and ReLu
    #

    def __activation(self, x, activation="sigmoid"):
        if activation == "sigmoid":
            return self.__sigmoid(x)
        if activation == "tanh":
            return self.__tanh(x)
        else:
            return self.__relu(x)

    #
    # TODO: Define the function for tanh, ReLu and their derivatives
    #

    def __activation_derivative(self, x, activation="sigmoid"):
        if activation == "sigmoid":
            return self.__sigmoid_derivative(x)
        if activation == "tanh":
            return self.__tanh_derivative(x)
        else:
            return self.__Relu_derivative(x)

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __tanh(self, x):
        x = 2*x
        result = (2 * self.__sigmoid(x))-1
        return result

    def __relu(self, x):
        relu = deepcopy(x)
        relu[relu <= 0] = 0
        return relu
    # derivative of sigmoid function, indicates confidence about existing weight

    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    def __tanh_derivative(self, x):

        return 1-(x ** 2)

    def __Relu_derivative(self, x):
        relu = deepcopy(x)
        relu[relu<=0] = 0
        relu[relu>0] = 1
        return relu
    #
    # TODO: Write code for pre-processing the dataset, which would include standardization, normalization,
    #   categorical to numerical, etc
    #

    def preprocess(self, X):
        # categorical attributes to numerical values first
        #encode = preprocessing.OrdinalEncoder()
        #encode.fit(X)
       # X = encode.transform(X)

        #Normalization

       # X=preprocessing.normalize(X)

        # standardization
       # X=preprocessing.scale(X)
        #transform from numpy.ndarray to pandas DataFrame
        #Y=pd.DataFrame({"Sample code number": X[:, 0], "Clump Thickness": X[:, 1],"Uniformity of Cell Size": X[:, 2],"Uniformity of Cell Shape": X[:, 3],"Marginal Adhesion": X[:, 4],"Single Epithelial Cell Size": X[:, 5],"Bare Nuclei": X[:, 6],"Bland Chromatin": X[:, 7],"Normal Nucleoli": X[:, 8],"Mitoses": X[:, 9],"Class": X[:, 10]})
        #Y=pd.DataFrame({"X1": X[:, 0],"X2": X[:, 1],"X3": X[:, 2],"y": X[:, 3]})
        #return Y
        #X=pd.DataFrame(X,index=X[:,0])
        #print("X= ",X)
        # standardizing and scaling X
            numeric_cols = X.columns[X.dtypes != 'object']
            numeric_col_means = X.loc[:, numeric_cols].mean()
            numeric_col_std = X.loc[:, numeric_cols].std()
            X.loc[:, numeric_cols] = (X.loc[:, numeric_cols] - numeric_col_means) / numeric_col_std
        # replace nan with 0
            X = X.fillna(0)
        # one hot
            X = pd.get_dummies(X)
            return X

    # Below is the training function

    def train(self, max_iterations=1000, learning_rate = 0.05,activation="sigmoid"):
        for iteration in range(max_iterations):
            out = self.forward_pass(activation)
            error = 0.5 * np.power((out - self.y), 2)
            self.backward_pass(out, activation)
            update_layer2 = learning_rate * self.X23.T.dot(self.deltaOut)
            update_layer1 = learning_rate * self.X12.T.dot(self.delta23)
            update_input = learning_rate * self.X01.T.dot(self.delta12)

            self.w23 += update_layer2
            self.w12 += update_layer1
            self.w01 += update_input

        print("After " + str(max_iterations) + " iterations, the total error is " + str(np.sum(error)))
        print("The final weight vectors are (starting from input to output layers)")
        print(self.w01)
        print(self.w12)
        print(self.w23)

    def forward_pass(self,activation="sigmoid"):
        if activation == "sigmoid":
        # pass our inputs through our neural network
            in1 = np.dot(self.X, self.w01 )
            self.X12 = self.__sigmoid(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__sigmoid(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__sigmoid(in3)
            return out
        if activation == "tanh":
            in1 = np.dot(self.X, self.w01 )
            self.X12 = self.__tanh(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__tanh(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__tanh(in3)
            return out
        if activation == "ReLu":
            in1 = np.dot(self.X, self.w01 )
            self.X12 = self.__relu(in1)
            in2 = np.dot(self.X12, self.w12)
            self.X23 = self.__relu(in2)
            in3 = np.dot(self.X23, self.w23)
            out = self.__relu(in3)
            return out

    def backward_pass(self, out, activation="sigmoid"):
        # pass our inputs through our neural network
        self.compute_output_delta(out, activation)
        self.compute_hidden_layer2_delta(activation)
        self.compute_hidden_layer1_delta(activation)

    # TODO: Implement other activation functions

    def compute_output_delta(self, out, activation):
        if activation == "sigmoid":
            delta_output = (self.y - out) * (self.__sigmoid_derivative(out))
        if activation == "tanh":
            delta_output = (self.y - out) * (self.__tanh_derivative(out))
        if activation == "ReLu":
            delta_output = (self.y - out) * (self.__Relu_derivative(out))
        self.deltaOut = delta_output

    # TODO: Implement other activation functions

    def compute_hidden_layer2_delta(self, activation):
        if activation == "sigmoid":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__sigmoid_derivative(self.X23))
        if activation == "tanh":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__tanh_derivative(self.X23))
        if activation == "ReLu":
            delta_hidden_layer2 = (self.deltaOut.dot(self.w23.T)) * (self.__Relu_derivative(self.X23))
        self.delta23 = delta_hidden_layer2

    # TODO: Implement other activation functions

    def compute_hidden_layer1_delta(self, activation):
        if activation == "sigmoid":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__sigmoid_derivative(self.X12))
        if activation == "tanh":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__tanh_derivative(self.X12))
        if activation == "ReLu":
            delta_hidden_layer1 = (self.delta23.dot(self.w12.T)) * (self.__Relu_derivative(self.X12))
        self.delta12 = delta_hidden_layer1

    def forward_pass_predict_error(self, X_test,activation):
        in1 = np.dot(X_test, self.w01)
        X12 = self.__activation(in1, activation)
        in2 = np.dot(X12, self.w12)
        X23 = self.__activation(in2, activation)
        in3 = np.dot(X23, self.w23)
        out = self.__activation(in3, activation)
        return out

    # TODO: Implement the predict function for applying the trained model on the  test dataset.
    # You can assume that the test dataset has the same format as the training dataset
    # You have to output the test error from this function

    def predict(self, test, header = True, activation="sigmoid"):
        ncols = len(self.test_dataset.columns)
        nrows = len(self.test_dataset.index)
        X_test = self.test_dataset.iloc[:, 0:(ncols - 1)].values.reshape(nrows, ncols - 1)
        y_test = self.test_dataset.iloc[:, (ncols - 1)].values.reshape(nrows, 1)
        y_prediction = self.forward_pass_predict_error(X_test, activation)
        error = 0.5 * np.power((y_prediction - y_test), 2)
        return np.sum(error)


if __name__ == "__main__":
    #neural_network = NeuralNet("train.csv")
    #neural_network = NeuralNet("breast-cancer-wisconsin.data")
    neural_network = NeuralNet("wdbc.data")
    neural_network.train()
    #testError = neural_network.predict("wdbc_test.data")
    testError = neural_network.predict("test.csv")
    print("prediction error is ", testError)
