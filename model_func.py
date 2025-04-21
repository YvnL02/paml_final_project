import numpy as np
import pandas as pd

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None

    def predict_probability(self, X, W, b=0):
        score = np.dot(X, W)+b
        y_pred = 1. / (1.+np.exp(-score))    
        return y_pred
    def compute_log_likelihood(self, X, Y, W, b):
        lp=None
        indicator = (Y==+1)
        scores = np.dot(X, W) + b
        logexp = np.log(1. + np.exp(-scores))
        mask = np.isinf(logexp)
        logexp[mask] = -scores[mask]
        lp = np.sum((indicator-1)*scores - logexp)/len(X)
        return lp
    def update_weights(self, X, Y, W, b, learning_rate, log_likelihood):           
        num_features, num_examples = X.shape
        y_pred = 1 / (1 + np.exp(-(X.dot(W) + b))) 
        
        dW = X.T.dot(Y-y_pred) / num_features 
        db = np.sum(Y-y_pred) / num_features 

        b = b + learning_rate * db
        W = W + learning_rate * dW

        log_likelihood = self.compute_log_likelihood(X, Y, W, b)
        return W, b, log_likelihood
    def predict(self, X, W, b):
        Z = 1 / (1 + np.exp(- (X.dot(W) + b)))
        Y = [-1 if z <= 0.5 else +1 for z in Z]
        return Y
    def fit(self, X, Y, num_iterations, learning_rate):   
        num_features, num_examples = X.shape       
        W = np.zeros(num_examples)
        b = 0
        log_likelihood=0
        likelihood_history=[]
        
        for i in range(num_iterations):          
            W, b, log_likelihood = self.update_weights(X, Y, W, b, learning_rate, log_likelihood)   
            likelihood_history.append(log_likelihood)
        return W, b, likelihood_history

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.w = None
        self.b = None
        
    def predict_score(self, X, W, b):
        scores = np.dot(X, W) + b
        return scores
    def compute_hinge_loss(self, X, Y, W, b, lambda_param):
        margins = 1 - Y * self.predict_score(X, W, b)
        margins = np.maximum(0, margins)
        loss = np.mean(margins) + (lambda_param / 2) * np.sum(W ** 2)
        return loss
    def predict(self, X, W, b):
        scores = self.predict_score(X, W, b)
        y_pred = np.where(scores >= 0, +1, -1)
        return y_pred
    def fit(self, X, Y, lambda_param, learning_rate, num_iterations):
        num_examples, num_features = X.shape
        W = np.zeros(num_features)
        b = 0
        likelihood_history = []

        for _ in range(num_iterations):
            scores = self.predict_score(X, W, b)
            indicator = (Y * scores) < 1
            
            dW = (-np.dot(X.T, (Y * indicator)) + 2 * lambda_param * W ) / num_examples
            db = -np.sum(Y * indicator) / num_examples
            
            W -= learning_rate * dW
            b -= learning_rate * db
            
            loss = self.compute_hinge_loss(X, Y, W, b, lambda_param)
            likelihood_history.append(loss)
        return W, b, likelihood_history