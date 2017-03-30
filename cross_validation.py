import numpy as np
import scipy.sparse as sp
from collections import Counter
import scipy.optimize
import cPickle as pickle

with open("X1.txt") as f:
    emails = f.readlines()
labels = np.loadtxt("y1.txt")

from natural_language_processing import tfidf
features, all_words = tfidf(emails)

class SVM:
    def __init__(self, X, y, reg):
        """ Initialize the SVM attributes and initialize the weights vector to the zero vector. 
            Attributes: 
                X (array_like) : training data intputs
                y (vector) : 1D numpy array of training data outputs
                reg (float) : regularizer parameter
                theta : 1D numpy array of weights
        """
        self.X = X
        self.y = y
        self.reg = reg
        self.theta = np.zeros(X.shape[1])
    
    def objective(self, X, y):
        """ Calculate the objective value of the SVM. When given the training data (self.X, self.y), this is the 
            actual objective being optimized. 
            Args:
                X (array_like) : array of examples, where each row is an example
                y (array_like) : array of outputs for the training examples
            Output:
                (float) : objective value of the SVM when calculated on X,y
        """
        hinge = 1 - y * (X.dot(self.theta.T))
        obj = sum(x for x in hinge if x > 0) + self.reg / 2 * self.theta.dot(self.theta.T)
#         print obj
        return obj
    
    def gradient(self):
        """ Calculate the gradient of the objective value on the training examples. 
            Output:
                (vector) : 1D numpy array containing the gradient
        """
        diag = sp.diags(self.y, 0)
        Xy = diag*self.X
        return -Xy.T.dot([a <= 1 for a in Xy.dot(self.theta)]) + self.reg*self.theta
    
    def train(self, niters=100, learning_rate=1, verbose=False):
        """ Train the support vector machine with the given parameters. 
            Args: 
                niters (int) : the number of iterations of gradient descent to run
                learning_rate (float) : the learning rate (or step size) to use when training
                verbose (bool) : an optional parameter that you can use to print useful information (like objective value)
        """
        for i in range(niters):
            self.theta -= self.gradient() * learning_rate;
            if verbose:
                print self.objective(self.X, self.y)
            
    
    def predict(self, X):
        """ Predict the class of each label in X. 
            Args: 
                X (array_like) : array of examples, where each row is an example
            Output:
                (vector) : 1D numpy array containing predicted labels
        """
#         print np.sign(X.dot(self.theta.T))
        return np.sign(X.dot(self.theta.T))

# Verify the correctness of your code on small examples
y0 = np.random.randint(0,2,5)*2-1
X0 = np.random.random((5,10))
t0 = np.random.random(10)
svm0 = SVM(X0,y0, 1e-4)
svm0.theta = t0


# def obj(theta):
#     pass

# def grad(theta):
#     pass

# scipy.optimize.check_grad(obj, grad, t0)

svm0.train(niters=100, learning_rate=1, verbose=True)

# svm = SVM(...)
# svm.train()
# yp = svm.predict(...)

import math

class ModelSelector:
    """ A class that performs model selection. 
        Attributes:
            blocks (list) : list of lists of indices of each block used for k-fold cross validation, e.g. blocks[i] 
            gives the indices of the examples in the ith block 
            test_block (list) : list of indices of the test block that used only for reporting results
            
    """
    def __init__(self, X, y, P, k, niters):
        """ Initialize the model selection with data and split into train/valid/test sets. Split the permutation into blocks 
            and save the block indices as an attribute to the model. 
            Args:
                X (array_like) : array of features for the datapoints
                y (vector) : 1D numpy array containing the output labels for the datapoints
                P (vector) : 1D numpy array containing a random permutation of the datapoints
                k (int) : number of folds
                niters (int) : number of iterations to train for
        """
        self.X = sp.coo_matrix(X).tocsr()
        self.y = y
        self.P = P
        self.k = k
        self.niters = niters
        self.blocks = []
        foldlen = len(P) / (k+1)
        for i in range(k+1):
            self.blocks.append(P[i*foldlen:(i+1)*foldlen])
        self.test_block = self.blocks[-1]
        self.blocks = self.blocks[:-1]
        
#         print P
#         for b in self.blocks:
#             print b

    def cross_validation(self, lr, reg):
        """ Given the permutation P in the class, evaluate the SVM using k-fold cross validation for the given parameters 
            over the permutation
            Args: 
                lr (float) : learning rate
                reg (float) : regularizer parameter
            Output: 
                (float) : the cross validated error rate
        """
        error = 0.0
        for hold in range(len(self.blocks)):
#             print "****"
            training = []
            holdout = self.blocks[hold]
#             print "holdout", holdout
#             print self.blocks[0]
            trainingx = None
            trainingy = None
            for t in range(len(self.blocks)):
#                 print t, hold
                if t != hold:
#                     print self.blocks[t]
#                     print self.X[self.blocks[t]]
                    if trainingx is not None:
                        trainingx = sp.vstack((trainingx, self.X[self.blocks[t]]))
                    else:
                        trainingx = self.X[self.blocks[t]]
                    if trainingy is not None:
                        trainingy = np.hstack((trainingy, self.y[self.blocks[t]]))
                    else:
                        trainingy = self.y[self.blocks[t]]
#                     print training
#             print training
#             print type(trainingx), type(trainingy)
        
            holdoutx = [self.X[i] for i in holdout]
            holdouty = [self.y[i] for i in holdout]
#             print np.array(trainingx).shape
            svm = SVM(trainingx, trainingy, reg)
            svm.train(self.niters, lr, False)
#             print sum([svm.predict(holdoutx[i]) != holdouty[i] for i in range(len(holdoutx))])
            error += float(sum(svm.predict(holdoutx[i]) != holdouty[i] for i in range(len(holdoutx)))) / len(holdoutx)
#             print "error", error
#             print "training", training
#             print "traningx", trainingx
#             print "trainingy", trainingy
#             print "holdout", holdout
#             print "holdoutx", holdoutx
#             print "holdouty", holdouty
#             print "******"
        
        return error/self.k
    
    def grid_search(self, lrs, regs):
        """ Given two lists of parameters for learning rate and regularization parameter, perform a grid search using
            k-wise cross validation to select the best parameters. 
            Args:  
                lrs (list) : list of potential learning rates
                regs (list) : list of potential regularizers
            Output: 
                (lr, reg) : 2-tuple of the best found parameters
        """
        best = float('inf')
        bestlr = 0
        bestreg = 0
        for lr in lrs:
            for reg in regs:
                cur = self.cross_validation(lr, reg)
                if cur < best:
                    best = cur
                    bestlr = lr
                    bestreg = reg
        return bestlr, bestreg
    
    def test(self, lr, reg):
        """ Given parameters, calculate the error rate of the test data given the rest of the data. 
            Args: 
                lr (float) : learning rate
                reg (float) : regularizer parameter
            Output: 
                (err, svm) : tuple of the error rate of the SVM on the test data and the learned model
        """
        trainingx = None
        trainingy = None
        for t in range(len(self.blocks)):
            if trainingx is not None:
                trainingx = sp.vstack((trainingx, self.X[self.blocks[t]]))
            else:
                trainingx = self.X[self.blocks[t]]
            if trainingy is not None:
                trainingy = np.hstack((trainingy, self.y[self.blocks[t]]))
            else:
                trainingy = self.y[self.blocks[t]]
             
        svm = SVM(trainingx, trainingy, reg)
        svm.train(self.niters, lr, False)   
        
        test = self.test_block
        testx = [self.X[i] for i in test]
        testy = [self.y[i] for i in test]
        
        error = float(sum(svm.predict(testx[i]) != testy[i] for i in range(len(testx)))) / len(testx)
        return error, svm

MS0 = ModelSelector(X0, y0, np.arange(X0.shape[0]), 3, 100)
MS0.cross_validation(0.1, 1e-4)

MS1 = ModelSelector(features, labels, np.arange(features.shape[0]), 5, 100)
MS1.cross_validation(0.1, 1e-4)

# MS = ModelSelector(...)
# lr, reg = MS.grid_search(...)
# print lr, reg
# print MS.test(lr,reg)

MS = ModelSelector(features, labels, np.arange(features.shape[0]), 4, 100)
lr, reg = MS.grid_search(np.logspace(-1,1,3), np.logspace(-2,1,4))
print lr, reg
print MS.test(lr,reg)

def find_frequent_indicator_words(docs, y, threshold):
    pass

s,h = find_frequent_indicator_words(emails, labels, 50)
with open('student_data.pkl', 'wb') as f:
    pickle.dump((s,h), f)


def email2features(emails):
    """ Given a list of emails, create a matrix containing features for each email. """
    with open('student_data.pkl', 'rb') as f:
        data = pickle.load(f)
    pass

small_features = email2features(emails)
# MS = ModelSelector(...)
# lr, reg = MS.grid_search(...)
# print lr, reg
# err, svm = MS.test(lr,reg)
# print err

with open('student_data.pkl', 'wb') as f:
    pickle.dump((s,h), f)
    
with open('student_params.pkl', 'wb') as f:
    pickle.dump({
        "lr" : 1.0,
        "reg" : 1e-4,
        "niters" : 100
    }, f)
