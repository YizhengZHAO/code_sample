import pandas as pd
import numpy as np
import scipy.linalg as la
from collections import Counter

from time_series import load_data, split_trips
vdf, _ = load_data('bus_train.db')
all_trips = split_trips(vdf)

def label_and_truncate(trip, bus_stop_coordinates):
    """ Given a dataframe of a trip following the specification in the previous homework assignment,
        generate the labels and throw away irrelevant rows. 
        
        Args: 
            trip (dataframe): a dataframe from the list outputted by split_trips from homework 2
            stop_coordinates ((float, float)): a pair of floats indicating the (latitude, longitude) 
                                               coordinates of the target bus stop. 
            
        Return:
            (dataframe): a labeled trip that is truncated at Forbes and Morewood and contains a new column 
                         called `eta` which contains the number of minutes until it reaches the bus stop. 
        """
    trip = trip.reset_index()    
    tmp = []
    for index, row in trip.iterrows():       
        tmp.append((row['lat'] - bus_stop_coordinates[0])**2 + (row['lon'] - bus_stop_coordinates[1])**2)
    stopidx = tmp.index(min(tmp))
    newdf = trip.iloc[:stopidx+1]
    stoptime = newdf.iloc[stopidx]['tmstmp']
    etacol = [(stoptime - row['tmstmp']).seconds/60 for index, row in newdf.iterrows()]
#     etacol = (stoptime - newdf.tmstmp).seconds / 60
 
    newdf = newdf.assign(eta = etacol)
 
    return newdf.set_index('tmstmp')
    
morewood_coordinates = (40.444671114203, -79.94356058465502) # (lat, lon)
labeled_trips = [label_and_truncate(trip, morewood_coordinates) for trip in all_trips]
labeled_vdf = pd.concat(labeled_trips).reset_index()
# We remove datapoints that make no sense (ETA more than 10 hours)
labeled_vdf = labeled_vdf[labeled_vdf["eta"] < 10*60].reset_index(drop=True)
print Counter([len(t) for t in labeled_trips])
print labeled_vdf

def create_features(vdf):
    """ Given a dataframe of labeled and truncated bus data, generate features for linear regression. 
    
        Args:
            df (dataframe) : dataframe of bus data with the eta column and truncated rows
        Return: 
            (dataframe) : dataframe of features for each example
        """
    n = len(vdf)
    biasvec = np.ones(n)
    
    sin_hdgvec = [np.sin(2*np.pi*row/360.) for index, row in vdf.hdg.iteritems()]
    cos_hdgvec = [np.cos(2*np.pi*row/360.) for index, row in vdf.hdg.iteritems()]
    
    sin_dayofweekvec = [np.sin(2*np.pi*row.weekday()/7.) for index, row in vdf.tmstmp.iteritems()]
    cos_dayofweekvec = [np.cos(2*np.pi*row.weekday()/7.) for index, row in vdf.tmstmp.iteritems()]
    
    sin_hour_of_dayvec = [np.sin(2*np.pi*row.hour/24.) for index, row in vdf.tmstmp.iteritems()]
    cos_hour_of_dayvec = [np.cos(2*np.pi*row.hour/24.) for index, row in vdf.tmstmp.iteritems()]
    
    sin_time_of_dayvec = [np.sin(2*np.pi*(row.hour*60 + row.minute)/60./24.) for index, row in vdf.tmstmp.iteritems()]
    cos_time_of_dayvec = [np.cos(2*np.pi*(row.hour*60 + row.minute)/60./24.) for index, row in vdf.tmstmp.iteritems()]
    
    weekday = [1 if row.weekday() < 5 else 0 for index, row in vdf.tmstmp.iteritems()]
    
    desdf = pd.get_dummies([row for index, row in vdf.des.iteritems()])
    rtdf = pd.get_dummies([row for index, row in vdf.rt.iteritems()])
    
    
    newdf = vdf.loc[:, ['pdist', 'spd', 'lat', 'lon', 'eta']]
#     newdf = vdf
    newdf = newdf.assign(bias = biasvec, sin_hdg = sin_hdgvec, cos_hdg = cos_hdgvec)
    newdf = newdf.assign(sin_day_of_week = sin_dayofweekvec, cos_day_of_week = cos_dayofweekvec)
    newdf = newdf.assign(sin_hour_of_day = sin_hour_of_dayvec, cos_hour_of_day = cos_hour_of_dayvec)
    newdf = newdf.assign(sin_time_of_day = sin_time_of_dayvec, cos_time_of_day = cos_time_of_dayvec)
    newdf = newdf.assign(weekday = weekday)
    newdf = pd.concat([newdf, desdf, rtdf], axis=1)
    
    return newdf

vdf_features = create_features(labeled_vdf)

with pd.option_context('display.max_columns', 26):
    print vdf_features.columns
    print vdf_features.head()

class LR_model():
    """ Perform linear regression and predict the output on unseen examples. 
        Attributes: 
            beta (array_like) : vector containing parameters for the features """
    
    def __init__(self, X, y):
        """ Initialize the linear regression model by computing the estimate of the weights parameter
            Args: 
                X (array-like) : feature matrix of training data where each row corresponds to an example
                y (array like) : vector of training data outputs 
            """
        self.beta = np.linalg.solve(X.T.dot(X) + 1e-4*np.eye(X.shape[1]), X.T.dot(y))
        
    def predict(self, X_p): 
        """ Predict the output of X_p using this linear model. 
            Args: 
                X_p (array_like) feature matrix of predictive data where each row corresponds to an example
            Return: 
                (array_like) vector of predicted outputs for the X_p
            """
        return X_p.dot(self.beta.T)

# Calculate mean squared error on both the training and validation set
def compute_mse(LR, X, y, X_v, y_v):
    """ Given a linear regression model, calculate the mean squared error for the 
        training dataset, the validation dataset, and for a mean prediction
        Args:
            LR (LR_model) : Linear model
            X (array-like) : feature matrix of training data where each row corresponds to an example
            y (array like) : vector of training data outputs 
            X_v (array-like) : feature matrix of validation data where each row corresponds to an example
            y_v (array like) : vector of validation data outputs 
        Return: 
            (train_mse, train_mean_mse, 
             valid_mse, valid_mean_mse) : a 4-tuple of mean squared errors
                                             1. MSE of linear regression on the training set
                                             2. MSE of predicting the mean on the training set
                                             3. MSE of linear regression on the validation set
                                             4. MSE of predicting the mean on the validation set
                         
            
    """
    prediction = LR.predict(X)
    prediction_v = LR.predict(X_v)
    prediction_mean = np.full(len(prediction), y.mean())
    prediction_mean_v = np.full(len(prediction_v), y.mean())
    
    train_mse = ((prediction - y) ** 2).mean(axis=None)
    train_mean_mse = ((prediction_mean - y) ** 2).mean(axis=None)
    valid_mse = ((prediction_v - y_v) ** 2).mean(axis=None)
    valid_mean_mse = ((prediction_mean_v - y_v) ** 2).mean(axis=None)
    return train_mse, train_mean_mse, valid_mse, valid_mean_mse

# First replicate the same processing pipeline as we did to the training set
vdf_valid, pdf_valid = load_data('bus_valid.db')
all_trips_valid = split_trips(vdf_valid)
labeled_trips_valid = [label_and_truncate(trip, morewood_coordinates) for trip in all_trips_valid]
labeled_vdf_valid = pd.concat(labeled_trips_valid).reset_index()
vdf_features_valid = create_features(labeled_vdf_valid)

# Separate the features from the output and pass it into your linear regression model.
X_df = vdf_features.drop('eta', axis=1)
y_df = vdf_features.eta
X_valid_df = vdf_features_valid.drop('eta', axis=1)
y_valid_df = vdf_features_valid.eta
LR = LR_model(X_df, y_df)
print compute_mse(LR, 
                  X_df, 
                  y_df, 
                  X_valid_df, 
                  y_valid_df)

def compare_truetime(LR, labeled_vdf, pdf):
    """ Compute the mse of the truetime predictions and the linear regression mse on entries that have predictions.
        Args:
            LR (LR_model) : an already trained linear model
            labeled_vdf (pd.DataFrame): a dataframe of the truncated and labeled bus data (same as the input to create_features)
            pdf (pd.DataFrame): a dataframe of TrueTime predictions
        Return: 
            (tt_mse, lr_mse): a tuple of the TrueTime MSE, and the linear regression MSE
        """
    merge = labeled_vdf.merge(pdf)
    print len(merge)
    merge = merge.loc[merge['typ'] == 'A']
    print len(merge)
#     print merge
    prediction_tt = [item.seconds/60. for item in merge.prdtm - merge.tmstmp]
    feature = create_features(merge)
    prediction_lr = LR.predict(feature.drop('eta', axis=1))
    y_v = feature.eta
    tt_mse = ((prediction_tt - y_v) ** 2).mean(axis=None)
    lr_mse = ((prediction_lr - y_v) ** 2).mean(axis=None)
    return tt_mse, lr_mse
    
compare_truetime(LR, labeled_vdf_valid, pdf_valid)

def contest_features(vdf, vdf_train):
    """ Given a dataframe of UNlabeled and truncated bus data, generate ANY features you'd like for linear regression. 
        Args:
            vdf (dataframe) : dataframe of bus data with truncated rows but unlabeled (no eta column )
                              for which you should produce features
            vdf_train (dataframe) : dataframe of training bus data, truncated and labeled 
        Return: 
            (dataframe) : dataframe of features for each example in vdf
        """
    # create your own engineered features
    pass
    
contest_cols = list(labeled_vdf.columns)
contest_cols.remove("eta")
contest_features(labeled_vdf_valid[contest_cols], labeled_vdf).head()
