from utils import *

import os
import pandas as pd
import numpy as np
import argparse

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from matplotlib import pyplot

from sklearn                        import metrics, svm
from sklearn.linear_model           import LinearRegression
from sklearn.linear_model           import LogisticRegression
from sklearn.tree                   import DecisionTreeClassifier
from sklearn.ensemble               import RandomForestClassifier
from sklearn.neighbors              import KNeighborsClassifier
from sklearn.discriminant_analysis  import LinearDiscriminantAnalysis
from sklearn.naive_bayes            import GaussianNB
from sklearn.svm                    import SVC
from sklearn.metrics                import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection        import train_test_split
from sklearn.preprocessing          import MinMaxScaler

parser = argparse.ArgumentParser()

parser.add_argument("--data_folder", type=str, default="./raw_data/", help="Path to the data folder")
parser.add_argument("--result_folder", type=str, default="./Results/lstm_results/", help="path to directory where the results are to be stored")
parser.add_argument("--n_days", type=int, default=28, help="Number of days in future for which the model will predict the occurance of flash droughts")
parser.add_argument("--threshold", type=int, default=0, help="Value used to hard encode the value output from tanh function to -1(no flash drought) or 1 (flash drought) [value>threshold -> 1, value<threshold -> -1], i.e. classify the output as positive or negative flash drought.")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=72)

args = parser.parse_args()

if __name__ == '__main__':

    data_folder = args.data_folder
    curr_dir_path = os.getcwd()
    raw_data_folder_path = os.path.join(curr_dir_path, data_folder)

    os.chdir(data_folder)
    list_files = os.listdir() # This stores the name of all files that have the data for each latitude and longitude
    os.chdir(curr_dir_path)

    n_future = args.n_days
    threshold = args.threshold
    epochs = args.epochs
    batch_size = args.batch_size

    result_folder = args.result_folder

    lstm_results = []
    num = 0

    for file_name in list_files:

        df = get_df(raw_data_folder_path, file_name) # Reading the file as DataFrame
        df = date_columns(df) # Adding year, month and day column dates
        df = scale_date_variables(df) # Scaling Date variables
        X_scaled = scale_features(df) # Getting the scaled feature variables
        y = df['FLASH']
        y = y.replace(0, -1) # Replacing 0 with -1

        # Now we will be converting the data into a format that could be used by the LSTM Model
        X_timeseries = []
        y_timeseries = []
        n_past = 30  # Number of past days we want to use to predict the future.

        # Reformat input data into a shape: (n_samples x timesteps x n_features)
        for i in range(n_past, len(X_scaled) - n_future +1):
            X_timeseries.append(X_scaled[i - n_past:i])
            y_timeseries.append([y[i:i+n_future]])

        # Creating Training and Test sets
        X_train, X_test, y_train, y_test = train_test_split(X_timeseries, y_timeseries, test_size=0.2, shuffle=True)
        X_train, X_test, y_train, y_test = np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)
        y_train = y_train.reshape((y_train.shape[0], y_train.shape[2], 1))
        y_test = y_test.reshape((y_test.shape[0], y_test.shape[2], 1))

        # Creating the LSTM Model
        model = Sequential()
        model.add(LSTM(64, activation='tanh', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
        model.add(LSTM(32, activation='tanh', return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(y_train.shape[1], activation='tanh'))
        model.compile(optimizer='adam', loss='mse')

        # Training the LSTM Model
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0, shuffle=False)

        # Evaluating the LSTM Model
        y_pred = model.predict(X_test)

        # Getting predictions
        y_pred_hard_encoded = []
        for idx0 in range(0, y_pred.shape[0]):
            y_pred_hard_encoded_instance = []
            for idx1 in range(0, y_pred.shape[1]):
                if(y_pred[idx0][idx1]>=threshold):
                    y_pred_hard_encoded_instance.append(1)
                else:
                    y_pred_hard_encoded_instance.append(-1)
            y_pred_hard_encoded.append(y_pred_hard_encoded_instance)

        # Storing weights
        weights_path = os.path.join(result_folder, "weights", file_name)
        os.mkdir(weights_path)
        model.save(os.path.join(weights_path, "model.h5"))

        # Storing the predictions and test data
        store_testing_prediction_data(result_folder, file_name, X_test, y_test, y_pred_hard_encoded, y_pred)

        # Storing evaluation results
        store_evaluation_results(result_folder, file_name, y_test, y_pred_hard_encoded, num, n_future)

        num = num + 1





