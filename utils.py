import os
import pandas as pd
import numpy as np
import argparse

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Bidirectional
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




def get_df(raw_data_folder_path, file_name):

    df = pd.read_csv(os.path.join(raw_data_folder_path, file_name), delimiter=r"\s+")
    if "SPI" in df.columns:
        df.drop(["SPI"], axis = 1, inplace = True)

    return df

def date_columns(df):

    # Getting the day, month and year
    dates = np.char.mod('%d', df["Time"])

    years = []
    months = []
    days = []

    for date in dates:

        year = int(date[:4])
        month = int(date[4:6])
        day = int(date[6:8])

        years.append(year)
        months.append(month)
        days.append(day)

    df["years"] = years
    df["months"] = months
    df["days"] = days

    df.drop(['Time'], axis=1, inplace = True)

    return df



def scale_date_variables(df):

    years = df["years"]
    months = df["months"]
    days = df["days"]
    
    scaler = MinMaxScaler()
    years = np.array(years).reshape(-1, 1)
    model = scaler.fit(years)
    scaled_years = model.transform(years)
    
    scaler = MinMaxScaler()
    months = np.array(months).reshape(-1, 1)
    model = scaler.fit(months)
    scaled_months = model.transform(months)
    
    scaler = MinMaxScaler()
    days = np.array(days).reshape(-1, 1)
    model = scaler.fit(days)
    scaled_days = model.transform(days)
    
    df["scaled_years"] = scaled_years
    df["scaled_months"] = scaled_months
    df["scaled_days"] = scaled_days
    
    df.drop(columns = ["years", "months", "days"], inplace = True)

    return df



def scale_features(df):

    X = df.loc[:, ['rh', 'SM', 'Tmean', 'e', 'scaled_years', 'scaled_months', 'scaled_days']]

    scaler = StandardScaler()
    scaler = scaler.fit(X)
    X_scaled = scaler.transform(X)

    return X_scaled



def get_date(normalized_date):

    year = normalized_date[-3]
    month = normalized_date[-2]
    day = normalized_date[-1]

    year_std = 0.2958039891549807
    year_mean = 0.5
    year_scaled = year*year_std + year_mean

    month_std = 0.3138229572304239
    month_mean = 0.5
    month_scaled = month*month_std + month_mean

    day_std = 0.29814239699997197
    day_mean = 0.5
    day_scaled = day*day_std + day_mean

    real_year = year_scaled*40 + 1980
    real_month = month_scaled*11 + 1
    real_day = day_scaled*30 + 1
    
    return real_year, real_month, real_day



def store_testing_prediction_data(Results_dir_path, file_name, X_test, y_test, y_pred, y_pred_probability):

  os.mkdir(os.path.join(Results_dir_path, "testing_prediction_data", file_name))

  for instance in range(0, len(X_test)):

    X_test_instance = X_test[instance]
    y_test_instance = y_test[instance].flatten()
    y_pred_instance = y_pred[instance]
    y_pred_probability_instance = y_pred_probability[instance]

    normalized_date = X_test_instance[-1][-3:]
    
    real_year, real_month, real_day = get_date(normalized_date)
    real_year, real_month, real_day = int(np.round(real_year)), int(np.round(real_month)), int(np.round(real_day))
    
    folder_path = os.path.join(Results_dir_path, "testing_prediction_data", file_name, f"{real_year}_{real_month}_{real_day}")

    os.mkdir(folder_path)

    np.save(os.path.join(folder_path, "X_test"), X_test_instance)
    np.savetxt(os.path.join(folder_path, "y_pred.csv"), y_pred_instance, delimiter=',')
    np.savetxt(os.path.join(folder_path, "y_pred_probability.csv"), y_pred_probability_instance, delimiter=',')
    np.savetxt(os.path.join(folder_path, "y_test.csv"), y_test_instance, delimiter=',')

    if(len(np.unique(y_test_instance)) > 1):
      file = open(os.path.join(Results_dir_path, "testing_prediction_data", "y_test_instances_with_flash_drought.txt"), "a")
      file.writelines(f"{file_name}_{real_year}_{real_month}_{real_day}\n")
      file.close()
      


def store_evaluation_results(Results_dir_path, file_name, y_test, y_pred, num, n_future):

  accuracy_list = []
  precision_list = []
  recall_list = []
  f1_list = []

  daywise_predicted_vals = {i: [] for i in range(n_future)}
  daywise_test_vals = {i: [] for i in range(n_future)}

  for instance in range(0, len(y_pred)):
    for day in range(0, len(y_pred[1])):
      daywise_predicted_vals[day].append(y_pred[instance][day])
      daywise_test_vals[day].append(y_test[instance][day][0])

  for day in range(0, len(daywise_predicted_vals)):
    accuracy_list.append(accuracy_score(daywise_test_vals[day], daywise_predicted_vals[day]))
    precision_list.append(precision_score(daywise_test_vals[day], daywise_predicted_vals[day]))
    recall_list.append(recall_score(daywise_test_vals[day], daywise_predicted_vals[day]))
    f1_list.append(f1_score(daywise_test_vals[day], daywise_predicted_vals[day]))

  for day in range(0, len(daywise_predicted_vals)):
    file = open(os.path.join(Results_dir_path, f"{day}.txt"), "a")
    accuracy = accuracy_list[day]
    precision = precision_list[day]
    recall = recall_list[day]
    f1 = f1_list[day]
    file.writelines(f"{file_name}, {np.round(accuracy*100, 2)}, {np.round(precision*100, 2)}, {np.round(recall*100, 2)}, {np.round(f1*100, 2)}, {num}\n")
    file.close()