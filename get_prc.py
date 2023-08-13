import os
import pandas as pd
import numpy as np
import csv
import argparse
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from matplotlib import pyplot as plt
import sys

parser = argparse.ArgumentParser()

parser.add_argument("--result_folder", type=str, help="Path to the result folder that was passed in main.py, storing all the results")

args = parser.parse_args()

if __name__ == '__main__':

    result_folder = args.result_folder

    y_pred_all = []
    y_test_all = []

    curr_dir_path = os.getcwd()
    Results_dir_path = os.path.join(result_folder, "testing_prediction_data")
    coor_result_files = os.listdir(Results_dir_path)

    for coor_file in coor_result_files:
        if(coor_file != "y_test_instances_with_flash_drought.txt"):
            date_files = os.listdir(os.path.join(Results_dir_path, coor_file))
            for date_file in date_files:
                with open(os.path.join(Results_dir_path, coor_file, date_file,"y_pred_probability.csv"), newline='') as f:
                    reader = csv.reader(f)
                    y_pred_prob = list(reader)
                with open(os.path.join(Results_dir_path, coor_file, date_file, "y_test.csv"), newline='') as f:
                    reader = csv.reader(f)
                    y_test = list(reader)
                for i in range(0, len(y_pred_prob)):
                    y_pred_prob[i] = float(y_pred_prob[i][0])
                for i in range(0, len(y_test)):
                    y_test[i] = float(y_test[i][0])
                y_pred_all.append(y_pred_prob)
                y_test_all.append(y_test)

    y_pred_all = np.array(y_pred_all).flatten()
    y_test_all = np.array(y_test_all).flatten()

    lr_precision, lr_recall, thresholds = precision_recall_curve(y_test_all, y_pred_all)
    lr_precision, lr_recall = np.array(lr_precision), np.array(lr_recall)
    lr_auc = auc(lr_recall, lr_precision)

    os.mkdir(os.path.join(result_folder, "prc_results"))

    np.savetxt(os.path.join(result_folder, "prc_results", f'{result_folder}_lr_precision'), lr_precision, delimiter=',')
    np.savetxt(os.path.join(result_folder, "prc_results", f'{result_folder}_lr_recall'), lr_recall, delimiter=',')
    np.savetxt(os.path.join(result_folder, "prc_results", f'{result_folder}_lr_thresholds'), thresholds, delimiter=',')

    plt.plot(lr_recall, lr_precision, label=f'lr_auc = {lr_auc}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('precision_recall_curve')
    plt.legend()
    plt.savefig(os.path.join(result_folder, "prc_results", 'total_precision_recall_curve.png'))
    plt.clf()


        

                
                
                