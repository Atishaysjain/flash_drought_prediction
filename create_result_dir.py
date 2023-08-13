import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--results_dir", type=str, default="./Results/lstm_results/", help="path to directory where the results are to be stored")
parser.add_argument("--n_days", type=int, default=28, help="Number of days in future for which the model will predict the occurance of flash droughts")
args = parser.parse_args()


if __name__ == '__main__':

    results_dir = args.results_dir
    n_days = args.n_days

    os.makedirs(os.path.join(results_dir))
    
    for day in range(0, n_days):
        file_location = os.path.join(results_dir, f'{day}.txt')
        with open(file_location, 'w') as fp:
            pass

    os.makedirs(os.path.join(results_dir, "weights"))
    os.makedirs(os.path.join(results_dir, "testing_prediction_data"))

    with open(os.path.join(results_dir, "testing_prediction_data", "y_test_instances_with_flash_drought.txt"), 'w') as fp:
        pass
