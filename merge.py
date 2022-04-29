import pandas as pd
import numpy as np
import os

working_directory = os.path.dirname(os.path.abspath('train_features.csv'))
x_test = pd.read_csv(working_directory + '/test_features.csv')
task1 = pd.read_csv(working_directory + '/task1_results.csv')
task2 = pd.read_csv(working_directory + '/task2_results.csv')
task3 = pd.read_csv(working_directory + '/task3_results.csv')


def sorting(x_test):
    pid = np.array([])
    for patient, features in x_test.groupby('pid'):
        sorted_pid = features.iat[0, 0]
        pid = np.append(pid, sorted_pid)
    df_pid = pd.DataFrame(pid, columns=['pid'])
    return df_pid


pid_ordered = sorting(x_test)


def join(pid, task1, task2, task3):
    task12 = pd.concat([task1, task2], axis=1)
    task12 = task12.drop(['Unnamed: 0'], axis=1)
    task123 = pd.concat([task12, task3], axis=1)
    task123 = task123.drop(['Unnamed: 0'], axis=1)
    task123.insert(0, "pid", pid)
    task123['pid'] = task123['pid'].astype('int')

    task_final = task123.round(decimals=3)

    task_final.to_csv(working_directory + '/predictionf.zip', index=False, float_format='%.3f',
                      compression='zip')
    return 0
