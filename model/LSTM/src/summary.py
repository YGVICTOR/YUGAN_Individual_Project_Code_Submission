import os
import pandas as pd
from matplotlib import pyplot as plt
filePath = '../temporary_model'
all_result_list = os.listdir(filePath)
result_dict = {
    "mape":[],
    "epoch":[],
    "memory_window":[],
    "number_of_lstm_layers":[],
    "number_of_dense_layers":[],
    "number_of_unit_per_layer":[],

}
for current_result in all_result_list:
    current_result_list = current_result.split(sep='_')
    result_dict['mape'].append(float(current_result_list[0]))
    result_dict['epoch'].append(int(current_result_list[1]))
    result_dict['memory_window'].append(int(current_result_list[4]))
    result_dict['number_of_lstm_layers'].append(int(current_result_list[7]))
    result_dict['number_of_dense_layers'].append(int(current_result_list[9]))
    result_dict['number_of_unit_per_layer'].append(int(current_result_list[11]))

raw_summary_df = pd.DataFrame(result_dict)
summary = raw_summary_df.groupby(by = ['memory_window','number_of_lstm_layers','number_of_dense_layers','number_of_unit_per_layer'])['mape'].min()
print(summary)
summary.to_csv("summary.csv")

filePath = '../without_trend_temporary_model'
all_result_list = os.listdir(filePath)
result_dict = {
    "mape":[],
    "epoch":[],
    "memory_window":[],
    "number_of_lstm_layers":[],
    "number_of_dense_layers":[],
    "number_of_unit_per_layer":[],

}
for current_result in all_result_list:
    current_result_list = current_result.split(sep='_')
    result_dict['mape'].append(float(current_result_list[0]))
    result_dict['epoch'].append(int(current_result_list[1]))
    result_dict['memory_window'].append(int(current_result_list[4]))
    result_dict['number_of_lstm_layers'].append(int(current_result_list[7]))
    result_dict['number_of_dense_layers'].append(int(current_result_list[9]))
    result_dict['number_of_unit_per_layer'].append(int(current_result_list[11]))

raw_summary_df = pd.DataFrame(result_dict)
summary = raw_summary_df.groupby(by = ['memory_window','number_of_lstm_layers','number_of_dense_layers','number_of_unit_per_layer'])['mape'].min()
print(summary)
summary.to_csv("with_out_summary.csv")


df1 =pd.read_csv("summary.csv")
df2 = pd.read_csv("with_out_summary.csv")

