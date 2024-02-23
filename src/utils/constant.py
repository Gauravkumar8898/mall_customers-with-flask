from pathlib import Path

curr_path = Path(__file__).parents[1]
data_directory = curr_path / 'data'


mall_customer_dataset_path = data_directory / "mall_customers.csv"
curr_path1 = Path(__file__).parents[2]
test_dataset_path = curr_path1 / "test_suite/testdata.csv"


model_path = curr_path / "flask/model.joblib"

lists = ["CustomerID"]

axis = 1

# for dataset plot
x = 'Age'
y = 'Spending Score (1-100)'
title = 'Scatter plot of Age v/s Spending Score'
x_label = 'Age'
y_label = 'Spending Score'

filename = 'model.joblib'

# k value for number cluster
k_value = 4
