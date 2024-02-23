import pandas as pd
import matplotlib.pyplot as plt
from src.utils.constant import axis, x, y, x_label, y_label, title
import joblib


def load_dataset(data_path):
    """
       Use Pandas read_csv function to read the dataset from the specified file path.
       Return the loaded dataset.
       """
    data_set = pd.read_csv(data_path)
    return data_set


def drop_unused_feature(dataset, lists):
    """
        :param dataset: The input dataset in a Pandas DataFrame.
        :param lists: A list of feature names to be dropped from the dataset.
        :return: Return the modified dataset.
        """
    datasets = dataset.drop(lists, axis=axis)
    return datasets


def plot_data_point(data_set):
    """
    In this function we plot data point
    :param data_set: dataframe
    :return: None
    """
    plt.figure(figsize=(15, 8))
    plt.title(title, fontsize=20)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.scatter(x=x, y=y, data=data_set)
    plt.show()


def data_for_k_mean(data_set):
    """
    this function is used for feature selection for model training
    :param data_set: (DataFrame)
    :return: new dataset return with selected feature
    """
    new_data = data_set[['Age', 'Spending Score (1-100)', 'Annual Income (k$)']].iloc[:, :].values
    return new_data


def load_model(model_path):
    """
    function helps to load saved data
    :return:
    """
    model = joblib.load(model_path)
    return model
