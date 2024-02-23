import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import logging
import joblib
from src.utils.constant import filename
logging.basicConfig(level=logging.INFO)


class KMeansClustering:

    @staticmethod
    def inertia_kmeans(new_data):
        """
        function finds the inertia
        :param new_data: (DataFrame)
        :return: inertia list
        """
        inertia = []
        logging.info("finding inertia")
        for n in range(1, 15):
            algorithm = (KMeans(n_clusters=n, init='k-means++', n_init=15,
                                random_state=71, algorithm='lloyd'))
            algorithm.fit(new_data)
            inertia.append(algorithm.inertia_)

        return inertia

    @staticmethod
    def plot_elbow(inertia):
        """
        function plot a curve which helps to find k value.
        :param inertia:
        :return:
        """
        plt.figure(1, figsize=(15, 6))
        plt.plot(np.arange(1, 15), inertia, 'o')
        plt.plot(np.arange(1, 15), inertia, '-', alpha=0.5)
        plt.xlabel('Number of Clusters'), plt.ylabel('Inertia')
        plt.show()

    @staticmethod
    def k_mean_model(k_value, dataset):
        """
        function train our model on the basis of given dataset and k value
        :param k_value: number cluster model find
        :param dataset: DataFrame
        :return: model
        """
        model = (KMeans(n_clusters=k_value, init='k-means++', n_init=10,
                        algorithm='lloyd'))
        model.fit(dataset)
        return model

    @staticmethod
    def model_save(model):
        """
        function saves our model in joblib file.
        :param model:  input we take trained model
        :return: None
        """
        local_path = filename
        joblib.dump(model, local_path)
        logging.info("model saved")
