from src.utils.constant import mall_customer_dataset_path, lists
from src.utils.helpers import load_dataset, drop_unused_feature, plot_data_point,data_for_k_mean


class MallCustomerDataPreprocess:

    @staticmethod
    def preprocess_runner():
        dataset = load_dataset(mall_customer_dataset_path)
        dataset = drop_unused_feature(dataset=dataset, lists=lists)
        plot_data_point(data_set=dataset)
        dataset = data_for_k_mean(dataset)
        # print(dataset.columns)
        return dataset



