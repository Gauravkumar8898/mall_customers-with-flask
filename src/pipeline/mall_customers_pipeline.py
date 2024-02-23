from src.preprocess.mall_customer_preprocessing import MallCustomerDataPreprocess
from src.mall_customers_model.customers_model import KMeansClustering
from src.utils.constant import k_value


def pipline():
    pre_obj = MallCustomerDataPreprocess()
    dataset = pre_obj.preprocess_runner()

    model_obj = KMeansClustering()
    inertia = model_obj.inertia_kmeans(dataset)
    model_obj.plot_elbow(inertia)
    model = model_obj.k_mean_model(k_value=k_value, dataset=dataset)
    model_obj.model_save(model)




