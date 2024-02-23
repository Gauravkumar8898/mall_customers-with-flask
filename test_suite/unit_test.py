import unittest
from src.utils.helpers import load_dataset, drop_unused_feature
from src.utils.constant import test_dataset_path
import pandas as pd


class TestHelpers(unittest.TestCase):

    def test_load_dataset(self):
        dataset = load_dataset(test_dataset_path)
        assert type(dataset) is pd.DataFrame

    def test_drop_unused_feature(self):
        dataset = pd.read_csv(test_dataset_path)
        datasets = drop_unused_feature(dataset, lists=["CustomerID"])
        assert len(dataset) > len(datasets)

