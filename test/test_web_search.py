from mlutil.web_search import *


def test_metric_learning_definition():
    query = "metric learning"
    definition = get_searx_definition(query).lower()
    assert "metric" in definition
    assert "distance" in definition
    assert "similarity" in definition


def test_stock_prediction_definition():
    query = "stock prediction"
    definition = get_searx_definition(query).lower()
    assert "stock" in definition
    assert "price" in definition
    assert "financ" in definition
