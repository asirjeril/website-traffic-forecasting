def test_import():
    import src.traffic_forecasting as pkg
    assert hasattr(pkg, "__version__")
