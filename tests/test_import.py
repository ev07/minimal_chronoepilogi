def test_import_chronoepilogi():
    import importlib

    mod = importlib.import_module("chronoepilogi")
    assert mod is not None
    assert getattr(mod, "__name__", None) == "chronoepilogi"
