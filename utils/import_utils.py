import importlib

def is_torch_greater_2_0():
    if _is_python_greater_3_8:
        from importlib.metadata import version

        torch_version = version("torch")
    else:
        import pkg_resources

        torch_version = pkg_resources.get_distribution("torch").version
    return torch_version >= "2.0"