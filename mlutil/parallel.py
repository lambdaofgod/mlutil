from concurrent.futures import ProcessPoolExecutor


def mapp(f, iterable, executor_cls=ProcessPoolExecutor, **kwargs):
    with executor_cls() as pool:
        result = pool.map(f, iterable, **kwargs)
    return result
