from time import time
from functools import wraps


def timeit(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f"<{f.__name__}> Elapsed : {te-ts} [msec]")
        return result
    return wrap