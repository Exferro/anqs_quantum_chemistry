import time


def timed(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        if isinstance(result, tuple):
            return result + (end_time - start_time, )
        else:
            return result, end_time - start_time
    return wrapper
