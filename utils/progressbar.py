#!/usr/bin/env python

import sys
import time
import threading
from functools import wraps

def progressbar(function=None, char='.', pause=0.2, bar_len = 60):
    """
    This function is a decorator for a progess bar.
    Use it as follows:

    .. python code:

        @progressbar
        def any_function()
            ... do something ...

        any_function()
    ..

    """
    if function is None:
        return lambda func: progressbar(func, char, pause, bar_len)

    @wraps(function)
    def wrapped_function(*args, **kwargs):
        stop = False

        def progress_bar():
            while not stop:
                sys.stdout.write(char)
                sys.stdout.flush()
                time.sleep(pause)
            sys.stdout.write('\n Done. \n')
            sys.stdout.flush()

        try:
            p = threading.Thread(target=progress_bar)
            p.start()
            return function(*args, **kwargs)
        finally:
            stop = True

    return wrapped_function
