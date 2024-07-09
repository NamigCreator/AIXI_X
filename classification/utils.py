import os
import numpy as np
from typing import Union, List, Iterable, Dict, Callable, Optional
from tqdm import tqdm
import multiprocessing
import multiprocessing.pool
import logging
from pathlib import Path
import datetime

# ==========================
# Classes and functions taken from previous projects
# ==========================

class Progress:
    """
    Class for visualization of progress bar in command line.
    Just some wrapping over tqdm.

    Methods
    -------
    update(n=1, desc=None)
        Updates progress bar.
    close()
        Closes progress bar.
    desc_suff(val=None)
        Adds suffix to progress bar description.
    """

    def __init__(
        self,
        iterable: Union[Iterable, None] = None,
        total: Union[int, None] = None,
        desc: Union[str, None] = None,
        bar_format: str = "{l_bar}{bar:10}{r_bar}{bar:-10b}",
        show: bool = True,
        **kwargs,
    ):
        """
        Class initialization.

        Parameters
        ----------
        iterable : Iterable, optional
            Values to iterate over in tqdm. (default is None)
        total : int or None, optional
            Number of iterations. (default is None)
        desc : str or None, optional
            Name for progress bar, desc argument for tqdm. (default is None)
        bar_format : str, optional
            Bar format for tqdm. (default is "{l_bar}{bar:10}{r_bar}{bar:-10b}")
        show : bool, optional
            If False, progress bar is not shown. (default is True)
        **kwargs: Additional keyword arguments for tqdm.
        """
        self.show = show
        self.iterable = iterable
        self.total = total
        self.desc = desc
        if self.total is None:
            try:
                self.total = len(self.iterable)
            except:
                pass
        if self.show:
            self.progress = tqdm(
                total=self.total, desc=desc, bar_format=bar_format, **kwargs
            )
        else:
            self.progress = None

    def update(self, n: int = 1, desc: Union[str, None] = None):
        """Updates progress bar."""
        self.desc_suff(desc)
        if self.show:
            self.progress.update(n)

    def close(self):
        """Closes progress bar."""
        self.desc_suff()
        if self.show:
            self.progress.close()

    def __iter__(self):
        for i in self.iterable:
            yield i
            if self.show:
                self.progress.update()
        self.close()

    def desc_suff(self, val: Union[str, int, Iterable, None] = None):
        """Adds suffix to progress bar description."""
        if val is None:
            s = self.desc
        else:
            if isinstance(val, (tuple, list, np.ndarray)):
                val = ";".join(f"{v}" for v in val)
            if self.desc is None:
                s = f"{val}"
            else:
                s = f"{self.desc}: {val}"
        if self.show and s is not None:
            self.progress.set_description(s)


def _target_function_p(args):
    index, func, args = args
    if isinstance(args, dict):
        res = func(**args)
    elif isinstance(args, (list, tuple)):
        res = func(*args)
    else:
        res = func(args)
    return index, res


# from
# https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic
# https://stackoverflow.com/a/54304172
# Wrapping classes for multiprocessing Process and Pool with daemon=False
# to avoid error
# AssertionError: daemonic processes are not allowed to have children
# when creating a Process inside of Process from Pool
class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, val):
        pass


class NoDaemonProcessPool(multiprocessing.pool.Pool):
    def Process(self, *args, **kwargs):
        proc = super(NoDaemonProcessPool, self).Process(*args, **kwargs)
        proc.__class__ = NoDaemonProcess
        return proc


def run_multiprocess(
        target_function : Callable, 
        argument_generator : Iterable, 
        threads : int = 1, 
        total : Union[int, None] = None, 
        progress : Union[Progress, None] = None, 
        show_progress : bool = True,
        start_method : Union[str, bool, None] = None,
        results : Union[list, tuple, np.ndarray, bool, None] = None,
        maxtasksperchild : Optional[int] = None,
        chunksize : int = 1,
        ) -> Union[list, tuple, np.ndarray, None]:
    """
    Runs specified target function in parallel.

    Parameters
    ----------
    target_function : Callable
        Target function to run.
    argument_generator : Iterable
        List or generator of input parameters for target function. These
        can be list or tuple for passing them as a positional arguments,
        or dict for passing as keyword arguments.
    threads : int, optional
        Number of processes to run on. If 1, function is running in the
        same process. (default is 1)
    total : int or None, optional
        Number of iterations. If None, number is defined from
        argument_generator, but this can give incorrect result, if
        argument_generator is generator instead of list. (default is None)
    progress : Progress or None, optional
        Progress bar to use for showing iterations. If None and
        show_progress is True, new Progress bar will be created
        and number of iterations will be set to 'tota'. (default is None)
    show_progress : bool, optional
        If True, progress bar is shown. (default is None)
    start_method : {"spawn", None}, optional
        Start method for starting child processes:
        "spawn" -- new Python process starts;
        None or "fork" -- copies a Python process from an existing process.
        (default is None)
    results : list or tuple or np.ndarray or bool or None, optional
        Iterable to store results for separate runs of target_function into.
        If None, new list for results is created. If False, results are not
        saved. (default is None)
    maxtasksperchild : int or None, optional
        The number of tasks a worker proces can complete before it will exit
        and be replaced with a fresh worker process, to enable unused resources
        to be freed.
        (default is None)
    chunksize : int, optional
        It controls the mapping of tasks issued to the pool
        Set it to a reasonably large number if you are running a lot of lightweight tasks.
        Otherwise, the overhead due to multiprocessing will be large and you will not
        get any speed improvement.
        (default is None)

    Returns
    -------
    results : list
        List of results of separate runs of target_function.
    """

    def argument_generator_local():
        for index, arg in enumerate(argument_generator):
            yield index, target_function, arg

    if total is None and hasattr(argument_generator, "__len__"):
        total = len(argument_generator)
    if results is None:
        if total is not None:
            results = [None for _ in range(total)]
        else:
            raise ValueError("total number of iterations should be provided")
    elif results is False:
        results = None
    if progress is None:
        progress = Progress(total=total, show=show_progress)
    threads = min(threads, total)
    if threads == 1:
        # Single-process mode
        for arg in argument_generator_local():
            i, r = _target_function_p(arg)
            if results is not None:
                results[i] = r
            progress.update()
    else:
        # Multi-process mode
        if start_method is not None:
            multiprocessing.set_start_method(start_method, force=True)
        pool = NoDaemonProcessPool(processes=threads, maxtasksperchild=maxtasksperchild)
        for i, r in pool.imap_unordered(
            _target_function_p, argument_generator_local(), chunksize=chunksize):
            if results is not None:
                results[i] = r
            progress.update()
        pool.close()
        pool.join()
    progress.close()
    return results


# from https://stackoverflow.com/questions/44691558/suppress-multiple-messages-with-same-content-in-python-logging-module-aka-log-co
class DuplicateFilter(logging.Filter):
    """Filter for removal of repeated messages in logger."""

    def filter(self, record):
        current_log = (record.module, record.levelno, record.msg)
        if current_log != getattr(self, "last_log", None):
            self.last_log = current_log
            return True
        return False


class LogFormatter(logging.Formatter):
    def __init__(
        self,
        *args,
        time_from_start: bool = True,
        log_name: bool = False,
        **kwargs,
    ):
        if time_from_start:
            s = "%(delta)s"
        else:
            s = "%(asctime)s"
            kwargs["datefmt"] = "%Y-%m-%d:%H:%M:%S"
        s += " {%(filename)-22s:%(lineno)4d}"
        if log_name:
            s += " %(name)-30s"
        s += " %(levelname)-8s: %(message)s"
        super().__init__(s, *args, **kwargs)
        self.time_from_start = time_from_start

    def format(self, record):
        duration = datetime.datetime.utcfromtimestamp(record.relativeCreated / 1000)
        record.delta = duration.strftime("%H:%M:%S")
        return super().format(record)


logging_env_var = "LOGGING_LEVEL"

def get_logging_level() -> str:
    """Retrieves logging level string from environmental variable."""
    return os.getenv(logging_env_var)

if os.getenv(logging_env_var) is None:
    os.environ[logging_env_var] = "INFO"

def init_logger(
        name : str = "", 
        level : Optional[str] = None,
        filename : Union[Path, None] = None,
        level_file : str = "INFO",
        level_stream : str = "DEBUG",
        **kwargs,
        ):
    """Creates logger."""

    if level is None:
        # getting logging level from environmental variable
        #   set it with:
        #   export LOGGING_LEVEL=DEBUG
        #   DEBUG / INFO / WARNING / ERROR
        level = os.getenv(logging_env_var)
        if not level:
            level is None
    if level is None:
        level = "INFO"

    level = logging.getLevelName(level)
    level_file = logging.getLevelName(level_file)
    level_stream = logging.getLevelName(level_stream)

    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = LogFormatter(**kwargs)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level_stream)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if filename is not None:
        dirname = filename.parent
        dirname.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(filename, "w", "utf-8")
        file_handler.setLevel(level_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.addFilter(DuplicateFilter())

    return logger