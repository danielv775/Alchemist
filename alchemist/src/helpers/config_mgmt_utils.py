import logging
import os
import pickle
import time
import inspect


def calculate_time(logger=None, log_level=logging.INFO):
    '''
    Decorator to calculate time to execute a function
    obs.: The decorated function cannot be executed in multiprocessing
    :param log_level:
    :param id:
    :return:
    '''


    def calculate_time_decorator(func):
        def wrapper(*args, **kwargs):

            start = time.time()

            func(*args, **kwargs)

            finish = time.time()

            logger.log(log_level, f"{func.__name__} - Total execution time: {finish - start} [s]")

        return wrapper

    return calculate_time_decorator



def get_main_logger(name):

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(f"config_mgmt.log", mode='a')
    logger.addHandler(file_handler)

    return logger


def get_run_logger(run_folder):

    logger = logging.getLogger(run_folder)
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(f"{run_folder}/run.log", mode='w')
    logger.addHandler(file_handler)

    return logger


def create_run_folder(results_folder='results', subfolder_number=None, logger=None):

    # Create a folder to store results
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    # get all subfolders
    subfolders = [x[1] for x in os.walk(f"{results_folder}")][0]
    subfolders = [x for x in subfolders if x[0] != '.']
  
    if subfolder_number is None:
        if len(subfolders) > 0:
            subfolders = list(map(int, subfolders))
            subfolders.sort()
            last_sub_folder = subfolders[-1]

        else:
            last_sub_folder = 0

        subfolder_number = f"{results_folder}/{int(last_sub_folder) + 1}"

    # Create the new subfolder
    try:
        os.makedirs(subfolder_number)

        if logger is not None:
            logger.info(f"Directory {subfolder_number} created")

    except FileExistsError:

        if logger is not None:
            logger.info(f"Directory {subfolder_number} already existed")

        raise

    return subfolder_number


def save(data, run_folder, filename):

    file = f"{run_folder}/{filename}.pickle"

    with open(file, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load(run_folder, filename):

    file = f"{run_folder}/{filename}.pickle"

    with open(file, 'rb') as handle:
        data = pickle.load(handle)

    return data


def log_locals(logger, description=None):

    # Get the function name
    curframe = inspect.currentframe()
    caller_func = inspect.getouterframes(curframe, 2)[1]

    logger.info('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

    if description is not None:
        logger.info(description)

    logger.info(f'Function: {caller_func[3]} - with locals:')

    caller_args = caller_func.frame.f_locals

    for key, value in caller_args.items():
        logger.info(f"{key} = {value}")

    logger.info('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n')


def print_args_kwargs(func):

    def wrapper(*args, **kwargs):
        all_args = inspect.getfullargspec(func)[0]

        func_args = inspect.signature(func).bind(*args, **kwargs).arguments

        args_not_bind = list(set(all_args) - set(func_args.keys()))

        print(f'{func.__module__}.{func.__qualname__}')

        for key, value in func_args.items():
            print(f"{key} = {value}")

        for arg_not_bind in args_not_bind:
            value = inspect.signature(func).parameters[arg_not_bind].default
            print(f"{arg_not_bind} = {value}")

        return func(*args, **kwargs)
    return wrapper


def log_params(logger, dict_params):

    for key, value in dict_params.items():

        logger.info(f"{key}: {value}")
