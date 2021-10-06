import logging
import os
import pickle
import time
import inspect

logger = logging.getLogger(__name__)

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


class ResultsLogger:

    def __init__(self, results_folder='results') -> None:
        
        self.results_folder = f"{os.environ['PYTHONPATH']}alchemist/{results_folder}"

        self.current_subfolder = self._create_run_folder()

        self.logger = logging.getLogger(self.current_subfolder)
        self.logger.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        self.logger.addHandler(console_handler)

        file_handler = logging.FileHandler(f"{self.current_subfolder}/run.log", mode='w')

        self.logger.addHandler(file_handler)

    
    def _create_run_folder(self, subfolder_number=None):

        # Create a folder to store results
        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)

        # get all subfolders
        subfolders = [x[1] for x in os.walk(f"{self.results_folder}")][0]
        subfolders = [x for x in subfolders if x[0] != '.']
    
        if subfolder_number is None:
            if len(subfolders) > 0:
                subfolders = list(map(int, subfolders))
                subfolders.sort()
                last_sub_folder = subfolders[-1]

            else:
                last_sub_folder = 0

            subfolder_number = f"{self.results_folder}/{int(last_sub_folder) + 1}"

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


    def log(self, data: str):        

        self.logger.info(data)
    
    
    def save_pickle(self, data, filename):

        file = f"{self.current_subfolder}/{filename}.pickle"

        with open(file, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


    def load_pickle(self, filename):

        file = f"{self.current_subfolder}/{filename}.pickle"

        with open(file, 'rb') as handle:
            data = pickle.load(handle)

        return data
