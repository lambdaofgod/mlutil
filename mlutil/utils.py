import logging
import os
import pickle


class maybe_csv_writer:
    def __init__(self, csv_path):
        self.csv_path = csv_path

    def __enter__(self):
        return self

    def write_csv_if_not_exists(self, get_df):
        if not os.path.exists(self.csv_path):
            get_df().to_csv(self.csv_path)
        else:
            logging.warning("{} already exists".format(self.csv_path))

    def get_csv_if_not_exists(self, get_df):
        if not os.path.exists(self.csv_path):
            return get_df()
        else:
            return pd.read_csv(self.csv_path)

    def __exit__(self, *args):
        pass


class maybe_pickler:
    def __init__(self, pickle_path, pickler=pickle):
        self.pickle_path = pickle_path
        self.pickler = pickle

    def __enter__(self):
        return self

    def write_pickle_if_not_exists(self, get_object):
        if not os.path.exists(self.pickle_path):
            self.pickler.dump(get_object(), f)
        else:
            logging.warning("{} already exists".format(self.pickle_path))

    def get_pickle_if_exists(self, get_object):
        if not os.path.exists(self.pickle_path):
            return get_object()
        else:
            with open(self.pickle_path, "rb") as f:
                return pickle.load(f)

    def __exit__(self, *args):
        pass
