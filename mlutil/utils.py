import logging


class maybe_csv_writer:

    def __init__(self, csv_path):
        self.csv_path = csv_path

    def __enter__(self):
        return self

    def write_if_not_exists(self, df):
        if not os.path.exists(self.csv_path):
            df.to_csv(self.csv_path)
        else:
            logging.warning("{} already exists".format(self.csv_path))

    def __exit__(self, *args):
        pass
