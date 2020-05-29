import pandas as pd
from sklearn import compose, preprocessing


def one_hot_encoder_column_transformer(columns):
    """
    transformer that stacks outputs of one-hot encoders for specified columns
    """
    return compose.ColumnTransformer(
        [
            (col, preprocessing.OneHotEncoder(), [col])
            for col in columns
        ]
    )


def make_gzipped_csv(data, name, dtype='float16'):
    pd.DataFrame(data.astype(dtype)).to_csv(name + '.csv.gz')
