import numpy as np
import pandas as pd
from sklearn import compose, preprocessing
import matplotlib.pyplot as plt


def display_img_vector(img_vector, cmap="gray"):
    vector_size = img_vector.shape[0]
    img_size = int(np.sqrt(vector_size))
    assert img_size ** 2 == vector_size
    img = img_vector.reshape(img_size, img_size)
    plt.imshow(img, cmap=cmap)


def one_hot_encoder_column_transformer(columns):
    """
    transformer that stacks outputs of one-hot encoders for specified columns
    """
    return compose.ColumnTransformer(
        [(col, preprocessing.OneHotEncoder(), [col]) for col in columns]
    )


def make_gzipped_csv(data, name, dtype="float16"):
    pd.DataFrame(data.astype(dtype)).to_csv(name + ".csv.gz")
