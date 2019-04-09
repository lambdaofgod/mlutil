import warnings
from gensim.models import KeyedVectors
import gensim.downloader as gensim_data_downloader


def load_gensim_embedding_model(model_name):
    available_models = gensim_data_downloader.info()['models'].keys()
    assert model_name in available_models, 'Invalid model_name: {}. Choose one from {}'.format(model_name, ', '.join(available_models))
    
    ## gensim throws some nasty warnings about vocabulary 
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
        model_path = gensim_data_downloader.load(model_name, return_path=True)
    return KeyedVectors.load_word2vec_format(model_path, binary=True, unicode_errors='ignore')
