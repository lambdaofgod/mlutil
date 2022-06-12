import wordcloud
import matplotlib.pyplot as plt
import pandas as pd


def show_word_cloud_from_texts(texts: pd.Series, figure_kwargs={"figsize": (8, 5)}):
    texts_values = texts.fillna("").values
    cloud = get_word_cloud(texts_values)
    show_word_cloud(cloud, figure_kwargs)


def show_word_cloud(wc: wordcloud.WordCloud, figure_kwargs):
    plt.figure(**figure_kwargs)
    plt.imshow(wc)
    plt.axis("off")
    plt.show()


def get_word_cloud(texts):
    text = " ".join(texts)
    return wordcloud.WordCloud(max_font_size=40).generate(text)
