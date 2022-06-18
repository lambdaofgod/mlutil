import pandas as pd
from sklearn import metrics


def get_text_embedding_comparison_df(
    sentence_transformer,
    debug_texts_ref,
    debug_texts_other,
    similarity=metrics.pairwise.cosine_similarity,
):
    embs_ref = sentence_transformer.encode(debug_texts_ref)
    embs_other = sentence_transformer.encode(debug_texts_other)
    similarities = similarity(embs_ref, embs_other)
    return pd.DataFrame(
        data=similarities,
        index=debug_texts_ref,
        columns=debug_texts_other,
    )
