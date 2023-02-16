import pandas as pd
import ranky
import numpy as np


class RankyPandasAggregator:
    def __init__(self, rank_aggregation_method=ranky.borda, reverse_ranking=False):
        """
        a wrapper for ranky rank_aggregation method

        reverse_ranking means is passed to ranky method
        default=False - use descending order
        """
        self.rank_aggregation_method = rank_aggregation_method
        self.reverse_ranking = reverse_ranking

    def get_fillna_value(self, values):
        numeric_values = values.select_dtypes(np.number)
        if self.reverse_ranking:
            return numeric_values.min()
        else:
            return numeric_values.max()

    @classmethod
    def _merge_ranking_dfs(cls, ranking_dfs, merge_cols, get_fillna_value):
        results = ranking_dfs[0]
        for df in ranking_dfs[1:]:
            results = results.merge(df, on=merge_cols, how="outer")
        return results.fillna(get_fillna_value(results))

    def _aggregate_merged_df(
        self,
        merged_ranking_df,
    ):
        scores = merged_ranking_df.select_dtypes(np.number)
        agg_scores = self.rank_aggregation_method(scores, reverse=self.reverse_ranking)
        return (
            merged_ranking_df.drop(columns=scores.columns)
            .assign(score=agg_scores)
            .sort_values("score", ascending=not self.reverse_ranking)
        )

    def get_aggregated_ranking_df(self, ranking_dfs, merge_cols):
        merged_df = self._merge_ranking_dfs(
            ranking_dfs, merge_cols, self.get_fillna_value
        )
        return self._aggregate_merged_df(merged_df)
