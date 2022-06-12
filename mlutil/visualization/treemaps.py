import matplotlib.pyplot as plt
import squarify


def truncate_counts(counts, min_counts):
    truncated_counts = counts[counts >= min_counts]
    truncated_counts["other"] = counts[counts < min_counts].sum()
    return truncated_counts.sort_values(ascending=False)


def plot_treemap_with_counts(x, min_counts, use_percentages=False, figsize=(9, 6)):
    plt.figure(figsize=figsize)
    x_counts = truncate_counts(x.value_counts(), min_counts)
    if use_percentages:
        x_counts_str = (100 * (x_counts / x_counts.sum())).round(1).apply("{}%".format)
    else:
        x_counts_str = x_counts.apply(str)
    labels = x_counts.index + "\n" + x_counts_str
    squarify.plot(x_counts.values, label=labels)
    plt.axis("off")
