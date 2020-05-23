def make_gzipped_csv(data, name, dtype='float16'):
    pd.DataFrame(data.astype(dtype)).to_csv(name + '.csv.gz')
