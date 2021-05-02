import transformers


class TransformerVectorizer:

    def __init__(self, model_type, aggregating_function=np.mean):
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_type)
        model = transformers.AutoModel.from_pretrained(model_type)
        self.pipeline = transformers.FeatureExtractionPipeline(model, tokenizer)
        self.aggregating_function = aggregating_function

    def transform(self, texts):
        return np.array([
            self.aggregating_function(pipeline(text), axis=1)[0]
            for text in texts
        ])
