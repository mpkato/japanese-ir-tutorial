import pyterrier as pt
import pandas as pd
import re
from argparse import ArgumentParser
from fugashi import Tagger
from datasets import load_dataset

STOPWORD_REGEX = re.compile('[!/]')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("dataset_name")
    parser.add_argument("split")
    parser.add_argument("index_filepath")
    parser.add_argument("output_filepath")
    parser.add_argument("--wmodel", default="BM25")
    parser.add_argument("--num_results", default=100)
    args = parser.parse_args()

    if not pt.started():
        pt.init()

    dataset = load_dataset(args.dataset_name)[args.split]
    index = pt.IndexFactory.of(args.index_filepath)

    bm25 = pt.BatchRetrieve(index, 
                    wmodel=args.wmodel, 
                    num_results=args.num_results,
                    verbose=True,
                    properties={"tokeniser": "UTFTokeniser", "termpipelines": ""}
                    )

    tagger = Tagger("-Owakati")
    def ja_preprocess(query):
        query = STOPWORD_REGEX.sub('', query)
        tokens = tagger.parse(query)
        return tokens
    bm25 = pt.apply.query(lambda row: ja_preprocess(row.query)) >> bm25

    topic_df = pd.DataFrame(dataset)
    topic_df = topic_df.rename(columns={"query_id": "qid"})

    result = bm25.transform(topic_df)
    pt.io.write_results(result, args.output_filepath, format='trec')