import pyterrier as pt
import pandas as pd
import re
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from fugashi import Tagger
from datasets import load_dataset

STOPWORD_REGEX = re.compile('[!/]')

if __name__ == '__main__':
    parser = ArgumentParser(
        description="Retrieves documents from an Terrier inverted index.",
        formatter_class=ArgumentDefaultsHelpFormatter
        )
    parser.add_argument("dataset_name", help="the dataset containing query_id and query.")
    parser.add_argument("index_filepath", help="the directory including the inverted index.")
    parser.add_argument("output_filepath", help="the filepath at which the result file is created.")
    parser.add_argument("--split", default=None, help="the split of the dataset.")
    parser.add_argument("--wmodel", default="BM25", help="the name of the weighting model. "
        "See http://terrier.org/docs/current/javadoc/org/terrier/matching/models/package-summary.html")
    parser.add_argument("--num_results", default=100, help="the number of documents to be retrieved for each query.")
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