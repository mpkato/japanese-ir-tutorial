import pyterrier as pt
import pandas as pd
import re
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from fugashi import Tagger
from datasets import load_dataset
from utils import create_japanese_analyzer

def main():
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
    parser.add_argument("--num_results", type=int, default=100, help="the number of documents to be retrieved for each query.")
    parser.add_argument("--stopword_filepath", default="./scripts/stopwords.txt", help="the filepath to a stopword file.")
    args = parser.parse_args()

    if not pt.started():
        pt.init()

    dataset = load_dataset(args.dataset_name)[args.split]
    index = pt.IndexFactory.of(args.index_filepath)

    bm25 = pt.BatchRetrieve(index, 
                    wmodel=args.wmodel, 
                    num_results=args.num_results,
                    verbose=True,
                    properties={"tokeniser": "UTFTokeniser", "termpipelines": "",
                                "bm25.k_1": 0.9, "bm25.b": 0.4}
                    )

    analyzer = create_japanese_analyzer(args.stopword_filepath)
    bm25 = pt.apply.query(lambda row: analyzer(row.query)) >> bm25

    topic_df = pd.DataFrame(dataset)
    topic_df = topic_df.rename(columns={"query_id": "qid"})

    result = bm25.transform(topic_df)
    pt.io.write_results(result, args.output_filepath, format='trec')

if __name__ == '__main__':
    main()
