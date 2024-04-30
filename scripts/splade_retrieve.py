import pyterrier as pt
import pandas as pd
import re
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from fugashi import Tagger
from datasets import load_dataset

STOPWORD_REGEX = re.compile('[!/]')

def main():
    parser = ArgumentParser(
        description="Retrieves documents from an Terrier inverted index.",
        formatter_class=ArgumentDefaultsHelpFormatter
        )
    parser.add_argument("dataset_name", help="the dataset containing query_id and query.")
    parser.add_argument("index_filepath", help="the directory including the inverted index.")
    parser.add_argument("model_name_or_path", help="the model name or path of SPLADE.")
    parser.add_argument("output_filepath", help="the filepath at which the result file is created.")
    parser.add_argument("--split", default=None, help="the split of the dataset.")
    parser.add_argument("--num_results", default=100, help="the number of documents to be retrieved for each query.")
    parser.add_argument("--max_length", type=int, default=256, help="the maximum length of tokens to be processed per document.")
    args = parser.parse_args()

    if not pt.started():
        pt.init()

    import pyt_splade
    splade = pyt_splade.SpladeFactory(args.model_name_or_path,
                                      max_length=args.max_length)

    dataset = load_dataset(args.dataset_name)[args.split]
    index = pt.IndexFactory.of(args.index_filepath)

    splade_retriever = splade.query() >> pt.BatchRetrieve(index, 
                                                          num_results=args.num_results,
                                                          verbose=True,
                                                          wmodel='Tf')

    topic_df = pd.DataFrame(dataset)
    topic_df = topic_df.rename(columns={"query_id": "qid"})

    result = splade_retriever.transform(topic_df)
    pt.io.write_results(result, args.output_filepath, format='trec')

if __name__ == '__main__':
    main()