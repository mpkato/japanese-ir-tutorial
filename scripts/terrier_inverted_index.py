import pyterrier as pt
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from datasets import load_dataset
from tqdm import tqdm
from utils import create_japanese_analyzer

def main():
    parser = ArgumentParser(
        description="Creates an Terrier inverted index.",
        formatter_class=ArgumentDefaultsHelpFormatter
        )
    parser.add_argument("dataset_name", help="the document collection name of a dataset to be used for datasets.load_dataset.")
    parser.add_argument("index_filepath", help="the directory in which the inverted index is stored.")
    parser.add_argument("--split", default=None, help="the split of the dataset.")
    parser.add_argument("--stopword_filepath", default="./scripts/stopwords.txt", help="the filepath to a stopword file.")
    args = parser.parse_args()

    def generate_docno_text(dataset):
        analyzer = create_japanese_analyzer(args.stopword_filepath)
        for data in tqdm(dataset):
            text = analyzer(data['text'])
            yield {"docno": data['docid'], "text": text}

    if not pt.started():
        pt.init()

    dataset = load_dataset(args.dataset_name)
    if args.split:
        dataset = dataset[args.split]
    iter_indexer = pt.IterDictIndexer(args.index_filepath, meta={'docno': 20, 'text': 4096},
                                    fields=('text',),
                                    overwrite=True,
                                    stemmer=None, stopwords=None, tokeniser="UTFTokeniser")
    indexref = iter_indexer.index(generate_docno_text(dataset))
    index = pt.IndexFactory.of(indexref)
    print(index.getCollectionStatistics().toString())

if __name__ == '__main__':
    main()