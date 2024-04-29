import pyterrier as pt
from argparse import ArgumentParser
from fugashi import Tagger
from datasets import load_dataset
from tqdm import tqdm

def generate_docno_text(dataset):
    tagger = Tagger("-Owakati")
    for data in tqdm(dataset):
        tokens = tagger.parse(data['text'])
        yield {"docno": data['docid'], "text": tokens}

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("dataset_name", help="the document collection name of a dataset to be used for datasets.load_dataset.")
    parser.add_argument("index_filepath", help="the directory in which the inverted index is stored.")
    parser.add_argument("--split", default=None, help="the split of the dataset.")
    args = parser.parse_args()

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