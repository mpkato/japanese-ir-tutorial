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
    parser.add_argument("dataset_name")
    parser.add_argument("index_filepath")
    args = parser.parse_args()

    if not pt.started():
        pt.init()

    dataset = load_dataset(args.dataset_name)['train']
    iter_indexer = pt.IterDictIndexer(args.index_filepath, meta={'docno': 20, 'text': 4096},
                                    fields=('text',),
                                    overwrite=True,
                                    stemmer=None, stopwords=None, tokeniser="UTFTokeniser")
    indexref = iter_indexer.index(generate_docno_text(dataset))
    index = pt.IndexFactory.of(indexref)
    print(index.getCollectionStatistics().toString())