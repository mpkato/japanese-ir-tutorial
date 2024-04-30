import pyterrier as pt
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from datasets import load_dataset
from tqdm import tqdm

def generate_docno_text(dataset):
    for data in tqdm(dataset):
        yield {"docno": data['docid'], "text": data['text']}

def main():
    parser = ArgumentParser(
        description="Creates an Terrier inverted index with SPLADE.",
        formatter_class=ArgumentDefaultsHelpFormatter
        )
    parser.add_argument("dataset_name", help="the document collection name of a dataset to be used for datasets.load_dataset.")
    parser.add_argument("index_filepath", help="the directory in which the inverted index is stored.")
    parser.add_argument("model_name_or_path", help="the model name or path of SPLADE.")
    parser.add_argument("--split", default=None, help="the split of the dataset.")
    parser.add_argument("--batch_size", type=int, default=128, help="the batch size of the document encoder.")
    parser.add_argument("--max_length", type=int, default=256, help="the maximum length of tokens to be processed per document.")
    parser.add_argument("--mult", type=int, default=10, help="the scale factor for quantized weights.")
    args = parser.parse_args()

    if not pt.started():
        pt.init()

    dataset = load_dataset(args.dataset_name)
    if args.split:
        dataset = dataset[args.split]

    import pyt_splade
    splade = pyt_splade.SpladeFactory(args.model_name_or_path,
                                      max_length=args.max_length)

    iter_indexer = pt.IterDictIndexer(args.index_filepath, overwrite=True)
    iter_indexer.setProperty("termpipelines", "")
    iter_indexer.setProperty("tokeniser", "WhitespaceTokeniser")
    
    indexer_pipe = (splade.indexing() >> pyt_splade.toks2doc(mult=args.mult) >> iter_indexer)
    indexref = indexer_pipe.index(generate_docno_text(dataset), batch_size=args.batch_size)
    index = pt.IndexFactory.of(indexref)
    print(index.getCollectionStatistics().toString())

if __name__ == '__main__':
    main()