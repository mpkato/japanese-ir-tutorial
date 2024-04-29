# PyTerrierによる検索

## 単語一致による検索

```bash
poetry run python scripts/terrier_inverted_index.py mpkato/miracl-japanese-small-corpus ./miracl_index --split train
```
```bash
poetry run python scripts/terrier_retrieve.py mpkato/miracl-japanese-small ./miracl_index ./results/bm25.trec --split dev
```
```bash
poetry run python scripts/terrier_retrieve.py mpkato/miracl-japanese-small ./miracl_index ./results/tfidf.trec --split dev --wmodel TF_IDF
```

## 評価

```bash
wget https://huggingface.co/datasets/mpkato/miracl-japanese-small/raw/main/qrels.miracl-v1.0-ja-dev.tsv
```

```bash
poetry run ir_measures qrels.miracl-v1.0-ja-dev.tsv results/bm25.trec nDCG@10 RR
```
```bash
nDCG@10 0.3175
RR      0.3279
```

```bash
poetry run ir_measures qrels.miracl-v1.0-ja-dev.tsv results/tfidf.trec nDCG@10 RR
```

```bash
nDCG@10 0.4068
RR      0.4211
```