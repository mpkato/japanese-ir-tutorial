

## コーパス中の文書をエンコーディング
```bash
poetry run python -m tevatron.driver.encode \
  --output_dir=temp \
  --model_name_or_path aken12/dpr-japanese \
  --fp16 \
  --per_device_eval_batch_size 156 \
  --p_max_len 128 \
  --dataset_name mpkato/miracl-japanese-small-corpus \
  --encoded_save_path corpus_emb.pkl
```

## クエリのエンコーディング
```bash
poetry run python -m tevatron.driver.encode \
  --output_dir=temp \
  --model_name_or_path aken12/dpr-japanese \
  --fp16 \
  --per_device_eval_batch_size 156 \
  --dataset_name mpkato/miracl-japanese-small/dev \
  --encoded_save_path query_emb.pkl \
  --q_max_len 32 \
  --encode_is_qry
```

## 近似最近傍探索
```bash
poetry run python -m tevatron.faiss_retriever \
  --query_reps query_emb.pkl \
  --passage_reps corpus_emb.pkl \
  --depth 100 \
  --batch_size -1 \
  --save_text \
  --save_ranking_to rank.txt
```

## 出力をTRECフォーマットに変換
```bash
poetry run python -m tevatron.utils.format.convert_result_to_trec \
  --input rank.txt \
  --output rank.trec
```

## 評価

```bash
wget https://huggingface.co/datasets/mpkato/miracl-japanese-small/raw/main/qrels.miracl-v1.0-ja-dev.tsv
```

```bash
poetry run ir_measures qrels.miracl-v1.0-ja-dev.tsv rank.trec nDCG@10 RR
```

```bash
nDCG@10 0.6588
RR      0.6836
```

## 追加学習

```bash
poetry run python -m tevatron.driver.train \
  --output_dir model_ft \
  --dataset_name mpkato/miracl-japanese-small \
  --model_name_or_path aken12/dpr-japanese \
  --do_train \
  --save_steps 20000 \
  --fp16 \
  --per_device_train_batch_size 64 \
  --train_n_passages 2 \
  --learning_rate 1e-5 \
  --q_max_len 32 \
  --p_max_len 128 \
  --num_train_epochs 5
```

## コーパス中の文書をエンコーディング
```bash
poetry run python -m tevatron.driver.encode \
  --output_dir=temp \
  --model_name_or_path model_ft \
  --fp16 \
  --per_device_eval_batch_size 156 \
  --p_max_len 128 \
  --dataset_name mpkato/miracl-japanese-small-corpus \
  --encoded_save_path ft_corpus_emb.pkl
```

## クエリのエンコーディング
```bash
poetry run python -m tevatron.driver.encode \
  --output_dir=temp \
  --model_name_or_path model_ft \
  --fp16 \
  --per_device_eval_batch_size 156 \
  --dataset_name mpkato/miracl-japanese-small/dev \
  --encoded_save_path ft_query_emb.pkl \
  --q_max_len 32 \
  --encode_is_qry
```

## 近似最近傍探索
```bash
poetry run python -m tevatron.faiss_retriever \
  --query_reps ft_query_emb.pkl \
  --passage_reps ft_corpus_emb.pkl \
  --depth 100 \
  --batch_size -1 \
  --save_text \
  --save_ranking_to ft_rank.txt
```

## 出力をTRECフォーマットに変換
```bash
poetry run python -m tevatron.utils.format.convert_result_to_trec \
  --input ft_rank.txt \
  --output ft_rank.trec
```

## 評価

```bash
poetry run ir_measures qrels.miracl-v1.0-ja-dev.tsv ft_rank.trec nDCG@10 RR
```

```bash
nDCG@10 0.6980
RR      0.7258
```