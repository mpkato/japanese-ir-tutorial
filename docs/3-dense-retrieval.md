# 教師あり密検索

PyTerrierでは伝統的な単語一致による検索を行い性能の評価を行いました．
次に，BERTなどの大規模言語モデルを利用した密検索モデルの性能を評価してみましょう．

密検索の実行には，PyTerrierを利用することもできますが，ここではコマンドラインでの実行がサポートされている[Tevatron](https://github.com/texttron/tevatron/)を利用していきます．

以下では，まず既存の密検索モデルを利用して検索を実行してみます．その後，MIRACLデータセットに含まれる日本語の学習データを利用して，この密検索モデルを追加学習することで性能を向上させてみます．

## コーパス中の文書をエンコーディング

密検索モデルとして，[DPR (Dense Passage Retrieval)](https://arxiv.org/abs/2004.04906)を利用します．この検索モデルは，もともと質問応答タスクを解くために提案されたものですが，今日の多くの密検索モデルの基礎を含むため，密検索モデルを理解するために有用です．

DPRはBERTなどの大規模言語モデルを利用することで，クエリと文書を，それらを表現する数百次元のベクトルに変換（エンコード）します．検索時には，与えられたクエリのベクトルと文書のベクトルの類似度を計算し，その類似度をスコアとして扱って，文書をスコアの降順で並び替えることで順位付けしいます．ただし，文書数が多い場合には，各文書のベクトルとクエリのベクトルの類似度をいちいち計算するのに時間がかかるため，近似最近傍探索と呼ばれるアルゴリズムを利用します．このアルゴリズムでは，与えられたベクトルと最も類似するベクトルを高速に探すことができます．ただし，その名の通り，得られる解は近似解となります（例えば，得られた10件の最も類似するベクトルの内，1件は50番目に最も類似するベクトルであった，ということがありえます）．

DPRを実行するために，まず以下のコマンドを実行することで，文書をすべてベクトルに変換します．

```bash
$ poetry run python -m tevatron.driver.encode \
  --output_dir=temp \
  --model_name_or_path aken12/dpr-japanese \
  --fp16 \
  --per_device_eval_batch_size 156 \
  --p_max_len 128 \
  --dataset_name mpkato/miracl-japanese-small-corpus \
  --encoded_save_path embeddings/corpus_emb.pkl
```

ここでは，Hugging Faceで公開されている日本語用DPRのモデルである`aken12/dpr-japanese`を利用しています．文書データは`mpkato/miracl-japanese-small-corpus`を指定しており，ベクトルは`embeddings/corpus_emb.pkl`に保存されます．

なお，2024/4/29時点では，複数GPUを搭載するサーバでは以下のエラーがでるようです：
```
NotImplementedError: Multi-GPU encoding is not supported.
```
その場合には，コマンドの先頭に`CUDA_VISIBLE_DEVICES=0`をつけるようにしてください．
e.g., `CUDA_VISIBLE_DEVICES=0 poetry run python ...`.


## クエリのエンコーディング

文書と同様にクエリもベクトルに変換します．

```bash
$ poetry run python -m tevatron.driver.encode \
  --output_dir=temp \
  --model_name_or_path aken12/dpr-japanese \
  --fp16 \
  --per_device_eval_batch_size 156 \
  --dataset_name mpkato/miracl-japanese-small/dev \
  --encoded_save_path embeddings/query_emb.pkl \
  --q_max_len 32 \
  --encode_is_qry
```

`dataset_name`に`mpkato/miracl-japanese-small/dev`と指定することで`dev` splitを指定できるようです．クエリのベクトルは`embeddings/query_emb.pkl`に保存されます．

## 近似最近傍探索

近似最近傍探索によって，各クエリに対して検索を実行します．

```bash
$ poetry run python -m tevatron.faiss_retriever \
  --query_reps embeddings/query_emb.pkl \
  --passage_reps embeddings/corpus_emb.pkl \
  --depth 100 \
  --batch_size -1 \
  --save_text \
  --save_ranking_to results/dpr.txt
```

検索結果は`results/dpr.txt`に保存されますが，こちらはTRECフォーマットになっていません．そのため，以下のコマンドを実行することでTRECフォーマットに変換します．

```bash
$ poetry run python -m tevatron.utils.format.convert_result_to_trec \
  --input results/dpr.txt \
  --output results/dpr.trec
```

results/dpr.txt
```txt
0	2681119#1	227.35012817382812
0	2681119#0	225.04696655273438
0	343986#0	222.68771362304688
0	2681119#16	219.94061279296875
0	1113240#1	219.3719024658203
```

results/dpr.trec
```txt
0 Q0 2681119#1 1 227.35012817382812 dense
0 Q0 2681119#0 2 225.04696655273438 dense
0 Q0 343986#0 3 222.68771362304688 dense
0 Q0 2681119#16 4 219.94061279296875 dense
0 Q0 1113240#1 5 219.3719024658203 dense
```

## 評価

これまでと同様に評価を行います．BM25やTF-IDFよりも高い性能が得られていることがわかります．

```bash
$ poetry run ir_measures qrels.miracl-v1.0-ja-dev.tsv results/dpr.trec nDCG@10 RR
```

```bash
nDCG@10 0.6588
RR      0.6836
```

## 追加学習

既存の日本語DPRモデルを，MIRACLの学習用データで追加学習することによって，性能を向上させてみましょう．以下の例では，`aken12/dpr-japanese`を追加学習して`models/dpr-japanese-ft`というディレクトリに追加学習したモデルを保存しています．

```bash
$ poetry run python -m tevatron.driver.train \
  --output_dir models/dpr-japanese-ft \
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

追加学習したモデルを利用して文書とクエリのエンコーディングを行います．
上記のコードとの違いは，`--model_name_or_path`に`models/dpr-japanese-ft`を指定するところ，`--encoded_save_path`のファイル名を変えているところです．

コーパス中の文書をエンコーディング
```bash
$ poetry run python -m tevatron.driver.encode \
  --output_dir=temp \
  --model_name_or_path models/dpr-japanese-ft \
  --fp16 \
  --per_device_eval_batch_size 156 \
  --p_max_len 128 \
  --dataset_name mpkato/miracl-japanese-small-corpus \
  --encoded_save_path embeddings/ft_corpus_emb.pkl
```

クエリのエンコーディング
```bash
$ poetry run python -m tevatron.driver.encode \
  --output_dir=temp \
  --model_name_or_path models/dpr-japanese-ft \
  --fp16 \
  --per_device_eval_batch_size 156 \
  --dataset_name mpkato/miracl-japanese-small/dev \
  --encoded_save_path embeddings/ft_query_emb.pkl \
  --q_max_len 32 \
  --encode_is_qry
```

近似最近傍探索も同様に行います．`--query_reps`と`--passage_reps`のパス，および，`--save_ranking_to`のパスを変えていることに注意してください．
```bash
$ poetry run python -m tevatron.faiss_retriever \
  --query_reps embeddings/ft_query_emb.pkl \
  --passage_reps embeddings/ft_corpus_emb.pkl \
  --depth 100 \
  --batch_size -1 \
  --save_text \
  --save_ranking_to results/ft_dpr.txt
```

出力をTRECフォーマットに変換
```bash
$ poetry run python -m tevatron.utils.format.convert_result_to_trec \
  --input results/ft_dpr.txt \
  --output results/ft_dpr.trec
```

評価
```bash
$ poetry run ir_measures qrels.miracl-v1.0-ja-dev.tsv results/ft_dpr.trec nDCG@10 RR
```

```bash
nDCG@10	0.7018
RR	0.7314
```


## 追加学習の効果

追加学習前と後の効果は以下のようになりました：

|  | nDCG@10 | RR |
| ---- | ---- | ---- |
| 追加学習前 | 0.6588 | 0.6836 |
| 追加学習後 | 0.7018 | 0.7314 |

追加学習後の方が性能が向上しているのがわかります．一般に，あるデータセットでの性能はそのデータセット外で学習されたモデルよりも，そのデータセット自体で学習したモデルの方が高くなります．前者をZero-shot設定やOut-of-domain設定，後者をIn-domain設定と呼ぶことがあります．