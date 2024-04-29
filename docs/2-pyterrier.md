# PyTerrierによる検索

[PyTerrier](https://github.com/terrier-org/pyterrier)は情報検索実験用のプラットフォームであり，Javaで書かれたTerrierを内部的に利用しています．類似するプラットフォームとして[Pyserini](https://github.com/castorini/pyserini)があります．

PyTerrierはTerrierをコアにした老舗で，Pythonからの実行が容易，
一方で，Pyseriniは再現実験を重視しており，コマンドラインからの実行が容易，というイメージです．

## 単語一致による検索

単語一致による検索モデルであるBM25やTF-IDFによる検索を実行してみましょう．
以下の例では，次の文書コレクション（検索対象の文書集合），および，トピック（≒クエリ）を利用します．
- 文書コレクション: [miracl-japanese-small-corpus](https://huggingface.co/datasets/mpkato/miracl-japanese-small-corpus)
- トピック: [miracl-japanese-small](https://huggingface.co/datasets/mpkato/miracl-japanese-small) (評価には`dev` splitを利用)

上記リンクからデータセットのフォーマットや中身を確認できるので，どのようなデータを実際に利用するのかを確かめてください．

### 転置索引の作成

単語一致による検索モデルでは，まず，コレクション中の文書をトークナイズして，転置索引を作る処理が必要になります．Pythonで簡単に書くこともできますが，まずは，`scripts/terrier_inverted_index.py`に用意されたコードを利用して動作を確認してみます．

このようなスクリプトは，多くの場合，内部で`argparse`というPythonパッケージを利用しており，引数として`-h`を指定することで，ヘルプメッセージを表示させることができます．

```bash
$ poetry run python scripts/terrier_inverted_index.py -h
usage: terrier_inverted_index.py [-h] [--split SPLIT] dataset_name index_filepath

Creates an Terrier inverted index.

positional arguments:
  dataset_name    the document collection name of a dataset to be used for datasets.load_dataset.
  index_filepath  the directory in which the inverted index is stored.

options:
  -h, --help      show this help message and exit
  --split SPLIT   the split of the dataset. (default: None)
```

- `dataset_name`: Pythonパッケージdatasetsの`load_dataset`という関数の引数を指定します．この関数で読み込めるようなデータであって，かつ，`docid`と`text`というフィールドをもっているようなデータであれば何でも良いです．ここでは，[MIRACL](https://huggingface.co/datasets/miracl/miracl)という多言語検索用のデータセットから日本語版のみを抽出して，さらに文書数を十分に減らしたデータ`mpkato/miracl-japanese-small-corpus`を用意してあるのでこちらを利用します．このデータは[Hugging Face](https://huggingface.co/)にアップロードされているデータセットであり，`load_dataset`関数でダウンロードして読み込むことができます．
- `index_filepath`: 転置索引をファイルとして保存するディレクトリを指定します．どこでも良いですが，空き容量が十分にあって，読み込み速度が早いディスク上にあるファイルパス（ネットワークで共有されたディスクなどではなくローカルディスクなど）を指定してください．
- `--split`: オプショナルな引数です．`load_dataset`などで読み込まれたデータは`train`, `dev`, `test`などのデータ（それぞれをsplitと呼ぶ）に分かれている場合があり，どのsplitを使用するかをこの引数で指定します．今回利用する`mpkato/miracl-japanese-small-corpus`は`train`splitのみを含むデータなのですが，splitを含んでいるため指定する必要があります．

上記の説明を踏まえて，以下のコマンドを実行することで，転置索引を構築することが可能です．
```bash
$ poetry run python scripts/terrier_inverted_index.py \
    mpkato/miracl-japanese-small-corpus \
    ./miracl_index \
    --split train
```

十分に文書数を減らした文書コレクションであるため数分で転置索引の構築が完了します．

### 転置索引に基づく検索

次に，用意された各トピック（クエリ）に対して，文書コレクションから文書を検索してみます．
先ほどと同様に，`scripts/terrier_retrieve.py`というスクリプトが用意されていますので，
こちらを利用して各トピック（クエリ）に対して，BM25およびTF-IDF検索モデルを使って，
100件の文書を検索してその結果をファイルに出力します．

```bash
$ poetry run python scripts/terrier_retrieve.py -h
usage: terrier_retrieve.py [-h] [--split SPLIT] [--wmodel WMODEL] [--num_results NUM_RESULTS] dataset_name index_filepath output_filepath

Retrieves documents from an Terrier inverted index.

positional arguments:
  dataset_name          the dataset containing query_id and query.
  index_filepath        the directory including the inverted index.
  output_filepath       the filepath at which the result file is created.

options:
  -h, --help            show this help message and exit
  --split SPLIT         the split of the dataset. (default: None)
  --wmodel WMODEL       the name of the weighting model. See http://terrier.org/docs/current/javadoc/org/terrier/matching/models/package-summary.html (default: BM25)
  --num_results NUM_RESULTS
                        the number of documents to be retrieved for each query. (default: 100)
```

- `dataset_name`: `terrier_inverted_index.py`と同様に，`load_dataset`の引数を指定します．データは，`query_id`と`query`というフィールドを含むことが想定されています．また，splitがある場合には，`--split`オプションで指定します．
- `index_filepath`: `terrier_inverted_index.py`で作成した転置索引のファイルパスを指定します．
- `output_filepath`: 検索結果の保存先のファイルパスを指定します．
- `--wmodel`: 様々な重み付けモデル（検索モデル）を指定できます．今回は，`BM25`と`TF_IDF`を指定します．

今回のチュートリアルでは，dev splitのトピック（クエリ）を利用します．トピックは学習用（train），開発用（dev），テスト用（test）に分かれていることがあり，評価には基本的にテスト用トピックを使用します．しかし，テスト用トピックが公開されてしまうと，公平な評価が難しくなったりするため，テスト用トピックが公開されていない場合もあります．そのような場合には，開発用トピックを利用して評価を行います．
以下のコマンドでBM25による検索結果を得ます（引数`--wmodel`のデフォルト値は`BM25`なので，省略することでBM25を指定しています）：

```bash
poetry run python scripts/terrier_retrieve.py \
    mpkato/miracl-japanese-small \
    ./miracl_index \
    ./results/bm25.trec \
    --split dev
```

以下のコマンドでTF-IDFによる検索結果を得ます：
```bash
poetry run python scripts/terrier_retrieve.py mpkato/miracl-japanese-small ./miracl_index ./results/tfidf.trec --split dev --wmodel TF_IDF
```

検索結果が出力されたら出力フォーマットを確認してみましょう．
以下では，`results/bm25.trec`の先頭から数行を示しています：
```txt
0 Q0 2681119#1 0 19.892995032344007 pyterrier
0 Q0 2875225#4 1 18.23801870613749 pyterrier
0 Q0 2681119#0 2 17.821968518812923 pyterrier
0 Q0 343986#0 3 13.277273149915517 pyterrier
0 Q0 2681119#14 4 11.090700266442642 pyterrier
```

このフォーマットはTRECフォーマットと呼ばれ，各行は左から順に以下の項目を表しています（e.g., https://github.com/joaopalotti/trectools）：
- クエリID
- "Q0"（利用されない）
- 文書ID
- 順位（0始まり）
- スコア（多くの検索モデルは文書にスコアをつけて降順に順位付けする）
- システム名

`results/bm25.trec`の例では，クエリIDが0のクエリに対して，1位から順に，
2681119#1, 2875225#4, 2681119#0, ...という文書が検索されたことを意味しています．
100行くらい下の方には，クエリIDが3のクエリに対する結果があります．

## 評価

検索モデルの良し悪しを評価するためには，適合性判定結果（通称qrel, qrels）が必要になります．今回利用しているMIRACLデータセットのdevトピック（クエリ）に対する適合性判定結果は以下のようにダウンロードできます．

```bash
$ wget https://huggingface.co/datasets/mpkato/miracl-japanese-small/raw/script/qrels.miracl-v1.0-ja-dev.tsv
```

中身は以下のようになっています：
```tsv
0	Q0	2681119#1	1
0	Q0	2681119#0	0
0	Q0	1113240#1	0
0	Q0	343986#0	0
0	Q0	2194424#0	0
```

このTSVファイルは，各行にTAB文字で区切られた以下の情報が含まれます：
- クエリID
- "Q0"
- 文書ID
- 適合度（場合によるが，0が不適合，1が適合．0, 1, 2など，多段階の場合もある）

今回のように，クエリが質問形式の場合には，その質問の回答を含むような文書が適合（1），
そうでない場合には，不適合（0）と判定されている場合が多いです．
ただし，適合性判定結果はあくまでも一部の文書のみに対する適合度を含んでおり，
適合性判定が行われなかった文書の適合性はわかりません．
評価をする上では，適合性判定が行われなかった文書は不適合文書として扱います．
そのため，適合性判定が十分でないデータセットにおいては，新規の検索モデルが過小評価されることに注意してください．

適合性判定を下にして，各クエリの検索結果を以下のように評価することができます：

```bash
$ poetry run ir_measures qrels.miracl-v1.0-ja-dev.tsv results/bm25.trec nDCG@10 RR
$ poetry run ir_measures qrels.miracl-v1.0-ja-dev.tsv results/tfidf.trec nDCG@10 RR
```

`ir_measures`コマンドは3つの引数を持ち，最後の引数は可変長引数となっています：
- qrels: 適合性判定ファイルのパス
- run: TRECフォーマットの検索結果
- measures: 評価指標

今回の例では，nDCG@10とRRという評価指標を用いています．それぞれの評価指標について，その計算方法について調べてみてください．

以下のような数値が得られているかと思います：

|  | nDCG@10 | RR |
| ---- | ---- | ---- |
| BM25 | 0.3175 | 0.3279 |
| TF-IDF | 0.4068 | 0.4211 |

今回利用したMIRACLの日本語データセットでは，TF-IDFの方がBM25よりも優れていることがわかります．ただし，実際に結論づけるためには，統計的検定が必要になることに注意してください．