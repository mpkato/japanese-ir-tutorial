# データセットの準備

Hugging Faceにアップロードされていないデータセットを利用したり，独自のデータセットを利用したりする場合があります．このような場面に備えて，`datasets`パッケージを用いたデータセット準備の仕方を体験してみましょう．

以下では，[NTCIR Data Search](https://ntcir.datasearch.jp/data_search_1/)のデータセットをダウンロードしてこれらを読み込めるようにしてみます．

以下のコマンドで文書コレクションをダウンロードし，また，bz2形式のファイルを解凍します：
```bash
$ wget https://huggingface.co/datasets/mpkato/ntcir_data_search/resolve/main/data_search_j_collection.jsonl.bz2
$ bunzip2 data_search_j_collection.jsonl.bz2
```

ダウンロードされた文書コレクションはそのままデータセットとして利用することも可能ですが，
これまで扱ってきた文書コレクション同様に，docid，title，textの3つのフィールドをもつデータに変換して保存してみましょう．
以下のように，datasetsパッケージを利用することでデータセットの保存や読み込みを簡単に行うことができます．

```python
import json
from datasets import Dataset

ORIGINAL_DATASET_FILEPATH = "./data_search_j_collection.jsonl"
DATASET_FILEPATH = "./datasets/data-search-j-corpus/data-search-j-corpus.parquet"

def generate_corpus_data():
    """
    Dataset.from_generatorメソッドでデータセットを作るためには，
    各データをdict型にしてyieldする関数を定義しこれを引数として与える．
    今回はdocid，title，textの3つのフィールドをもつデータセットを作るので，
    各データを {"docid": ..., "title": ..., "text": ...} という形式に変換しyieldする．
    """
    with open(ORIGINAL_DATASET_FILEPATH) as f:
        for line in f:
            """
            jsonl形式は1行1行がJSON形式の文字列を含むファイルフォーマットであり，
            各行をjson.loadsメソッドで読み込むことができる．
            """
            data = json.loads(line)
            """
            今回利用するデータは以下の形式である：
            {
                "id": "000031519435",
                "url": "https://www.e-stat.go.jp/stat-search/files?page=1&toukei=00200231&result_page=1&layout=dataset&stat_infid=000031519435",
                "attribution": "出典：政府統計の総合窓口(e-Stat)（https://www.e-stat.go.jp/）",
                "title": "地方公共団体の議会の議員及び長の所属党派別人員調 地方公共団体の議会の議員及び長の所属党派別人員調等（H20.12.31現在） 選挙執行件数 | ファイルから探す | 統計データを探す | 政府統計の総合窓口",
                "description": "地方公共団体の議会の議員及び長の所属党派別人員調 / 地方公共団体の議会の議員及び長の所属党派別人員調等（H20.12.31現在）",
                ...
            }
            特にtextはどのように対応付けてもよいが今回はdescriptionの値を用いる．
            """
            yield {
                "docid": data["id"],
                "title": data["title"],
                "text": data["description"],
            }


if __name__ == '__main__':
    """
    Dataset.from_generatorメソッドでデータセットを作るためには，
    各データをdict型にしてyieldする関数を定義しこれを引数として与える．
    """
    dataset = Dataset.from_generator(generate_corpus_data)
    """
    大規模なデータに適したparquet形式で保存する．
    """
    dataset.to_parquet(DATASET_FILEPATH)

```

評価用のトピックおよび適合性判定結果も以下のようにダウンロードします：
```bash
$ wget https://huggingface.co/datasets/mpkato/ntcir_data_search/resolve/main/data_search_j_test_topics.tsv
$ wget https://huggingface.co/datasets/mpkato/ntcir_data_search/resolve/main/data_search_j_test_qrels.txt
```

評価用トピックを含むファイル`data_search_j_test_topics.tsv`はTSV形式になっており，各列にクエリID，クエリを含みます．

**練習 1： `data_search_j_test_topics.tsv`をdatasetsパッケージを用いて読み込み，`query_id`と`query`の2つのフィールドをもつデータに変換した上で，`./datasets/data-search-j/test/data_search_j_test_topics.parquet`にparquet形式で保存してみましょう．**

**練習 2: `./datasets/data-search-j-corpus`をデータセットに指定して，転置索引を作成してみましょう．なお，読み込み先直下に配置されたファイルは`train` splitとして扱われるため注意しましょう．**

**練習 3: `./datasets/data-search-j`をデータセットに指定して，BM25によって検索をしてみましょう．なお，読み込み先直下の`test`ディレクトリの中にファイルが含まれているため，トピックは`test` splitとして読み込まれるため注意しましょう．**

**練習 4: 評価指標としてnDCG@10を用いて評価を行ってみましょう．ただし，適合性判定結果`data_search_j_test_qrels.txt`は"Q0"を含む列がなく，また，適合性がL0，L1，L2の3段階で示されているため，フォーマットを変える必要があります．**

