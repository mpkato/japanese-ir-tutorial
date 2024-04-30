# 環境設定

## サーバの利用

本チュートリアルはGPUが備わっているLinux環境を前提にしています．
CUDA 11.8用の`torch`をインストールする前提で環境構築をしていますが，
`pyproject.toml`を変更すれば他のバージョンでも動作すると思われます．

## pyenv

https://github.com/pyenv/pyenv

サーバ上でローカルのPython環境をインストールするためにpyenvをインストールする．複数の異なるバージョンのPythonをインストールすることもできる．

1. インストール（ターミナルで実行）
```bash
$ curl https://pyenv.run | bash
```

2. インストール後の設定（ターミナルで実行）
```bash
$ echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
$ echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
$ echo 'eval "$(pyenv init -)"' >> ~/.bashrc
```
上記を実行すると，`~/.bashrc`にて以下の行が追加される：
```
export PYENV_ROOT="$HOME/.pyenv"
command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```

`~/.bashrc`中のコマンドはログイン時に自動的に実行される．上記が実行されると，
pyenvコマンドの自動補完が有効となり便利に利用できる．

## Pythonのインストール

Anaconda バージョン`anaconda3-2024.02-1`をインストールする．Python 3.11 (pyproject.tomlで指定) が入っていればどのバージョンでも良い．

```bash
$ pyenv install anaconda3-2024.02-1
```

`pyenv i`のあと，もしくは，`pyenv install `のあとに，TABキーを1度か2度押すと補完が可能なので試してみると良い．補完機能は打ち間違いを減らす効果や効率を上げる効果がある．使えるときは積極的に使うと良い．

インストール完了後に，以下のコマンドによって，ユーザデフォルトのPythonを，Anaconda バージョン`anaconda3-2024.02-1`に切り替えておく．
```bash
$ pyenv global anaconda3-2024.02-1
```

以下のコマンドを実行し，以下のように表示されれば問題なく切り替えられている．
```bash
$ python --version
Python 3.11.7
```

## レポジトリのクローン

チュートリアル用のファイルを含むレポジトリをクローン（サーバ上にコピー）します．

```bash
$ git clone https://github.com/mpkato/japanese-ir-tutorial.git
$ cd japanese-ir-tutorial
```

## poetry

https://github.com/python-poetry/poetry

poetryはパッケージ管理ツールであり，異なる環境でコードを実行するときに有用である．
特に，本チュートリアルはいろいろなパッケージに依存しているため，poetryを利用してパッケージのインストールを簡単にできるようにしている．

1. poetryのインストール
```bash
$ pip install poetry
```

2. パッケージのインストール

上記でクローンしたレポジトリ内の，`pyproject.toml`があるディレクトリで実行してください．

```bash
$ poetry install 
```

＊「error: can't find Rust compiler」というエラーがでる場合には，Rustコンパイラをインストールしてください．

必要なパッケージをインストールしたあとは，以下のようにPythonを実行することで，インストールされたパッケージを参照して利用することができます．

```bash
$ poetry run python [filepath]
```

単に`python [filepath]`と実行してもインストールしたパッケージを参照してくれないので気をつけてください．毎回，`poetry run`をつけるのが面倒だと感じる場合には，`poetry shell`を実行しておけば`poetry run`をつける必要はなくなります．