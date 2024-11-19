1. venv環境の作成と有効化

   ```sh
    # venv環境を作成
    python3 -m venv myenv

    # venv環境を有効化
    # Windows の場合
    myenv\Scripts\activate
    # Mac/Linux の場合
    source myenv/bin/activate
   ```

2. 必要なパッケージのインストール

    ```sh
    # requirements.txtからパッケージをインストール
    pip install -r requirements.txt

    # Apple Silicon の場合、torch関連は別途インストール
    pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```

3. Ubuntuの場合、追加のシステムパッケージをインストール
    ```
    sudo apt install python3-dev build-essential default-libmysqlclient-dev
    sudo apt install libncursesw5-dev libgdbm-dev libc6-dev libctypes-ocaml-dev zlib1g-dev libsqlite3-dev tk-dev
    sudo apt install libssl-dev libmysqlclient-dev
    sudo apt install librust-libsodium-sys-dev
    ```

4. ディレクトリ構造の準備
    ```sh
    # 入力用ディレクトリの作成
    mkdir input

    # 出力用ディレクトリの作成（オプション）
    mkdir output
    ```

5. 画像ファイルの配置
    ```sh
    # 処理したい画像ファイルをinputディレクトリに配置
    ```

6. プログラムの実行
    ```sh
    python3 image_sorter.py
    ```

7. venv環境の終了
    ```sh
    deactivate
    ```

**結果は以下のディレクトリに出力されます：**

-   `output/OK/`: 適切と判断された画像
-   `output/NG/`: 不適切と判断された画像
-   `output/logs/`: 処理ログとサマリー情報

## 参考
- ModuleNotFoundError: No module named _ctypes エラーが出た場合
    - https://blog.hidenori.biz/1419
- VSCodeでPythonのmissing imports lintエラーが出た場合
    - https://qiita.com/tsukinomi/items/92e57da2ea4d0dd93957