# ライブラリを呼び出す
from icrawler.builtin import BingImageCrawler

# クローラーを生成、保存先などを指定（今回は'images'フォルダに指定）
bing_crawler = BingImageCrawler(downloader_threads=4, storage={"root_dir": "images"})

# キーワードや枚数を入力させてそれに応じて画像収集する
bing_crawler.crawl(
    keyword=str(input("名前を打ってね！>>>")),
    filters=None,
    offset=0,
    max_num=int(input("何枚集める？>>>")),
)
