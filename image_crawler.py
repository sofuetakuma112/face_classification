# ライブラリを呼び出す
from icrawler.builtin import BingImageCrawler
from icrawler.builtin import ImageDownloader
from pathlib import Path

# カスタムダウンローダーを作成
class PrefixedDownloader(ImageDownloader):
    def get_filename(self, task, default_ext):
        filename = super().get_filename(task, default_ext)
        return f"{self.prefix}_{filename}"

# キーワードを入力
keyword = str(input("名前を打ってね！>>>"))
max_num = int(input("何枚集める？>>>"))

# クローラーを生成、保存先などを指定
bing_crawler = BingImageCrawler(
    downloader_cls=PrefixedDownloader,
    downloader_threads=4,
    storage={"root_dir": "images"}
)

# プレフィックスを設定
bing_crawler.downloader.prefix = keyword

# 画像収集を実行
bing_crawler.crawl(
    keyword=keyword,
    filters=None,
    offset=0,
    max_num=max_num,
)