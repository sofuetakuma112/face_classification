.PHONY: list-dirs-with-multiple-files clean help

# デフォルトターゲット: ヘルプを表示
.DEFAULT_GOAL := help

# ファイル数が2つ以上のディレクトリをリストアップ
list-dirs-with-multiple-files:
	@for dir in output/person_*; do \
		count=$$(ls -1 "$$dir" | wc -l); \
		if [ $$count -ge 2 ]; then \
			echo "$$dir: $$count files"; \
		fi; \
	done

# 生成されたファイルやキャッシュを削除
clean:
	@echo "Cleaning up..."
	@rm -f embeddings_cache.pkl
	@rm -f photo_sorter.log
	@echo "Clean up completed."

# ヘルプメッセージを表示
help:
	@echo "Available targets:"
	@echo "  list-dirs-with-multiple-files  - List directories containing 2 or more files"
	@echo "  clean                          - Remove generated files and cache"
	@echo "  help                           - Show this help message"