"""
顔画像分類プログラム

このプログラムは、複数の人物の顔画像を入力として受け取り、
同じ人物の画像をグループ化して分類するものです。

主な機能：
1. 顔の検出と特徴抽出
2. 顔画像間の類似度計算
3. 類似度に基づく画像のグループ化

使用方法：
1. 分類したい顔画像を「input」ディレクトリに配置
2. プログラムを実行
3. 分類結果が「output」ディレクトリに出力される
"""

from pathlib import Path
import logging
import shutil
from deepface import DeepFace
import numpy as np
import os
from multiprocessing import Pool, cpu_count
import pickle
from collections import defaultdict


def safe_path(img_path):
    """
    日本語のファイル名を扱うための関数

    Args:
        img_path: 元の画像パス（日本語を含む可能性あり）

    Returns:
        一時的な英語名のファイルパス
    """
    temp_dir = Path("temp")  # 一時ディレクトリを作成
    temp_dir.mkdir(exist_ok=True)

    # ファイル名をハッシュ値に変換して一時ファイルを作成
    # これにより日本語ファイル名を英数字のみの名前に変換
    temp_path = temp_dir / f"temp_{hash(str(img_path))}.jpg"
    if not temp_path.exists():
        shutil.copy(img_path, temp_path)

    return temp_path


def get_embedding(img_path, cache_file="embeddings_cache.pkl"):
    """
    画像から顔の特徴ベクトルを抽出する関数

    特徴ベクトルとは：
    - 顔の特徴を数値の配列として表現したもの
    - この値を比較することで顔の類似度を計算できる

    Args:
        img_path: 画像ファイルのパス
        cache_file: 計算結果をキャッシュするファイル名

    Returns:
        顔の特徴ベクトル（numpy配列）
    """
    try:
        # キャッシュがある場合はそれを使用（処理時間短縮のため）
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                cache = pickle.load(f)
                if str(img_path) in cache:
                    return cache[str(img_path)]
        else:
            cache = {}

        # 日本語ファイル名対応
        safe_img_path = safe_path(img_path)

        # 画像から顔を検出
        # detector_backend: 顔検出に使用するAIモデル
        # align: 顔の向きを正面に補正
        face_objs = DeepFace.extract_faces(
            img_path=str(safe_img_path),
            align=True,
            detector_backend="retinaface",
            enforce_detection=False,  # 顔検出に失敗しても処理を続行
        )

        # 1つの顔のみを処理
        if len(face_objs) != 1:
            return None

        # 検出した顔から特徴ベクトルを抽出
        # Facenet512: 高精度な顔認識AI モデル
        embedding_objs = DeepFace.represent(
            img_path=str(safe_img_path),
            detector_backend="retinaface",
            model_name="Facenet512",
            align=True,
            enforce_detection=False,
            normalization="base",
        )

        if len(embedding_objs) != 1:
            return None

        embedding = embedding_objs[0]["embedding"]

        # 特徴ベクトルの正規化（値の範囲を調整）
        # これにより異なる画像間の比較が容易になる
        embedding = embedding - np.mean(embedding)  # 平均を0に
        embedding = embedding / np.std(embedding)  # 標準偏差を1に
        embedding = embedding / np.linalg.norm(embedding)  # ベクトルの長さを1に

        # 結果をキャッシュに保存
        cache[str(img_path)] = embedding
        with open(cache_file, "wb") as f:
            pickle.dump(cache, f)

        return embedding

    except Exception as e:
        logging.error(f"Error getting embedding for {img_path}: {str(e)}")
        return None


def calculate_similarity(emb1, emb2):
    """
    2つの顔特徴ベクトル間の類似度を計算する関数

    Args:
        emb1, emb2: 比較する2つの特徴ベクトル

    Returns:
        similarity: 0から1の間の類似度（1に近いほど似ている）
    """
    # コサイン類似度の計算
    cosine_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    # 類似度を0-1の範囲に変換
    similarity = (cosine_sim + 1) / 2
    return similarity


def process_embeddings(input_images):
    """
    すべての入力画像から特徴ベクトルを抽出する関数

    Args:
        input_images: 入力画像のパスのリスト

    Returns:
        embeddings: 画像パスと特徴ベクトルの対応辞書
    """
    embeddings = {}
    for img_path in input_images:
        embedding = get_embedding(img_path)
        if embedding is not None:
            embeddings[img_path] = embedding
    return embeddings


def calculate_similarity_matrix(input_images, embeddings):
    """
    すべての画像ペア間の類似度を計算する関数

    Args:
        input_images: 入力画像のリスト
        embeddings: 特徴ベクトルの辞書

    Returns:
        similarity_matrix: 画像間の類似度を格納した行列
    """
    n = len(input_images)
    similarity_matrix = np.zeros((n, n))

    # すべての画像ペアの組み合わせについて類似度を計算
    for i in range(n):
        for j in range(i + 1, n):
            img1 = input_images[i]
            img2 = input_images[j]
            if img1 in embeddings and img2 in embeddings:
                similarity = calculate_similarity(embeddings[img1], embeddings[img2])
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity

    return similarity_matrix


def create_groups(input_images, similarity_matrix, output_dir, threshold):
    """
    類似度に基づいて画像をグループ化する関数

    Args:
        input_images: 入力画像のリスト
        similarity_matrix: 類似度行列
        output_dir: 出力ディレクトリ
        threshold: グループ化の閾値
    """
    n = len(input_images)
    processed_images = set()
    group_counter = 1

    # 各画像について処理
    for i in range(n):
        if input_images[i] in processed_images:
            continue

        # 新しいグループを作成
        current_group = output_dir / f"person_{group_counter}"
        current_group.mkdir(exist_ok=True)
        shutil.copy(input_images[i], current_group)
        processed_images.add(input_images[i])

        # 類似度が閾値より高い画像を同じグループに追加
        for j in range(n):
            if i == j or input_images[j] in processed_images:
                continue

            if similarity_matrix[i, j] > threshold:
                shutil.copy(input_images[j], current_group)
                processed_images.add(input_images[j])
                logging.info(
                    f"Grouped {input_images[j].name} with {input_images[i].name} "
                    f"(similarity: {similarity_matrix[i, j]:.3f})"
                )

        group_counter += 1


def analyze_similarities(similarity_matrix):
    """
    類似度の統計情報を計算する関数

    Args:
        similarity_matrix: 類似度行列

    Returns:
        stats: 統計情報を含む辞書
    """
    similarities = []
    n = len(similarity_matrix)
    for i in range(n):
        for j in range(i + 1, n):
            if similarity_matrix[i, j] > 0:
                similarities.append(similarity_matrix[i, j])

    if similarities:
        return {
            "min": min(similarities),
            "max": max(similarities),
            "mean": np.mean(similarities),
            "median": np.median(similarities),
            "std": np.std(similarities),
        }
    return None


def main():
    """
    メイン処理を行う関数

    処理の流れ：
    1. 入力画像の読み込み
    2. 特徴ベクトルの抽出
    3. 類似度の計算
    4. グループ化
    5. 結果の出力
    """
    try:
        # ディレクトリの設定
        input_dir = Path("input")
        output_dir = Path("output")
        temp_dir = Path("temp")

        # 既存のディレクトリをクリア
        for dir_path in [output_dir, temp_dir]:
            if dir_path.exists():
                shutil.rmtree(dir_path)
            dir_path.mkdir()

        # 入力画像の読み込み
        input_images = list(input_dir.glob("*.jpg"))
        if not input_images:
            logging.error("No input images found in the input directory")
            return

        logging.info(f"Processing {len(input_images)} images...")

        # 特徴ベクトルの抽出
        embeddings = process_embeddings(input_images)
        if not embeddings:
            logging.error("Failed to extract embeddings from any images")
            return

        # 類似度行列の計算
        similarity_matrix = calculate_similarity_matrix(input_images, embeddings)

        # 類似度の統計分析
        stats = analyze_similarities(similarity_matrix)
        if stats:
            logging.info("\nSimilarity Statistics:")
            logging.info(f"Min similarity: {stats['min']:.3f}")
            logging.info(f"Max similarity: {stats['max']:.3f}")
            logging.info(f"Mean similarity: {stats['mean']:.3f}")
            logging.info(f"Median similarity: {stats['median']:.3f}")
            logging.info(f"Std similarity: {stats['std']:.3f}")

        # グループの作成
        create_groups(input_images, similarity_matrix, output_dir, SIMILARITY_THRESHOLD)

        # 結果の表示
        logging.info(f"\nGrouping Results:")
        for group_dir in output_dir.glob("person_*"):
            image_count = len(list(group_dir.glob("*.jpg")))
            logging.info(f"{group_dir.name}: {image_count} images")

    except Exception as e:
        logging.error(f"Main process error: {str(e)}")
        raise
    finally:
        # 一時ファイルの削除
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    SIMILARITY_THRESHOLD = 0.6  # 新しい類似度計算方法に合わせて閾値を調整
    main()
