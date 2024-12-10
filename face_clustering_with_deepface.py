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
    日本語などの非英語文字を含むパスを処理可能な形式に変換
    """
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    
    # 一時的な英語名のファイルを作成
    temp_path = temp_dir / f"temp_{hash(str(img_path))}.jpg"
    if not temp_path.exists():
        shutil.copy(img_path, temp_path)
    
    return temp_path

def get_embedding(img_path, cache_file="embeddings_cache.pkl"):
    try:
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                cache = pickle.load(f)
                if str(img_path) in cache:
                    return cache[str(img_path)]
        else:
            cache = {}

        # 安全なパスに変換
        safe_img_path = safe_path(img_path)

        # 顔検出時のパラメータを調整
        face_objs = DeepFace.extract_faces(
            img_path=str(safe_img_path), 
            align=True,
            detector_backend="retinaface",
            enforce_detection=False
        )

        if len(face_objs) != 1:
            return None

        # 特徴抽出
        embedding_objs = DeepFace.represent(
            img_path=str(safe_img_path),
            detector_backend="retinaface",
            model_name="Facenet512",
            align=True,
            enforce_detection=False,
            normalization="base"
        )

        if len(embedding_objs) != 1:
            return None

        embedding = embedding_objs[0]["embedding"]
        
        # 前処理を適用
        embedding = embedding - np.mean(embedding)
        embedding = embedding / np.std(embedding)
        embedding = embedding / np.linalg.norm(embedding)
        
        cache[str(img_path)] = embedding
        with open(cache_file, 'wb') as f:
            pickle.dump(cache, f)

        return embedding

    except Exception as e:
        logging.error(f"Error getting embedding for {img_path}: {str(e)}")
        return None

def calculate_similarity(emb1, emb2):
    """
    より安定した類似度計算
    """
    # コサイン類似度
    cosine_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    # [-1, 1]から[0, 1]の範囲に変換
    similarity = (cosine_sim + 1) / 2
    return similarity

def process_embeddings(input_images):
    embeddings = {}
    for img_path in input_images:
        embedding = get_embedding(img_path)
        if embedding is not None:
            embeddings[img_path] = embedding
    return embeddings

def calculate_similarity_matrix(input_images, embeddings):
    n = len(input_images)
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            img1 = input_images[i]
            img2 = input_images[j]
            if img1 in embeddings and img2 in embeddings:
                similarity = calculate_similarity(embeddings[img1], embeddings[img2])
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
                logging.debug(f"Similarity between {img1.name} and {img2.name}: {similarity:.3f}")
    
    return similarity_matrix

def create_groups(input_images, similarity_matrix, output_dir, threshold):
    n = len(input_images)
    processed_images = set()
    group_counter = 1
    
    for i in range(n):
        if input_images[i] in processed_images:
            continue

        current_group = output_dir / f"person_{group_counter}"
        current_group.mkdir(exist_ok=True)
        shutil.copy(input_images[i], current_group)
        processed_images.add(input_images[i])

        # 類似度の高い画像を同じグループに追加
        for j in range(n):
            if i == j or input_images[j] in processed_images:
                continue

            if similarity_matrix[i, j] > threshold:
                shutil.copy(input_images[j], current_group)
                processed_images.add(input_images[j])
                logging.info(f"Grouped {input_images[j].name} with {input_images[i].name} "
                           f"(similarity: {similarity_matrix[i, j]:.3f})")

        group_counter += 1

def analyze_similarities(similarity_matrix):
    """
    類似度行列の統計分析を行う
    """
    # 上三角行列から類似度値を取得（対角要素を除く）
    similarities = []
    n = len(similarity_matrix)
    for i in range(n):
        for j in range(i+1, n):
            if similarity_matrix[i, j] > 0:
                similarities.append(similarity_matrix[i, j])

    if similarities:
        return {
            'min': min(similarities),
            'max': max(similarities),
            'mean': np.mean(similarities),
            'median': np.median(similarities),
            'std': np.std(similarities)
        }
    return None

def main():
    try:
        input_dir = Path("input")
        output_dir = Path("output")
        temp_dir = Path("temp")
        
        # 既存のディレクトリをクリア
        for dir_path in [output_dir, temp_dir]:
            if dir_path.exists():
                shutil.rmtree(dir_path)
            dir_path.mkdir()

        # 入力画像のリストを作成
        input_images = list(input_dir.glob("*.jpg"))
        if not input_images:
            logging.error("No input images found in the input directory")
            return

        logging.info(f"Processing {len(input_images)} images...")

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
        # 一時ディレクトリの削除
        if temp_dir.exists():
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    SIMILARITY_THRESHOLD = 0.6  # 新しい類似度計算方法に合わせて閾値を調整
    main()