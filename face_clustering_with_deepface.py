from pathlib import Path
import logging
import shutil
from deepface import DeepFace
import numpy as np
import os
from multiprocessing import Pool, cpu_count
import pickle
from functools import partial
from collections import defaultdict

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 特徴ベクトルをキャッシュするための関数
def get_embedding(img_path, cache_file="embeddings_cache.pkl"):
    try:
        # キャッシュからの読み込みを試みる
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                cache = pickle.load(f)
                if str(img_path) in cache:
                    return cache[str(img_path)]
        else:
            cache = {}

        # 特徴ベクトルの抽出
        face_objs = DeepFace.extract_faces(
            img_path=str(img_path), 
            align=True, 
            detector_backend="retinaface"
        )

        if len(face_objs) != 1:
            return None

        embedding_objs = DeepFace.represent(
            img_path=str(img_path),
            detector_backend="retinaface",
            model_name="ArcFace",
            align=True,
        )

        if len(embedding_objs) != 1:
            return None

        embedding = embedding_objs[0]["embedding"]
        
        # キャッシュに保存
        cache[str(img_path)] = embedding
        with open(cache_file, 'wb') as f:
            pickle.dump(cache, f)

        return embedding

    except Exception as e:
        logging.error(f"Error getting embedding for {img_path}: {str(e)}")
        return None

# 画像を処理する関数
def process_image(args):
    img_path, output_dir, similarity_threshold, group_embeddings = args
    try:
        current_embedding = get_embedding(img_path)
        if current_embedding is None:
            return None

        min_distance = float('inf')
        best_match_dir = None

        # グループとの比較
        for group_dir, group_embs in group_embeddings.items():
            if not group_embs:
                continue
            
            distances = [np.linalg.norm(np.array(current_embedding) - np.array(emb))
                        for emb in group_embs if emb is not None]
            
            if distances:
                avg_distance = np.mean(distances)
                if avg_distance < min_distance:
                    min_distance = avg_distance
                    best_match_dir = group_dir

        return (img_path, min_distance, best_match_dir)

    except Exception as e:
        logging.error(f"Error processing {img_path}: {str(e)}")
        return None

def main():
    try:
        input_dir = Path("input")
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        SIMILARITY_THRESHOLD = 0.3

        # グループごとの特徴ベクトルを事前に計算
        group_embeddings = defaultdict(list)
        for person_dir in output_dir.glob("person_*"):
            for img_path in person_dir.glob("*.jpg"):
                embedding = get_embedding(img_path)
                if embedding is not None:
                    group_embeddings[person_dir].append(embedding)

        # 入力画像のリストを作成
        input_images = list(input_dir.glob("*.jpg"))
        
        # マルチプロセッシング用の引数を準備
        process_args = [(img_path, output_dir, SIMILARITY_THRESHOLD, dict(group_embeddings))
                       for img_path in input_images]

        # マルチプロセッシングで処理
        with Pool(processes=cpu_count()-1) as pool:
            results = pool.map(process_image, process_args)

        # 結果の処理
        for result in results:
            if result is None:
                continue

            img_path, min_distance, best_match_dir = result
            
            if min_distance < SIMILARITY_THRESHOLD and best_match_dir is not None:
                shutil.copy(img_path, best_match_dir)
                logging.info(f"Copied {img_path} to {best_match_dir} (distance: {min_distance:.3f})")
            else:
                new_group = output_dir / f"person_{len(list(output_dir.glob('person_*'))) + 1}"
                new_group.mkdir()
                shutil.copy(img_path, new_group)
                logging.info(f"Created new group {new_group} for {img_path}")

    except Exception as e:
        logging.error(f"Main process error: {str(e)}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()