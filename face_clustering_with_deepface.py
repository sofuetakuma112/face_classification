from pathlib import Path
import logging
import shutil
from deepface import DeepFace
import numpy as np
import os
from multiprocessing import Pool, cpu_count
import pickle
from collections import defaultdict

# グローバル定数
SIMILARITY_THRESHOLD = 0.80  # 類似度の閾値
GENDER_THRESHOLD = 0.95  # 性別判定の確実性の閾値
MIN_GROUP_SIZE = 3


def safe_path(img_path):
    """
    日本語を含むファイル名を安全に処理する関数
    """
    try:
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)

        # ファイル名をハッシュ値に変換して一時ファイルを作成
        temp_path = temp_dir / f"temp_{hash(str(img_path))}.jpg"

        if not temp_path.exists():
            # 画像ファイルのコピー
            shutil.copy2(img_path, temp_path)
            logging.debug(f"Created temporary file: {temp_path}")

        if not temp_path.exists():
            logging.error(f"Failed to create temporary file for {img_path}")
            return None

        return temp_path

    except Exception as e:
        logging.error(f"Error in safe_path for {img_path}: {str(e)}")
        return None


def get_embedding_and_gender(
    img_path, cache_file="embeddings_cache.pkl", gender_threshold=GENDER_THRESHOLD
):
    """
    画像から顔の特徴ベクトルと性別を抽出する関数
    """
    try:
        logging.debug(f"Starting processing for {img_path}")

        # 画像パスの確認
        if not os.path.exists(img_path):
            logging.error(f"Image file not found: {img_path}")
            return None

        # 日本語ファイル名を英数字に変換
        safe_img_path = safe_path(img_path)
        if safe_img_path is None or not os.path.exists(safe_img_path):
            logging.error(f"Safe path not created properly: {safe_img_path}")
            return None

        logging.debug(f"Processing image: {img_path}")

        # キャッシュの確認
        cache_data = {}
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                try:
                    cache_data = pickle.load(f)
                except EOFError:
                    logging.warning("Empty or corrupt cache file.")
                    cache_data = {}

        # 絶対パスと相対パスの両方でキャッシュを検索
        cached_result = cache_data.get(str(img_path.absolute())) or cache_data.get(
            str(img_path)
        )
        if cached_result:
            logging.debug(f"Using cached result for {img_path}")
            return cached_result

        # Step 1: 顔の検出と特徴抽出
        try:
            embedding_objs = DeepFace.represent(
                img_path=str(safe_img_path),
                detector_backend="retinaface",
                model_name="Facenet512",
                align=True,
                enforce_detection=False,
                normalization="base",
            )

            if not embedding_objs:
                logging.error(f"No face detected in {img_path}")
                return None

            if len(embedding_objs) > 1:
                logging.warning(
                    f"Multiple faces detected in {img_path}, using the first one"
                )

            embedding = np.array(embedding_objs[0]["embedding"], dtype=float)
            logging.debug(f"Successfully extracted embedding for {img_path}")

        except Exception as e:
            logging.error(f"Feature extraction failed for {img_path}: {str(e)}")
            return None

        # Step 2: 性別の分析
        try:
            gender_analysis = DeepFace.analyze(
                img_path=str(safe_img_path),
                actions=["gender"],
                detector_backend="retinaface",
                enforce_detection=False,
                align=True,
            )

            gender_scores = gender_analysis[0]["gender"]
            max_score = max(gender_scores.values())

            # より厳密な性別判定
            if max_score < gender_threshold:
                logging.warning(
                    f"Low confidence gender detection for {img_path}: {gender_scores}"
                )
                return None

            if gender_scores["Woman"] > gender_scores["Man"]:
                gender = "Woman"
            else:
                gender = "Man"

            logging.debug(
                f"Gender detection for {img_path}: {gender} (confidence: {max_score:.3f})"
            )

        except Exception as e:
            logging.error(f"Gender analysis failed for {img_path}: {str(e)}")
            return None

        # 特徴ベクトルの正規化
        embedding = embedding - np.mean(embedding)
        embedding = embedding / (np.std(embedding) + 1e-10)
        embedding = embedding / (np.linalg.norm(embedding) + 1e-10)

        # 結果の作成
        result = {
            "embedding": embedding,
            "gender": gender,
            "gender_confidence": max_score,
        }

        # 結果をキャッシュに保存
        cache_data[str(img_path.absolute())] = result
        cache_data[str(img_path)] = result  # 相対パスでもアクセスできるようにする
        with open(cache_file, "wb") as f:
            pickle.dump(cache_data, f)

        logging.debug(f"Successfully processed {img_path}")
        return result

    except Exception as e:
        logging.error(f"Error processing {img_path}: {str(e)}")
        return None


def calculate_similarity(emb1, emb2):
    """
    2つの顔特徴ベクトル間の類似度を計算する関数

    Args:
        emb1, emb2: 比較する2つの特徴ベクトル

    Returns:
        similarity: 0から1の間の類似度（1に近いほど似ている）
    """
    try:
        # ベクトルの正規化を確認
        emb1_norm = np.linalg.norm(emb1)
        emb2_norm = np.linalg.norm(emb2)

        if emb1_norm == 0 or emb2_norm == 0:
            return 0.0

        # コサイン類似度の計算
        cosine_sim = np.dot(emb1, emb2) / (emb1_norm * emb2_norm)
        # 類似度を0-1の範囲に正規化
        similarity = (cosine_sim + 1) / 2
        return float(similarity)
    except Exception as e:
        logging.error(f"Error calculating similarity: {str(e)}")
        return 0.0


def process_embeddings_parallel(input_images, num_processes):
    """
    マルチプロセスを使用して、すべての入力画像から特徴ベクトルと性別を抽出する関数
    """
    with Pool(processes=num_processes) as pool:
        results_list = pool.map(get_embedding_and_gender, input_images)

    results = {}
    successful_count = 0
    for img_path, result in zip(input_images, results_list):
        if result is not None and "embedding" in result and "gender" in result:
            # 特徴ベクトルが確実にnumpy配列であることを確認
            result["embedding"] = np.array(result["embedding"], dtype=float)

            # 両方のパスでアクセスできるようにする
            results[str(img_path.absolute())] = result.copy()
            results[str(img_path)] = result.copy()

            successful_count += 1
            logging.debug(f"Successfully processed {img_path.name}")
        else:
            logging.warning(f"Failed to process {img_path.name}")

    if successful_count == 0:
        logging.error("No embeddings were successfully extracted")
    else:
        logging.info(f"Successfully processed {successful_count} images")

    return results


def get_embedding(img_path, embeddings_data):
    """
    指定された画像の埋め込みベクトルを取得する関数
    """
    try:
        abs_path = str(img_path.absolute())
        rel_path = str(img_path)

        if abs_path in embeddings_data and "embedding" in embeddings_data[abs_path]:
            return embeddings_data[abs_path]["embedding"]
        elif rel_path in embeddings_data and "embedding" in embeddings_data[rel_path]:
            return embeddings_data[rel_path]["embedding"]
        return None
    except Exception as e:
        logging.error(f"Error getting embedding for {img_path}: {str(e)}")
        return None


def calculate_similarity_for_pair(pair, input_images, embeddings_data):
    """
    画像ペアの類似度を計算する関数 (トップレベル関数)
    """
    i, j = pair
    try:
        emb1 = get_embedding(input_images[i], embeddings_data)
        emb2 = get_embedding(input_images[j], embeddings_data)
        if emb1 is not None and emb2 is not None:
            similarity = calculate_similarity(emb1, emb2)
            return i, j, similarity
    except Exception as e:
        logging.error(f"Error calculating similarity for pair ({i}, {j}): {str(e)}")
    return i, j, 0.0  # エラーが発生した場合は類似度を0.0とする


def calculate_similarity_matrix_parallel(input_images, embeddings_data, num_processes):
    """
    マルチプロセスを使用して、すべての画像ペア間の類似度を計算する関数
    """
    n = len(input_images)
    similarity_matrix = np.zeros((n, n))

    # 類似度行列の計算
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(
            calculate_similarity_for_pair,
            [(pair, input_images, embeddings_data) for pair in pairs],
        )  # starmapに変更

    for i, j, similarity in results:
        similarity_matrix[i, j] = similarity
        similarity_matrix[j, i] = similarity

    for i in range(n):
        similarity_matrix[i, i] = 1.0  # 自分自身との類似度は1.0

    return similarity_matrix


def get_gender(img_path, embeddings_data):
    """性別情報を取得する補助関数"""
    abs_path = str(img_path.absolute())
    rel_path = str(img_path)

    if abs_path in embeddings_data and "gender" in embeddings_data[abs_path]:
        return embeddings_data[abs_path]["gender"]
    elif rel_path in embeddings_data and "gender" in embeddings_data[rel_path]:
        return embeddings_data[rel_path]["gender"]
    return None


def create_groups(
    input_images, similarity_matrix, embeddings_data, output_dir, threshold
):
    n = len(input_images)
    processed_images = set()
    groups = {}
    group_counter = 1

    # 各画像の特徴ベクトルの平均類似度を計算
    avg_similarities = {}
    for i in range(n):
        similarities = []
        img_gender = get_gender(input_images[i], embeddings_data)
        for j in range(n):
            if i != j and get_gender(input_images[j], embeddings_data) == img_gender:
                similarities.append(similarity_matrix[i, j])
        avg_similarities[i] = np.mean(similarities) if similarities else 0

    # 平均類似度でソート
    image_indices = sorted(range(n), key=lambda i: avg_similarities[i], reverse=True)

    # 初期グループの作成
    for i in image_indices:
        if input_images[i] in processed_images:
            continue

        current_gender = get_gender(input_images[i], embeddings_data)
        if current_gender is None:
            continue

        # 新しいグループの作成
        current_group = {
            "images": {input_images[i]},
            "gender": current_gender,
            "core_similarities": [],
        }

        # 類似画像の追加
        for j in range(n):
            if i == j or input_images[j] in processed_images:
                continue

            other_gender = get_gender(input_images[j], embeddings_data)
            if other_gender != current_gender:
                continue

            # グループ内の全画像との平均類似度を計算
            group_similarities = [
                similarity_matrix[j, k]
                for k in range(n)
                if input_images[k] in current_group["images"]
            ]
            avg_sim = np.mean(group_similarities)

            if avg_sim > threshold:
                current_group["images"].add(input_images[j])
                current_group["core_similarities"].append(avg_sim)
                processed_images.add(input_images[j])

        # 十分な大きさのグループのみを保持
        if len(current_group["images"]) >= MIN_GROUP_SIZE:
            groups[group_counter] = current_group
            processed_images.add(input_images[i])
            group_counter += 1

    # グループの統合
    final_groups = merge_similar_groups(
        groups, input_images, similarity_matrix, threshold
    )

    # 未分類の画像の処理
    unprocessed = [img for img in input_images if img not in processed_images]
    for img in unprocessed:
        best_group = find_best_group(
            img,
            final_groups,
            input_images,
            similarity_matrix,
            embeddings_data,
            threshold,
        )
        if best_group:
            final_groups[best_group]["images"].add(img)
        else:
            # 新しいグループを作成
            gender = get_gender(img, embeddings_data)
            final_groups[group_counter] = {
                "images": {img},
                "gender": gender,
                "core_similarities": [],
            }
            group_counter += 1

    # 結果の出力
    for group_id, group_info in final_groups.items():
        group_dir = output_dir / f"person_{group_id}"
        group_dir.mkdir(exist_ok=True)
        for img in group_info["images"]:
            shutil.copy(img, group_dir)
            logging.info(f"Added {img.name} to group {group_id}")

    return final_groups


def find_best_group(
    img, groups, input_images, similarity_matrix, embeddings_data, threshold
):
    """最適なグループを見つける補助関数"""
    img_idx = input_images.index(img)
    img_gender = get_gender(img, embeddings_data)
    best_group = None
    best_similarity = threshold

    for group_id, group_info in groups.items():
        if group_info["gender"] != img_gender:
            continue

        similarities = []
        for group_img in group_info["images"]:
            group_idx = input_images.index(group_img)
            similarities.append(similarity_matrix[img_idx, group_idx])

        if similarities:
            avg_sim = np.mean(similarities)
            if avg_sim > best_similarity:
                best_similarity = avg_sim
                best_group = group_id

    return best_group


def merge_similar_groups(groups, input_images, similarity_matrix, threshold):
    """
    類似度の高いグループを統合する関数
    """
    merged_groups = groups.copy()
    merge_occurred = True

    while merge_occurred:
        merge_occurred = False
        group_ids = list(merged_groups.keys())

        for i in range(len(group_ids)):
            for j in range(i + 1, len(group_ids)):
                group1_id = group_ids[i]
                group2_id = group_ids[j]

                if group1_id not in merged_groups or group2_id not in merged_groups:
                    continue

                group1 = merged_groups[group1_id]
                group2 = merged_groups[group2_id]

                if group1["gender"] != group2["gender"]:
                    continue

                # グループ間の平均類似度を計算
                similarities = []
                for img1 in group1["images"]:
                    for img2 in group2["images"]:
                        idx1 = input_images.index(img1)
                        idx2 = input_images.index(img2)
                        similarities.append(similarity_matrix[idx1, idx2])

                if similarities and np.mean(similarities) > threshold:
                    # グループを統合
                    merged_groups[group1_id]["images"].update(group2["images"])
                    del merged_groups[group2_id]
                    merge_occurred = True
                    logging.info(f"Merged group {group2_id} into group {group1_id}")
                    break

            if merge_occurred:
                break

    return merged_groups


def analyze_similarities(similarity_matrix):
    """
    類似度の統計情報を計算する関数
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
    """
    try:
        # キャッシュファイルの設定
        cache_file = "embeddings_cache.pkl"
        if os.path.exists(cache_file):
            os.remove(cache_file)

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

        # 特徴ベクトルと性別の抽出 (並列処理)
        num_processes = cpu_count()  # 使用可能なCPUコア数を取得
        embeddings_data = process_embeddings_parallel(input_images, num_processes)

        if not embeddings_data:
            logging.error("Failed to extract embeddings from any images")
            return

        # 類似度行列の計算 (並列処理)
        similarity_matrix = calculate_similarity_matrix_parallel(
            input_images, embeddings_data, num_processes
        )

        # 類似度の統計分析
        stats = analyze_similarities(similarity_matrix)
        if stats:
            logging.info("\nSimilarity Statistics:")
            for key, value in stats.items():
                logging.info(f"{key.capitalize()} similarity: {value:.3f}")

        # グループの作成
        create_groups(
            input_images,
            similarity_matrix,
            embeddings_data,
            output_dir,
            SIMILARITY_THRESHOLD,
        )

        # 結果の表示
        logging.info("\nGrouping Results:")
        for group_dir in sorted(output_dir.glob("person_*")):
            images = list(group_dir.glob("*.jpg"))
            if images:
                first_image = images[0]
                gender = get_gender(Path(input_dir / first_image.name), embeddings_data)
                logging.info(
                    f"{group_dir.name}: {len(images)} images (Gender: {gender})"
                )

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
    main()
