from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np

app = Flask(__name__)
CORS(app)

RANDOM_STATE = 42


# =========================================================
# 1) ARTIFACT YÜKLE (SADECE LOAD) - KMEANS
# =========================================================
kmeans = pd.read_pickle("../Models/kmeans_model.pkl")
game_vectors = pd.read_pickle("../Models/kmeans_game_vectors.pkl")  # index: Name
df_meta = pd.read_pickle("../Models/kmeans_df_meta.pkl")            # Name, Genres, scores, cluster

# (Bu dosyalar API'de direkt kullanılmasa da projede tutarlılık için yüklenebilir)
_ = pd.read_pickle("../Models/kmeans_tfidf.pkl")
_ = pd.read_pickle("../Models/kmeans_scaler_year.pkl")
_ = pd.read_pickle("../Models/kmeans_scaler_qp.pkl")
_ = pd.read_pickle("../Models/kmeans_genre_cols.pkl")

# Arama için
ALL_NAMES = df_meta["Name"].astype(str).tolist()
ALL_NAMES_LOWER = [n.lower() for n in ALL_NAMES]
NAME_MAP = {n.lower(): n for n in ALL_NAMES}

print("✅ API hazır | oyun sayısı:", len(df_meta))


# =========================================================
# 2) YARDIMCI: Min-Max normalize + rerank
# =========================================================
def rerank_candidates(candidates: pd.DataFrame, w_rec=0.6, w_meta=0.4) -> pd.DataFrame:
    """
    Popülerlik/kaliteye göre basit bir yeniden sıralama:
    final_score = w_rec * norm(recommendations) + w_meta * norm(metacritic)
    """
    c = candidates.copy()

    rec = pd.to_numeric(c["Recommendations"], errors="coerce").fillna(0).astype(float)
    meta = pd.to_numeric(c["Metacritic score"], errors="coerce").fillna(0).astype(float)

    rec_norm = (rec - rec.min()) / (rec.max() - rec.min() + 1e-9)
    meta_norm = (meta - meta.min()) / (meta.max() - meta.min() + 1e-9)

    c["final_score"] = w_rec * rec_norm + w_meta * meta_norm
    return c


# =========================================================
# 3) YARDIMCI: Kullanıcı vektörü + seçilen oyunların eşleşmesi
# =========================================================
def get_user_vector(selected_games, vectors_df: pd.DataFrame):
    """
    Kullanıcı 3 oyun seçiyor.
    Bu oyunların vektörleri alınır, ortalaması user_vec olur.
    """
    vectors = []
    matched = []

    # vectors_df index'i game name, hızlı case-insensitive map
    local_map = {n.lower(): n for n in vectors_df.index.astype(str).tolist()}

    for g in selected_games:
        key = str(g).strip().lower()
        real = local_map.get(key)
        if real is None:
            continue

        v = vectors_df.loc[real].values
        vectors.append(v)
        matched.append(real)

    if not vectors:
        return None, []

    user_vec = np.mean(np.stack(vectors, axis=0), axis=0).reshape(1, -1)
    return user_vec, matched


def get_selected_games_details(matched_games):
    """
    UI'da "Selected by user games" alanında göstermek için:
    Seçilen oyunların Genres / Metacritic / Recommendations bilgilerini döndürür.
    """
    if not matched_games:
        return []

    rows = df_meta[df_meta["Name"].isin(matched_games)].copy()
    # sırayı kullanıcı seçimine göre koruyalım
    order = {name: i for i, name in enumerate(matched_games)}
    rows["__ord"] = rows["Name"].map(order)
    rows = rows.sort_values("__ord").drop(columns=["__ord"])

    out = []
    for _, r in rows.iterrows():
        out.append({
            "Name": str(r.get("Name", "")),
            "Genres": str(r.get("Genres", "")),
            "Metacritic score": float(pd.to_numeric(r.get("Metacritic score", 0), errors="coerce") or 0),
            "Recommendations": float(pd.to_numeric(r.get("Recommendations", 0), errors="coerce") or 0),
        })
    return out


# =========================================================
# 4) BENZERLİK: Cosine similarity (1 iyi, 0 kötü)
# =========================================================
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    cosine(a,b) = (a·b) / (||a|| * ||b||)
    Sonuç: [-1, 1] ama bizde özellikler pozitif ağırlıklı olduğu için genelde [0,1].
    1'e yakın => daha benzer.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


# =========================================================
# 5) ÖNERİ FONKSİYONLARI (3 MOD)
# =========================================================
def recommend_kmeans(selected_games, top_n=5, candidate_n=50):
    user_vec, matched = get_user_vector(selected_games, game_vectors)
    if user_vec is None:
        return pd.DataFrame(), matched

    cluster = int(kmeans.predict(user_vec)[0])

    candidates = df_meta[df_meta["cluster"] == cluster].copy()
    candidates = candidates[~candidates["Name"].isin(matched)]

    if candidates.empty:
        return pd.DataFrame(), matched

    candidates = candidates.sample(n=min(candidate_n, len(candidates)), random_state=RANDOM_STATE)
    candidates = rerank_candidates(candidates, w_rec=0.6, w_meta=0.4)

    top = candidates.sort_values("final_score", ascending=False).head(top_n)
    return top[["Name", "Genres", "Metacritic score", "Recommendations", "final_score"]], matched


def recommend_kmeans_cosine(selected_games, top_n=5, candidate_n=200):
    """
    K-Means ile cluster seçiyoruz (hız + bağlam).
    Sonra o cluster içindeki adayları cosine similarity ile user_vec'e göre sıralıyoruz.
    İstersen meta/rec ile hafif rerank katkısı da ekliyoruz.
    """
    user_vec, matched = get_user_vector(selected_games, game_vectors)
    if user_vec is None:
        return pd.DataFrame(), matched

    cluster = int(kmeans.predict(user_vec)[0])

    candidates = df_meta[df_meta["cluster"] == cluster].copy()
    candidates = candidates[~candidates["Name"].isin(matched)]
    if candidates.empty:
        return pd.DataFrame(), matched

    # Çok büyük cluster olursa hesap ağırlaşmasın diye örnekle
    candidates = candidates.sample(n=min(candidate_n, len(candidates)), random_state=RANDOM_STATE)

    # Cosine similarity hesapla
    sims = []
    user_v = user_vec.reshape(-1)
    for nm in candidates["Name"].tolist():
        v = game_vectors.loc[nm].values
        sims.append(cosine_similarity(user_v, v))

    candidates = candidates.copy()
    candidates["cosine_similarity"] = sims

    # İstersen sadece cosine ile sırala:
    # top = candidates.sort_values("cosine_similarity", ascending=False).head(top_n)

    # Daha iyi pratik sonuç için: cosine ana sinyal + kalite/popülerlik küçük katkı
    rec = pd.to_numeric(candidates["Recommendations"], errors="coerce").fillna(0).astype(float)
    meta = pd.to_numeric(candidates["Metacritic score"], errors="coerce").fillna(0).astype(float)

    rec_norm = (rec - rec.min()) / (rec.max() - rec.min() + 1e-9)
    meta_norm = (meta - meta.min()) / (meta.max() - meta.min() + 1e-9)

    candidates["final_score"] = (
        0.80 * candidates["cosine_similarity"] +
        0.10 * meta_norm +
        0.10 * rec_norm
    )

    top = candidates.sort_values("final_score", ascending=False).head(top_n)
    return top[["Name", "Genres", "Metacritic score", "Recommendations", "cosine_similarity", "final_score"]], matched


def recommend_popular(selected_games, top_n=5):
    selected_lower = {str(x).strip().lower() for x in selected_games}
    candidates = df_meta[~df_meta["Name"].str.lower().isin(selected_lower)].copy()
    candidates = rerank_candidates(candidates, w_rec=0.6, w_meta=0.4)
    top = candidates.sort_values("final_score", ascending=False).head(top_n)
    return top[["Name", "Genres", "Metacritic score", "Recommendations", "final_score"]]


# =========================================================
# 6) ROUTES
# =========================================================
@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "🎮 Flask Game Recommender API is running!"})


@app.route("/search", methods=["GET"])
def search_games():
    q = (request.args.get("q") or "").strip().lower()
    if len(q) < 3:
        return jsonify({"results": []})

    limit = 15
    results = []

    # prefix match
    for name, low in zip(ALL_NAMES, ALL_NAMES_LOWER):
        if low.startswith(q):
            results.append(name)
            if len(results) >= limit:
                break

    # contains match
    if len(results) < limit:
        for name, low in zip(ALL_NAMES, ALL_NAMES_LOWER):
            if (q in low) and (name not in results):
                results.append(name)
                if len(results) >= limit:
                    break

    return jsonify({"results": results})


@app.route("/recommend", methods=["POST"])
def recommend():
    payload = request.get_json() or {}
    selected = payload.get("games", [])
    mode = (payload.get("mode") or "kmeans").lower()

    if not selected:
        return jsonify({"error": "Oyun listesi boş."}), 400

    # ---- KMEANS ----
    if mode == "kmeans":
        recs, matched = recommend_kmeans(selected, top_n=5, candidate_n=50)
        selected_details = get_selected_games_details(matched)

        if recs.empty:
            return jsonify({
                "mode": mode,
                "matched_games": matched,
                "selected_by_user_games": matched,
                "selected_by_user_games_details": selected_details,
                "message": "Uygun öneri bulunamadı."
            }), 404

        return jsonify({
            "mode": mode,
            "matched_games": matched,  # eski UI uyumluluğu
            "selected_by_user_games": matched,
            "selected_by_user_games_details": selected_details,
            "recommendations": recs.to_dict(orient="records")
        })

    # ---- KMEANS + COSINE ----
    if mode in ["kmeans_cosine", "cosine"]:
        recs, matched = recommend_kmeans_cosine(selected, top_n=5, candidate_n=200)
        selected_details = get_selected_games_details(matched)

        if recs.empty:
            return jsonify({
                "mode": "kmeans_cosine",
                "matched_games": matched,
                "selected_by_user_games": matched,
                "selected_by_user_games_details": selected_details,
                "message": "Uygun öneri bulunamadı."
            }), 404

        return jsonify({
            "mode": "kmeans_cosine",
            "matched_games": matched,
            "selected_by_user_games": matched,
            "selected_by_user_games_details": selected_details,
            "recommendations": recs.to_dict(orient="records")
        })

    # ---- POPULAR ----
    if mode == "popular":
        recs = recommend_popular(selected, top_n=5)

        # popular modunda da seçilenleri detaylı gösterelim
        # selected zaten UI’dan geliyor; ama isim düzeltmesi için matched gibi map etmek iyi olur
        _, matched = get_user_vector(selected, game_vectors)
        selected_details = get_selected_games_details(matched)

        return jsonify({
            "mode": mode,
            "matched_games": matched,
            "selected_by_user_games": matched,
            "selected_by_user_games_details": selected_details,
            "recommendations": recs.to_dict(orient="records")
        })

    return jsonify({"error": "Geçersiz mode: kmeans | kmeans_cosine | popular"}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
