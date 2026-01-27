
#“Önce içerik benzerliğiyle aday seç, sonra kalite/popülerlikle sırala.”
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import json

app = Flask(__name__)
CORS(app)

# =========================================================
# 1) Model ve dönüştürücüleri yükle
#    (Eğitimde kaydettiğin dosya adları bunlar olmalı)
# =========================================================
with open("../Models/model.pkl", "rb") as f:
    kmeans = pickle.load(f)

with open("../Models/tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("../Models/scaler.pkl", "rb") as f:
    scaler_year = pickle.load(f)

# =========================================================
# 2) JSON veri yükle
# =========================================================
with open("../Data/games.json", "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame({
    "Name": [v.get("name", "") for v in data.values()],
    "Genres": [",".join(v.get("genres", [])) if v.get("genres") else "" for v in data.values()],
    "Tags": [",".join(v["tags"].keys()) if v.get("tags") else "" for v in data.values()],
    "Windows": [int(v.get("windows", 0)) for v in data.values()],
    "Mac": [int(v.get("mac", 0)) for v in data.values()],
    "Linux": [int(v.get("linux", 0)) for v in data.values()],
    "Release date": [v.get("release_date", "2000-01-01") for v in data.values()],
    "Metacritic score": [v.get("metacritic_score", np.nan) for v in data.values()],
    "Recommendations": [v.get("recommendations", np.nan) for v in data.values()],
})

# İsim temizliği
df["Name"] = df["Name"].astype(str).str.strip()
df = df[df["Name"] != ""].copy()

# =========================================================
# 3) Eksik verileri temizle (Metacritic + Recommendations)
# =========================================================
df["Recommendations"] = pd.to_numeric(df["Recommendations"], errors="coerce").fillna(0)

df["Metacritic score"] = pd.to_numeric(df["Metacritic score"], errors="coerce")
df["Metacritic score"] = df["Metacritic score"].fillna(df["Metacritic score"].median())

df["Tags"] = df["Tags"].fillna("")
df["Genres"] = df["Genres"].fillna("")

df["release_year"] = pd.to_datetime(df["Release date"], errors="coerce").dt.year
df["release_year"] = df["release_year"].fillna(df["release_year"].median())
df.drop(columns=["Release date"], inplace=True)

# =========================================================
# 4) K-Means içerik vektörü X_content oluştur
#    Not: K-Means'i bununla eğitmiş olman gerekiyor (content space)
# =========================================================
genres_encoded = df["Genres"].astype(str).str.get_dummies(",")

tags_encoded = tfidf.transform(df["Tags"].astype(str)).toarray()

year_scaled = scaler_year.transform(df[["release_year"]])

platforms = df[["Windows", "Mac", "Linux"]].astype(int).values

X_content = np.hstack([genres_encoded.values, tags_encoded, year_scaled, platforms])
X_content = np.nan_to_num(X_content)

# Cluster etiketlerini bir kere üret (labels mismatch çözümü)
LABELS = kmeans.predict(X_content)

# User vector için
game_vectors = pd.DataFrame(X_content, index=df["Name"])

# Search cache
ALL_NAMES = df["Name"].dropna().astype(str).unique().tolist()
ALL_NAMES_LOWER = [n.lower() for n in ALL_NAMES]
name_map = {n.lower(): n for n in ALL_NAMES}

print("✅ Loaded:", len(df), "games | LABELS:", len(LABELS))

# =========================================================
# 5) Re-ranking skor fonksiyonu (normalize + ağırlık)
# =========================================================
def add_rerank_score(candidates: pd.DataFrame) -> pd.DataFrame:
    c = candidates.copy()

    rec = c["Recommendations"].astype(float)
    meta = c["Metacritic score"].astype(float)

    rec_norm = (rec - rec.min()) / (rec.max() - rec.min() + 1e-9)
    meta_norm = (meta - meta.min()) / (meta.max() - meta.min() + 1e-9)

    # Ağırlıklar (hocanın dediği gibi kalite+popülerlik)
    c["score"] = 0.6 * rec_norm + 0.4 * meta_norm
    return c

# =========================================================
# 6) Kullanıcı vektörü
# =========================================================
def get_user_vector(selected_games):
    vectors = []
    matched = []

    for g in selected_games:
        key = str(g).strip().lower()
        real = name_map.get(key)
        if real:
            vectors.append(game_vectors.loc[real].values)
            matched.append(real)

    if not vectors:
        return None, []

    user_vec = np.mean(vectors, axis=0).reshape(1, -1)
    return user_vec, matched

# =========================================================
# 7) 3 mod öneri fonksiyonları
# =========================================================
def recommend_kmeans(selected_games, top_n=5, candidate_n=50):
    user_vec, matched = get_user_vector(selected_games)
    if user_vec is None:
        return pd.DataFrame(), matched

    cluster = kmeans.predict(user_vec)[0]

    # Aday belirleme (küme içinden)
    cluster_games = df[LABELS == cluster].copy()
    cluster_games = cluster_games[~cluster_games["Name"].isin(matched)]

    if len(cluster_games) == 0:
        return pd.DataFrame(), matched

    # En fazla 50 aday (deterministic)
    candidates = cluster_games.sample(n=min(candidate_n, len(cluster_games)), random_state=42)

    # Yeniden sıralama (Metacritic + Recommendations)
    candidates = add_rerank_score(candidates)

    top = candidates.sort_values("score", ascending=False).head(top_n)
    return top[["Name", "Genres", "Metacritic score", "Recommendations", "score"]], matched


def recommend_popularity(selected_games, top_n=5):
    selected_lower = {str(x).strip().lower() for x in selected_games}

    candidates = df.copy()
    candidates = candidates[~candidates["Name"].str.lower().isin(selected_lower)]

    candidates = add_rerank_score(candidates)
    top = candidates.sort_values("score", ascending=False).head(top_n)

    return top[["Name", "Genres", "Metacritic score", "Recommendations", "score"]]


def recommend_random(selected_games, top_n=5):
    selected_lower = {str(x).strip().lower() for x in selected_games}

    candidates = df.copy()
    candidates = candidates[~candidates["Name"].str.lower().isin(selected_lower)]

    if len(candidates) == 0:
        return pd.DataFrame()

    top = candidates.sample(n=min(top_n, len(candidates)))
    top["score"] = np.nan
    return top[["Name", "Genres", "Metacritic score", "Recommendations", "score"]]

# =========================================================
# 8) ROUTES
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

    # prefix
    for name, low in zip(ALL_NAMES, ALL_NAMES_LOWER):
        if low.startswith(q):
            results.append(name)
            if len(results) >= limit:
                break

    # contains
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

    if mode == "kmeans":
        recs, matched = recommend_kmeans(selected, top_n=5, candidate_n=50)
        if recs.empty:
            return jsonify({"message": "Uygun öneri bulunamadı.", "matched_games": matched, "mode": mode}), 404
        return jsonify({
            "mode": mode,
            "matched_games": matched,
            "recommendations": recs.to_dict(orient="records")
        })

    if mode == "popular":
        recs = recommend_popularity(selected, top_n=5)
        if recs.empty:
            return jsonify({"message": "Uygun öneri bulunamadı.", "mode": mode}), 404
        return jsonify({
            "mode": mode,
            "recommendations": recs.to_dict(orient="records")
        })

    if mode == "random":
        recs = recommend_random(selected, top_n=5)
        if recs.empty:
            return jsonify({"message": "Uygun öneri bulunamadı.", "mode": mode}), 404
        return jsonify({
            "mode": mode,
            "recommendations": recs.to_dict(orient="records")
        })

    return jsonify({"error": "Geçersiz mode. 'kmeans' | 'popular' | 'random' olmalı."}), 400


if __name__ == "__main__":
    app.run(debug=True, port=5000)
