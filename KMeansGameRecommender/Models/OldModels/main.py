from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import pickle

# -------------------------
# Model ve veri yükleme
# -------------------------
with open("../Models/model.pkl", "rb") as f:
    kmeans = pickle.load(f)
with open("../Models/tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)
with open("../Models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

df = pd.read_csv("../Data/games.csv")
cols_to_keep = [
    "Name", "Genres", "Tags", "User score", "Recommendations",
    "Windows", "Mac", "Linux", "Release date"
]
df_selected = df[cols_to_keep].copy()
df_selected.dropna(subset=["Genres"], inplace=True)
df_selected["Tags"] = df_selected["Tags"].fillna("")
df_selected["User score"] = df_selected["User score"].fillna(df_selected["User score"].mean())
df_selected["release_year"] = pd.to_datetime(df_selected["Release date"], errors="coerce").dt.year
df_selected["release_year"].fillna(df_selected["release_year"].median(), inplace=True)
df_selected.drop(columns=["Release date"], inplace=True)

genres_encoded = df_selected["Genres"].str.get_dummies(",")
tags_encoded = tfidf.transform(df_selected["Tags"].astype(str)).toarray()
df_selected["Recommendations"] = np.log1p(df_selected["Recommendations"])
numeric_scaled = scaler.transform(df_selected[["User score", "Recommendations", "release_year"]])
platforms = df_selected[["Windows", "Mac", "Linux"]].astype(int).values
X = np.hstack([genres_encoded.values, tags_encoded, numeric_scaled, platforms])
game_vectors = pd.DataFrame(X, index=df_selected["Name"])

# -------------------------
# Fonksiyonlar
# -------------------------
def get_user_vector(selected_games):
    existing_games = [g for g in selected_games if g in game_vectors.index]
    missing_games = [g for g in selected_games if g not in game_vectors.index]

    if missing_games:
        print(f"Uyarı: Aşağıdaki oyunlar veri setinde bulunamadı: {missing_games}")

    if not existing_games:
        print("Hiçbir seçilen oyun veri setinde yok, öneri üretilemiyor.")
        return None

    vectors = game_vectors.loc[existing_games]
    user_vector = np.mean(vectors.values, axis=0).reshape(1, -1)
    return user_vector

def recommend_games(selected_games, top_n=5):
    user_vector = get_user_vector(selected_games)
    if user_vector is None:
        return pd.DataFrame(columns=["Name", "Genres", "User score", "Recommendations"])

    cluster = kmeans.predict(user_vector)[0]
    cluster_games = df_selected[kmeans.labels_ == cluster]
    cluster_games = cluster_games[~cluster_games["Name"].isin(selected_games)]

    if cluster_games.empty:
        return pd.DataFrame(columns=["Name", "Genres", "User score", "Recommendations"])

    recommendations = cluster_games.sample(n=min(top_n, len(cluster_games)), random_state=42)
    return recommendations[["Name", "Genres", "User score", "Recommendations"]]

# -------------------------
# Flask API
# -------------------------
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "🎮 K-Means Game Recommender API is running!"}), 200

@app.route("/recommend", methods=["POST"])
def recommend():
    data = request.get_json()

    if not data or "selected_games" not in data:
        return jsonify({"error": "Geçersiz istek formatı."}), 400

    selected_games = data["selected_games"]
    if not isinstance(selected_games, list) or len(selected_games) == 0:
        return jsonify({"error": "Oyun listesi boş veya hatalı."}), 400

    recommendations = recommend_games(selected_games)

    return jsonify({
        "input_games": selected_games,
        "recommendations": recommendations.to_dict(orient="records")
    }), 200

# -------------------------
# Sunucu başlatma
# -------------------------
if __name__ == "__main__":
    print("🚀 Flask sunucusu başlatılıyor...")
    app.run(debug=True, host="127.0.0.1", port=5000)
