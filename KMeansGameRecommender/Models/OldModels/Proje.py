import pandas as pd
import numpy as np
import json
import pickle
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer


# =========================================================
# 1️⃣ VERİ YÜKLEME (JSON)
# =========================================================
with open("../Data/games.json", "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame({
    "Name": [v["name"] for v in data.values()],
    "Genres": [",".join(v["genres"]) if v["genres"] else "" for v in data.values()],
    "Tags": [",".join(v["tags"].keys()) if v.get("tags") else "" for v in data.values()],
    "Windows": [v.get("windows", 0) for v in data.values()],
    "Mac": [v.get("mac", 0) for v in data.values()],
    "Linux": [v.get("linux", 0) for v in data.values()],
    "Release date": [v.get("release_date", "2000-01-01") for v in data.values()],
    "Metacritic score": [v.get("metacritic_score", np.nan) for v in data.values()],
    "Recommendations": [v.get("recommendations", np.nan) for v in data.values()]
})

print("Veri yüklendi:", df.shape)


# =========================================================
# 2️⃣ EKSİK VERİ TEMİZLEME
# =========================================================

# Recommendations → 0
df["Recommendations"] = df["Recommendations"].fillna(0)

# Metacritic → median
df["Metacritic score"] = df["Metacritic score"].fillna(
    df["Metacritic score"].median()
)

# Release year çıkar
df["release_year"] = pd.to_datetime(
    df["Release date"], errors="coerce"
).dt.year

df["release_year"] = df["release_year"].fillna(df["release_year"].median())

df.drop(columns=["Release date"], inplace=True)


# =========================================================
# 3️⃣ FEATURE ENGINEERING
# =========================================================

# Genres → One-hot
genres_encoded = df["Genres"].str.get_dummies(",")

# Tags → TF-IDF
tfidf = TfidfVectorizer(max_features=300)
tags_encoded = tfidf.fit_transform(df["Tags"].astype(str)).toarray()

# Platform
platforms = df[["Windows", "Mac", "Linux"]].astype(int).values

# Year normalize
scaler_year = MinMaxScaler()
year_scaled = scaler_year.fit_transform(df[["release_year"]])

# Quality + Popularity normalize
scaler_qp = MinMaxScaler()
quality_popularity = scaler_qp.fit_transform(
    df[["Metacritic score", "Recommendations"]]
)

# Final feature matrix
X = np.hstack([
    genres_encoded.values,
    tags_encoded,
    year_scaled,
    platforms,
    quality_popularity
])

X = np.nan_to_num(X)

print("Toplam oyun:", X.shape[0])
print("Feature boyutu:", X.shape[1])


# =========================================================
# 4️⃣ ELBOW YÖNTEMİ
# =========================================================
inertias = []
K_values = range(10, 60, 10)

for k in K_values:
    print(f"K={k} deneniyor...")
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

plt.plot(K_values, inertias, marker="o")
plt.xlabel("K (Küme Sayısı)")
plt.ylabel("Inertia")
plt.title("Elbow Yöntemi")
plt.show()


# =========================================================
# 5️⃣ NİHAİ MODEL
# =========================================================
K_FINAL = 30  # Grafiğe göre ayarlanabilir

kmeans = KMeans(n_clusters=K_FINAL, random_state=42, n_init=10)
kmeans.fit(X)

df["cluster"] = kmeans.labels_

print("Model eğitildi.")


# =========================================================
# 6️⃣ MODEL KAYDETME
# =========================================================
with open("../Models/kmeans.pkl", "wb") as f:
    pickle.dump(kmeans, f)

with open("../Models/tfidf.pkl", "wb") as f:
    pickle.dump(tfidf, f)

with open("../Models/scaler_year.pkl", "wb") as f:
    pickle.dump(scaler_year, f)

with open("../Models/scaler_qp.pkl", "wb") as f:
    pickle.dump(scaler_qp, f)

print("Modeller kaydedildi.")


# =========================================================
# 7️⃣ KULLANICI VEKTÖRÜ
# =========================================================
game_vectors = pd.DataFrame(X, index=df["Name"])

def get_user_vector(selected_games):
    vectors = []

    for game in selected_games:
        matches = [name for name in game_vectors.index if name.lower() == game.lower()]
        if matches:
            vectors.append(game_vectors.loc[matches[0]].values)
        else:
            print(f"Uyarı: '{game}' bulunamadı.")

    if not vectors:
        return None

    return np.mean(vectors, axis=0).reshape(1, -1)


# =========================================================
# 8️⃣ İKİ AŞAMALI ÖNERİ SİSTEMİ
# =========================================================
def recommend_games(selected_games, top_n=5):

    user_vector = get_user_vector(selected_games)
    if user_vector is None:
        return []

    # Aşama 1: K-Means ile aday küme
    cluster = kmeans.predict(user_vector)[0]
    candidates = df[df["cluster"] == cluster]

    candidates = candidates[~candidates["Name"].isin(selected_games)]

    # En fazla 50 aday
    candidates = candidates.sample(n=min(50, len(candidates)), random_state=42)

    # Aşama 2: Yeniden sıralama (kalite + popülerlik)
    candidates["final_score"] = (
        candidates["Metacritic score"] * 0.6 +
        candidates["Recommendations"] * 0.4
    )

    top5 = candidates.sort_values("final_score", ascending=False).head(top_n)

    return top5[["Name", "Genres", "Metacritic score", "Recommendations"]]


# =========================================================
# 9️⃣ TEST
# =========================================================
if __name__ == "__main__":
    user_games = ["Counter-Strike: Global Offensive", "DOOM Eternal", "PAYDAY 2"]

    recs = recommend_games(user_games)

    print("\n--- ÖNERİLER ---")
    print(recs)
