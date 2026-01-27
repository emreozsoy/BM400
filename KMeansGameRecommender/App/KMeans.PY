import pandas as pd
import numpy as np
import json
import pickle
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score


# =========================================================
# 0️⃣ AYARLAR
# =========================================================
TFIDF_MAX_FEATURES = 300
RANDOM_STATE = 42

# K seçimi için aralık
K_MIN = 10
K_MAX = 60
K_STEP = 10
best_k = 20

# Silhouette hesaplaması büyük veride pahalı olabilir → O(n2) 
SAMPLE_FOR_SILHOUETTE = 5000 #ortalama



# =========================================================
# 1️⃣ VERİ YÜKLEME 
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

# İsim temizliği + tekilleştirme (API'de duplicate name hatası olmasın diye ekledik)
df["Name"] = df["Name"].astype(str).str.strip()
df = df[df["Name"] != ""].copy()
df = df.drop_duplicates(subset=["Name"], keep="first").copy()

print("Veri yüklendi:", df.shape)


# =========================================================
# 2️⃣ EKSİK VERİ TEMİZLEME
# =========================================================
df["Genres"] = df["Genres"].fillna("")
df["Tags"] = df["Tags"].fillna("")

# Recommendations → 0 (popüler değilse 0 kabul)
df["Recommendations"] = pd.to_numeric(df["Recommendations"], errors="coerce").fillna(0)

# Metacritic → median (0 yapma, yanıltır)
df["Metacritic score"] = pd.to_numeric(df["Metacritic score"], errors="coerce")
df["Metacritic score"] = df["Metacritic score"].fillna(df["Metacritic score"].median())

# Release year çıkar
df["release_year"] = pd.to_datetime(df["Release date"], errors="coerce").dt.year
df["release_year"] = df["release_year"].fillna(df["release_year"].median())
df.drop(columns=["Release date"], inplace=True)


# =========================================================
# 3️⃣ FEATURE ENGINEERING (EĞİTİM)
# =========================================================
# Genres → One-hot
genres_encoded = df["Genres"].astype(str).str.get_dummies(",")
genre_cols = genres_encoded.columns.tolist()

# Tags → TF-IDF
tfidf = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)
tags_encoded = tfidf.fit_transform(df["Tags"].astype(str)).toarray()

# Platform
platforms = df[["Windows", "Mac", "Linux"]].astype(int).values

# Year normalize
scaler_year = MinMaxScaler()
year_scaled = scaler_year.fit_transform(df[["release_year"]])

# Quality + Popularity normalize
scaler_qp = MinMaxScaler()
qp_scaled = scaler_qp.fit_transform(df[["Metacritic score", "Recommendations"]])

# Final feature matrix
X = np.hstack([
    genres_encoded.values,
    tags_encoded,
    year_scaled,
    platforms,
    qp_scaled
])

# NaN/inf varsa temizliyoruz
X = np.nan_to_num(X)

print("Toplam oyun:", X.shape[0])
print("Feature boyutu:", X.shape[1])


# =========================================================
# 4️⃣ K SEÇİMİ: ELBOW + SILHOUETTE
# =========================================================
K_values = list(range(K_MIN, K_MAX + 1, K_STEP))

inertias = []
# sil_scores = []

# # silhouette için örneklem indeksleri (rastgele ama sabit)
# if X.shape[0] > SAMPLE_FOR_SILHOUETTE:
#     rng = np.random.RandomState(RANDOM_STATE)
#     sample_idx = rng.choice(X.shape[0], size=SAMPLE_FOR_SILHOUETTE, replace=False)
#     X_sil = X[sample_idx]
# else:
#     X_sil = X

print(f"Silhouette hesaplaması için örnek sayısı: {X_sil.shape[0]}")

for k in K_values:
    print(f"K={k} deneniyor...")

    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
    labels = km.fit_predict(X)

    # Elbow: inertia (küme içi toplam hata)
    inertias.append(km.inertia_)

    # # Silhouette: daha yüksek daha iyi (k=1 olamaz; burada zaten k>=10)
    # # Silhouette hesaplamasını örneklem üzerinden yapıyoruz
    # sil = silhouette_score(X_sil, km.predict(X_sil), metric="euclidean")
    # sil_scores.append(sil)

# # En iyi K: silhouette maksimum olan
# best_k = K_values[int(np.argmax(sil_scores))]
# print("\n✅ Silhouette'a göre en iyi K =", best_k)

# --- Grafikler ---
plt.figure()
plt.plot(K_values, inertias, marker="o")
plt.xlabel("K (Küme Sayısı)")
plt.ylabel("Inertia")
plt.title("Elbow Yöntemi")
plt.show()

# plt.figure()
# plt.plot(K_values, sil_scores, marker="o")
# plt.xlabel("K (Küme Sayısı)")
# plt.ylabel("Silhouette Score")
# plt.title("Silhouette Skoru (Yüksek daha iyi)")
# plt.show()


# =========================================================
# 5️⃣ NİHAİ MODEL EĞİTİMİ (BEST K)
# =========================================================
kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=20)
kmeans.fit(X)

print("✅ Model eğitildi. K =", best_k)

# Eğitim seti cluster label’ları (API’de tekrar hesaplamayacağız, kaydedeceğiz)
df["cluster"] = kmeans.labels_


# =========================================================
# 6️⃣ ARTIFACT KAYDETME 
# =========================================================
with open("../Models/kmeans_model.pkl", "wb") as f:
    pickle.dump(kmeans, f)

with open("../Models/kmeans_tfidf.pkl", "wb") as f:
    pickle.dump(tfidf, f)

with open("../Models/kmeans_scaler_year.pkl", "wb") as f:
    pickle.dump(scaler_year, f)

with open("../Models/kmeans_scaler_qp.pkl", "wb") as f:
    pickle.dump(scaler_qp, f)

with open("../Models/kmeans_genre_cols.pkl", "wb") as f:
    pickle.dump(genre_cols, f)

# oyun vektörleri
game_vectors = pd.DataFrame(X, index=df["Name"])
game_vectors.to_pickle("../Models/kmeans_game_vectors.pkl")

# metadata
df_meta = df[["Name", "Genres", "Metacritic score", "Recommendations", "cluster"]].copy()
df_meta.to_pickle("../Models/kmeans_df_meta.pkl")

# Ek: seçilen K ve skorları kaydetmek için 
k_info = {
    "K_values": K_values,
    "inertias": inertias,
    "silhouette_scores": sil_scores,
    "best_k": best_k,
    "silhouette_sample_size": int(X_sil.shape[0])
}
with open("../Models/kmeans_k_selection_info.pkl", "wb") as f:
    pickle.dump(k_info, f)

print("✅ KMeans Artifacts kaydedildi:")
print("- ../Models/kmeans_model.pkl")
print("- ../Models/kmeans_tfidf.pkl")
print("- ../Models/kmeans_scaler_year.pkl")
print("- ../Models/kmeans_scaler_qp.pkl")
print("- ../Models/kmeans_genre_cols.pkl")
print("- ../Models/kmeans_game_vectors.pkl")
print("- ../Models/kmeans_df_meta.pkl")
#print("- ../Models/kmeans_k_selection_info.pkl (K seçim logu)")


# =========================================================
# 7️⃣ TERMINAL TEST 
# =========================================================
def recommend_games(selected_games, top_n=5, candidate_n=50):
    selected_lower = {s.strip().lower() for s in selected_games}

    matched = []
    vectors = []

    # case-insensitive eşleşme (küçük iyileştirme)
    name_map = {name.lower(): name for name in game_vectors.index}

    for g in selected_games:
        key = str(g).strip().lower()
        real = name_map.get(key)
        if real:
            matched.append(real)
            vectors.append(game_vectors.loc[real].values)

    if not vectors:
        return pd.DataFrame()

    user_vec = np.mean(np.stack(vectors, axis=0), axis=0).reshape(1, -1)
    cluster = int(kmeans.predict(user_vec)[0])

    candidates = df_meta[df_meta["cluster"] == cluster].copy()
    candidates = candidates[~candidates["Name"].str.lower().isin(selected_lower)]

    if len(candidates) == 0:
        return pd.DataFrame()

    candidates = candidates.sample(n=min(candidate_n, len(candidates)), random_state=RANDOM_STATE)

    # Rerank: rec + meta normalize
    rec = candidates["Recommendations"].astype(float)
    meta = candidates["Metacritic score"].astype(float)
    rec_norm = (rec - rec.min()) / (rec.max() - rec.min() + 1e-9)
    meta_norm = (meta - meta.min()) / (meta.max() - meta.min() + 1e-9)

    candidates["final_score"] = 0.6 * rec_norm + 0.4 * meta_norm
    return candidates.sort_values("final_score", ascending=False).head(top_n)


if __name__ == "__main__":
    user_games = ["Counter-Strike: Global Offensive", "DOOM 3", "Dash Blitz"]
    recs = recommend_games(user_games)
    print("\n--- ÖNERİLER (TRAIN TEST) ---")
    if recs.empty:
        print("Öneri bulunamadı.")
    else:
        print(recs[["Name", "Genres", "Metacritic score", "Recommendations", "final_score"]])

