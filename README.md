
#  K-Means Kümeleme Tabanlı İçerik Odaklı Oyun Öneri Sistemi
<p align="center">
  <i>Bilgisayar Mühendisliği Bitirme Projesi</i>
</p>

---

##  Projenin Amacı

Bu projede, **Steam oyun veri seti** kullanılarak **içerik tabanlı** bir oyun öneri sistemi geliştirilmiştir.  
Sistem, kullanıcı–kullanıcı etkileşimlerine dayalı klasik işbirlikçi filtreleme yöntemleri yerine, **oyunların içerik özelliklerini** temel alır.

Amaç:
- Büyük ölçekli (100.000+ oyun) veri setleriyle çalışmaktadır,
- Açıklanabilir ve yorumlanabilir,
- Denetimsiz öğrenme temelli bir öneri sistemi geliştirmektir.

---

##  Temel Yaklaşım

Bu çalışmada:
- Oyunlar **K-Means kümeleme algoritması** ile benzer içeriklerine göre gruplandırılır.
- Kullanıcıdan **3 oyun seçmesi** istenir.
- Seçilen oyunların vektörlerinin ortalaması alınarak bir **kullanıcı vektörü** oluşturulur.
- Kullanıcı vektörü, en yakın kümeye atanır.
- Öneriler yalnızca bu kümeden seçilir.
- Son olarak kalite ve popülerlik bilgileriyle yeniden sıralama yapılır.
- Cosinus ben

---
## Kullanılan Teknolojiler

Projeyi gerçekleyebiliceğimiz en uygun teknolojiler olarak bu çalışmada kullanılan kütüphaneler:
- Python
- Pandas, NumPy
- Scikit-learn
- Flask
- HTML, JavaScript


##  Özellik Mühendisliği (Feature Engineering)

Her oyun aşağıdaki özellikler kullanılarak sayısallaştırılmıştır:

| Özellik | Kullanılan Yöntem |
|-------|------------------|
| Türler (Genres) | One-hot encoding |
| Etiketler (Tags) | TF-IDF |
| Çıkış Yılı | Min-Max normalizasyon |
| Platform | İkili (Windows / Mac / Linux) |
| Kalite (Metacritic) | Min-Max normalizasyon |
| Popülerlik (Recommendations) | Min-Max normalizasyon |

Tüm bu özellikler birleştirilerek tek bir vektör oluşturulur:

```python
X = np.hstack([
    genres_encoded,
    tags_encoded,
    year_scaled,
    platforms,
    qp_scaled
])
```
---
## Klasör Yapısı
```

├── Data/
│   └── games.json
│
├── Models/
│   ├── kmeans_model.pkl
│   ├── kmeans_game_vectors.pkl
│   ├── kmeans_df_meta.pkl
│   └── ...
│
├── Backend/
│   ├── KMeans.py   # Model eğitimi ve özellik mühendisliği
│   └── Main.py     # Flask API
│
├── Frontend/
│   ├── index.html
│   └── app.js
│
└── README.md
```

