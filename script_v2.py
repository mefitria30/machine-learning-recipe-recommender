#!/usr/bin/env python
# coding: utf-8

# In[12]:


import streamlit as st
import pandas as pd
import numpy as np
import random
import os  # Untuk menangani jalur file gambar
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("dataset.csv")

# Rename columns
data.rename(columns={'Unnamed: 0': 'id', 'Title': 'name', 'Ingredients': 'ingredients',
                     'Instructions': 'steps', 'Image_Name': 'image'}, inplace=True)

# Tambahkan kolom rating jika belum ada, lalu simpan agar permanen
if 'rating' not in data.columns:
    data['rating'] = np.random.randint(1, 6, size=len(data))
    data.to_csv("dataset.csv", index=False)  # Simpan ke file agar tetap konsisten

# Pastikan membaca dataset terbaru setelah pembaruan rating
data = pd.read_csv("dataset.csv")

# Pilih kolom yang relevan
new_data = data[['id', 'name', 'ingredients', 'steps', 'image', 'rating']]

# Tetapkan cluster berdasarkan nilai rating yang sekarang permanen
new_data['cluster'] = np.where(new_data['rating'] <= 3, 0, 1)
new_data['cluster_label'] = new_data['cluster'].map({0: 'biasa saja', 1: 'favorit'})

# Gunakan TF-IDF lebih optimal
tfidf = TfidfVectorizer(stop_words='english', max_features=20000, ngram_range=(1, 3))
X_ingredients = tfidf.fit_transform(new_data['ingredients'].values.astype('U'))

# Gunakan K-Means dengan cluster lebih optimal
optimal_clusters = 10
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
new_data['category_num'] = kmeans.fit_predict(X_ingredients)

# Pemetaan kategori lebih rinci berdasarkan ingredients
def categorize_food(ingredients):
    ingredients = ingredients.lower()

    # 🍖 Daging & Protein Hewani
    if any(item in ingredients for item in ['beef', 'lamb', 'pork', 'chicken', 'turkey', 'duck', 'bacon', 'sausage']):
        return "Meat-Based"
    elif any(item in ingredients for item in ['fish', 'shrimp', 'salmon', 'tuna', 'crab', 'mussels']):
        return "Seafood"
    
    # 🥦 Sayuran & Tanaman
    elif any(item in ingredients for item in ['spinach', 'kale', 'lettuce', 'broccoli', 'carrot', 'potato', 'tomato', 'cucumber', 'mushroom']):
        return "Vegetarian"

    # 🍚 Sumber Karbohidrat
    elif any(item in ingredients for item in ['rice', 'oats', 'quinoa', 'barley', 'flour', 'bread', 'pasta', 'tortilla', 'sweet potato']):
        return "Bakery"

    # 🍶 Produk Susu & Alternatifnya
    elif any(item in ingredients for item in ['milk', 'cheese', 'butter', 'yogurt', 'almond milk', 'soy milk']):
        return "Dairy-Based"

    # 🍯 Pemanis & Perasa
    elif any(item in ingredients for item in ['sugar', 'honey', 'maple syrup', 'stevia', 'salt', 'pepper', 'garlic', 'cinnamon', 'basil', 'oregano']):
        return "Seasoning & Sweetener"

    # 🍩 Dessert & Bahan Kue
    elif any(item in ingredients for item in ['chocolate', 'cocoa', 'vanilla', 'eggs', 'baking powder']):
        return "Dessert"
    
    else:
        return "Other"

new_data['category'] = new_data['ingredients'].apply(categorize_food)

# Gunakan Naïve Bayes untuk rekomendasi
cv = TfidfVectorizer(max_features=20000, stop_words='english', ngram_range=(1, 3))
X_name = cv.fit_transform(new_data['name'].values.astype('U'))

model_recommendation = MultinomialNB(alpha=0.5)
y = np.arange(len(new_data))
model_recommendation.fit(X_name, y)

# **Streamlit UI**
st.title("🔍 Sistem Rekomendasi Makanan")

# Tentukan folder gambar
image_folder = "Food Images"

# Pilihan pencarian
search_option = st.radio("Pilih metode pencarian:", ["Cari berdasarkan nama", "Cari berdasarkan kategori", "Cari berdasarkan cluster"])

# Fungsi untuk memberi warna berdasarkan rating
def color_rating(val):
    color = "#FFCCCB" if int(val) <= 3 else "#C6E5B3"
    return f"background-color: {color};"

# Fungsi untuk mendapatkan jalur gambar dengan pengecekan format
def get_image_path(image_name):
    for ext in [".jpg", ".png", ".jpeg"]:  # Coba beberapa kemungkinan format
        image_path = os.path.join(image_folder, image_name + ext)
        if os.path.exists(image_path):
            return image_path
    return None

if search_option == "Cari berdasarkan nama":
    food_name = st.text_input("Masukkan nama makanan:")
    if food_name:
        query_vec = cv.transform([food_name.lower()])
        predicted_probs = model_recommendation.predict_proba(query_vec)[0]
        top_5_indices = np.argsort(predicted_probs)[-5:][::-1]
        recommended_items = new_data.iloc[top_5_indices].dropna().reset_index(drop=True)

        st.subheader(f"🍽️ Rekomendasi untuk: {food_name}")

        for _, row in recommended_items.iterrows():
            image_path = get_image_path(row['image'].split('.')[0])  # Hapus ekstensi jika ada
            if image_path:
                st.image(image_path, caption=row['name'], use_column_width=True)
            else:
                st.write(f"⚠️ Gambar tidak ditemukan: {row['image']}")

        st.dataframe(recommended_items.style.applymap(color_rating, subset=['rating']), width=900)

elif search_option == "Cari berdasarkan kategori":
    category_name = st.selectbox("Pilih kategori:", new_data['category'].unique())
    if category_name:
        matching_items = new_data[new_data['category'] == category_name].sample(5).dropna().reset_index(drop=True)

        st.subheader(f"Makanan dalam kategori '{category_name}':")

        for _, row in matching_items.iterrows():
            image_path = get_image_path(row['image'].split('.')[0])
            if image_path:
                st.image(image_path, caption=row['name'], use_column_width=True)
            else:
                st.write(f"⚠️ Gambar tidak ditemukan: {row['image']}")

        st.dataframe(matching_items.style.applymap(color_rating, subset=['rating']), width=900)

elif search_option == "Cari berdasarkan cluster":
    cluster_label = st.selectbox("Pilih cluster:", new_data['cluster_label'].unique())
    if cluster_label:
        matching_items = new_data[new_data['cluster_label'] == cluster_label].sample(5).dropna().reset_index(drop=True)

        st.subheader(f"Makanan dalam cluster '{cluster_label}':")

        for _, row in matching_items.iterrows():
            image_path = get_image_path(row['image'].split('.')[0])
            if image_path:
                st.image(image_path, caption=row['name'], use_column_width=True)
            else:
                st.write(f"⚠️ Gambar tidak ditemukan: {row['image']}")

        st.dataframe(matching_items.style.applymap(color_rating, subset=['rating']), width=900)


# In[ ]:




