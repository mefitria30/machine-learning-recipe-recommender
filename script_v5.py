#!/usr/bin/env python
# coding: utf-8

# In[23]:


import streamlit as st
import pandas as pd
import numpy as np
import random
import os  # Untuk menangani jalur file gambar
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# In[24]:


# Load dataset
data = pd.read_csv("dataset.csv")


# In[25]:


# Tambahkan kolom rating jika belum ada, lalu simpan agar permanen
if 'rating' not in data.columns:
    data['rating'] = np.random.randint(1, 6, size=len(data))
    data.to_csv("dataset.csv", index=False)  # Simpan ke file agar tetap konsisten


# In[26]:


# Pastikan membaca dataset terbaru setelah pembaruan rating
data = pd.read_csv("dataset.csv")


# In[27]:


data.isnull().sum()


# In[28]:


data.dropna(inplace=True)


# In[29]:


data.isnull().sum()


# In[30]:


data.duplicated().sum()


# In[31]:


data.drop_duplicates(inplace=True)


# In[32]:


data['rating'] = pd.to_numeric(data['rating'], errors='coerce')


# In[33]:


data = data[(data['rating'] >= 1) & (data['rating'] <= 5)]


# In[34]:


# Ubah format ingredients dari list menjadi string dengan koma
data['ingredients'] = data['ingredients'].apply(lambda x: ', '.join(eval(x)) if isinstance(x, str) else x)


# In[35]:


data.drop(columns=['Cleaned_Ingredients'], inplace=True)


# In[36]:


# Rename columns
data.rename(columns={'Unnamed: 0': 'id', 'Title': 'name', 'Ingredients': 'ingredients',
                     'Instructions': 'steps', 'Image_Name': 'image'}, inplace=True)


# In[37]:


# Pilih kolom yang relevan
new_data = data[['id', 'name', 'ingredients', 'steps', 'image', 'rating']]


# In[38]:


# Tetapkan cluster berdasarkan nilai rating dengan kategori baru
new_data['cluster'] = np.where(new_data['rating'] == 1, 1, 
                        np.where(new_data['rating'] == 2, 2, 
                        np.where(new_data['rating'] == 3, 3, 
                        np.where(new_data['rating'] == 4, 4, 5)))) 

# Tetapkan label cluster sesuai rating yang diperbarui
new_data['cluster_label'] = new_data['cluster'].map({
    1: 'Boleh Dicoba',
    2: 'Enak',
    3: 'Lezat',
    4: 'Populer',
    5: 'Favorite!!!'
})


# In[39]:


# Gunakan TF-IDF lebih optimal
tfidf = TfidfVectorizer(stop_words='english', max_features=20000, ngram_range=(1, 3))
X_ingredients = tfidf.fit_transform(new_data['ingredients'].values.astype('U'))


# In[40]:


# Gunakan K-Means dengan cluster lebih optimal
optimal_clusters = 10
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
new_data['category_num'] = kmeans.fit_predict(X_ingredients)


# In[41]:


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


# In[42]:


new_data['category'] = new_data['ingredients'].apply(categorize_food)


# In[43]:


# Gunakan Naïve Bayes untuk rekomendasi
cv = TfidfVectorizer(max_features=20000, stop_words='english', ngram_range=(1, 3))
X_name = cv.fit_transform(new_data['name'].values.astype('U'))

model_recommendation = MultinomialNB(alpha=0.5)
y = np.arange(len(new_data))
model_recommendation.fit(X_name, y)


# In[44]:


# **Streamlit UI**
st.title("🔍 Sistem Rekomendasi Makanan")


# In[45]:


# Tentukan folder gambar
image_folder = "Food Images"


# In[46]:


# Buat dua tab: "Cari Makanan" & "Tambah Resep"
tab1, tab2 = st.tabs(["🔎 Cari Makanan", "➕ Tambah Resep"])


# In[47]:


# Fungsi untuk memberi warna berdasarkan rating
def color_rating(val):
    color = "#FFCCCB" if int(val) <= 3 else "#C6E5B3"
    return f"background-color: {color};"


# In[48]:


# Fungsi untuk mendapatkan jalur gambar dengan pengecekan format
def get_image_path(image_name):
    for ext in [".jpg", ".png", ".jpeg"]:  # Coba beberapa kemungkinan format
        image_path = os.path.join(image_folder, image_name + ext)
        if os.path.exists(image_path):
            return image_path
    return None


# In[51]:


with tab1:
    # Pilihan pencarian
    search_option = st.radio("Pilih metode pencarian:", ["Cari berdasarkan nama", "Cari berdasarkan kategori", "Cari berdasarkan cluster"])
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
                    html_code1 = f"""
                        <h1><span style='text-weight: bold;'>{row['name']}</span></h1>
                    """
                    st.markdown(html_code1, unsafe_allow_html=True)
                    st.image(image_path, caption=row['name'])
    
                    ingredients_list = row['ingredients'].split(',')
                    html_code_ingredients = "<ul>"
                    for ingredient in ingredients_list:
                        if ingredient.strip():  # Menghindari elemen kosong
                            html_code_ingredients += f"<li>{ingredient.strip()}</li>"
                    html_code_ingredients += "</ul>"
    
                    steps_list = row['steps'].split('.')
                    html_code_steps = "<ul>"
                    for step in steps_list:
                        if step.strip():  # Menghindari elemen kosong
                            html_code_steps += f"<li>{step.strip()}</li>"
                    html_code_steps += "</ul>"
    
                    html_code = f"""
                        <table border="1">
                            <tr>
                                <td>Category:</td>
                                <td>{row['category']}</td>
                            </tr>                        
                            <tr>
                                <td>Ingredients:</td>
                                <td>{html_code_ingredients}</td>
                            </tr>
                            <tr>
                                <td>Steps:</td>
                                <td>{html_code_steps}</td>
                            </tr>
                            <tr>
                                <td>Rating:</td>
                                <td>{row['rating']}</td>
                            </tr>
                            <tr>
                                <td>Cluster:</td>
                                <td>{row['cluster_label']}</td>
                            </tr>
                        </table>
                    """
                    st.markdown(html_code, unsafe_allow_html=True)
                    st.divider()
                    st.divider()
                else:
                    st.write(f"⚠️ Gambar tidak ditemukan: {row['image']}")
    
            # st.dataframe(recommended_items.style.applymap(color_rating, subset=['rating']), width=900)
    elif search_option == "Cari berdasarkan kategori":
        category_name = st.selectbox("Pilih kategori:", new_data['category'].unique())
        if category_name:
            matching_items = new_data[new_data['category'] == category_name].sample(5).dropna().reset_index(drop=True)
    
            st.subheader(f"Makanan dalam kategori '{category_name}':")
    
            for _, row in matching_items.iterrows():
                image_path = get_image_path(row['image'].split('.')[0])
                if image_path:
                    html_code1 = f"""
                        <h1><span style='text-weight: bold;'>{row['name']}</span></h1>
                    """
                    st.markdown(html_code1, unsafe_allow_html=True)
                    st.image(image_path, caption=row['name'])
    
                    ingredients_list = row['ingredients'].split(',')
                    html_code_ingredients = "<ul>"
                    for ingredient in ingredients_list:
                        if ingredient.strip():  # Menghindari elemen kosong
                            html_code_ingredients += f"<li>{ingredient.strip()}</li>"
                    html_code_ingredients += "</ul>"
    
                    steps_list = row['steps'].split('.')
                    html_code_steps = "<ul>"
                    for step in steps_list:
                        if step.strip():  # Menghindari elemen kosong
                            html_code_steps += f"<li>{step.strip()}</li>"
                    html_code_steps += "</ul>"
    
                    html_code = f"""
                        <table border="1">
                            <tr>
                                <td>Category:</td>
                                <td>{row['category']}</td>
                            </tr>                        
                            <tr>
                                <td>Ingredients:</td>
                                <td>{html_code_ingredients}</td>
                            </tr>
                            <tr>
                                <td>Steps:</td>
                                <td>{html_code_steps}</td>
                            </tr>
                            <tr>
                                <td>Rating:</td>
                                <td>{row['rating']}</td>
                            </tr>
                            <tr>
                                <td>Cluster:</td>
                                <td>{row['cluster_label']}</td>
                            </tr>
                        </table>
                    """
                    st.markdown(html_code, unsafe_allow_html=True)
                    st.divider()
                    st.divider()
                else:
                    st.write(f"⚠️ Gambar tidak ditemukan: {row['image']}")
    
            # st.dataframe(matching_items.style.applymap(color_rating, subset=['rating']), width=900)
    elif search_option == "Cari berdasarkan cluster":
        cluster_label = st.selectbox("Pilih cluster:", new_data['cluster_label'].unique())
        if cluster_label:
            matching_items = new_data[new_data['cluster_label'] == cluster_label].sample(5).dropna().reset_index(drop=True)
    
            st.subheader(f"Makanan dalam cluster '{cluster_label}':")
    
            for _, row in matching_items.iterrows():
                image_path = get_image_path(row['image'].split('.')[0])
                if image_path:
                    html_code1 = f"""
                        <h1><span style='text-weight: bold;'>{row['name']}</span></h1>
                    """
                    st.markdown(html_code1, unsafe_allow_html=True)
                    st.image(image_path, caption=row['name'])
    
                    ingredients_list = row['ingredients'].split(',')
                    html_code_ingredients = "<ul>"
                    for ingredient in ingredients_list:
                        if ingredient.strip():  # Menghindari elemen kosong
                            html_code_ingredients += f"<li>{ingredient.strip()}</li>"
                    html_code_ingredients += "</ul>"
    
                    steps_list = row['steps'].split('.')
                    html_code_steps = "<ul>"
                    for step in steps_list:
                        if step.strip():  # Menghindari elemen kosong
                            html_code_steps += f"<li>{step.strip()}</li>"
                    html_code_steps += "</ul>"
    
                    html_code = f"""
                        <table border="1">
                            <tr>
                                <td>Category:</td>
                                <td>{row['category']}</td>
                            </tr>                        
                            <tr>
                                <td>Ingredients:</td>
                                <td>{html_code_ingredients}</td>
                            </tr>
                            <tr>
                                <td>Steps:</td>
                                <td>{html_code_steps}</td>
                            </tr>
                            <tr>
                                <td>Rating:</td>
                                <td>{row['rating']}</td>
                            </tr>
                            <tr>
                                <td>Cluster:</td>
                                <td>{row['cluster_label']}</td>
                            </tr>
                        </table>
                    """
                    st.markdown(html_code, unsafe_allow_html=True)
                    st.divider()
                    st.divider()
                else:
                    st.write(f"⚠️ Gambar tidak ditemukan: {row['image']}")
    
            # st.dataframe(matching_items.style.applymap(color_rating, subset=['rating']), width=900)
with tab2:
    st.subheader("➕ Tambah Resep Baru ke Dataset")
    with st.form("Tambah Resep"):
        new_name = st.text_input("Nama Makanan")
        new_ingredients = st.text_area("Bahan-bahan (pisahkan dengan koma)")
        new_steps = st.text_area("Langkah-langkah")
        new_rating = st.slider("Rating (1-5)", 1, 5)
        submitted = st.form_submit_button("Tambahkan")

        if submitted:
            new_entry = {
                "id": len(data) + 1,
                "name": new_name,
                "ingredients": new_ingredients,
                "steps": new_steps,
                "image": "default.jpg",
                "rating": new_rating
            }

            data = pd.concat([data, pd.DataFrame([new_entry])], ignore_index=True)
            data.to_csv("dataset.csv", index=False)
            st.success("✅ Resep berhasil ditambahkan!")
    

    


# In[ ]:




