{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "6ea63182-3b5a-46ed-ab52-e45cb3fd7d44",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-05-31 21:39:55.108 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-31 21:39:55.118 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-31 21:39:55.122 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-31 21:39:55.125 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-31 21:39:55.127 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-31 21:39:55.131 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-31 21:39:55.135 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-31 21:39:55.139 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-31 21:39:55.146 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-31 21:39:55.148 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-31 21:39:55.151 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-31 21:39:55.153 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-31 21:39:55.155 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-05-31 21:39:55.156 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import random\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.naive_bayes import MultinomialNB\n",
    "from sklearn.cluster import KMeans\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "# Load dataset\n",
    "data = pd.read_csv(\"dataset.csv\")\n",
    "\n",
    "# Rename columns\n",
    "data.rename(columns={'Unnamed: 0': 'id', 'Title': 'name', 'Ingredients': 'ingredients',\n",
    "                     'Instructions': 'steps', 'Image_Name': 'image'}, inplace=True)\n",
    "\n",
    "# Pilih kolom yang relevan\n",
    "new_data = data[['id', 'name', 'ingredients', 'steps', 'image']]\n",
    "\n",
    "# Tambahkan kolom rating dengan nilai random antara 1-5\n",
    "new_data['rating'] = np.random.randint(1, 6, size=len(new_data)) \n",
    "\n",
    "# Tetapkan cluster berdasarkan nilai rating\n",
    "new_data['cluster'] = np.where(new_data['rating'] <= 3, 0, 1)  \n",
    "new_data['cluster_label'] = new_data['cluster'].map({0: 'biasa saja', 1: 'favorit'})\n",
    "\n",
    "# Gunakan TF-IDF lebih optimal\n",
    "tfidf = TfidfVectorizer(stop_words='english', max_features=20000, ngram_range=(1, 3))\n",
    "X_ingredients = tfidf.fit_transform(new_data['ingredients'].values.astype('U'))\n",
    "\n",
    "# Gunakan K-Means dengan cluster lebih optimal\n",
    "optimal_clusters = 10\n",
    "kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)\n",
    "new_data['category_num'] = kmeans.fit_predict(X_ingredients)\n",
    "\n",
    "# Pemetaan kategori lebih rinci berdasarkan ingredients\n",
    "def categorize_food(ingredients):\n",
    "    ingredients = ingredients.lower()\n",
    "\n",
    "    if any(item in ingredients for item in ['beef', 'lamb', 'pork', 'chicken']):\n",
    "        return \"Meat-Based\"\n",
    "    elif any(item in ingredients for item in ['fish', 'shrimp', 'salmon']):\n",
    "        return \"Seafood\"\n",
    "    elif any(item in ingredients for item in ['spinach', 'kale', 'broccoli']):\n",
    "        return \"Vegetarian\"\n",
    "    elif any(item in ingredients for item in ['rice', 'bread', 'pasta']):\n",
    "        return \"Bakery\"\n",
    "    elif any(item in ingredients for item in ['milk', 'cheese', 'yogurt']):\n",
    "        return \"Dairy-Based\"\n",
    "    else:\n",
    "        return \"Other\"\n",
    "\n",
    "new_data['category'] = new_data['ingredients'].apply(categorize_food)\n",
    "\n",
    "# Gunakan Naïve Bayes untuk rekomendasi\n",
    "cv = TfidfVectorizer(max_features=20000, stop_words='english', ngram_range=(1, 3))\n",
    "X_name = cv.fit_transform(new_data['name'].values.astype('U'))\n",
    "\n",
    "model_recommendation = MultinomialNB(alpha=0.5)\n",
    "y = np.arange(len(new_data))\n",
    "model_recommendation.fit(X_name, y)\n",
    "\n",
    "# **Streamlit UI**\n",
    "st.title(\"🔍 Sistem Rekomendasi Makanan\")\n",
    "\n",
    "# Form Input\n",
    "search_option = st.radio(\"Pilih metode pencarian:\", [\"Cari berdasarkan nama\", \"Cari berdasarkan kategori\", \"Cari berdasarkan cluster\"])\n",
    "\n",
    "if search_option == \"Cari berdasarkan nama\":\n",
    "    food_name = st.text_input(\"Masukkan nama makanan:\")\n",
    "    if food_name:\n",
    "        query_vec = cv.transform([food_name.lower()])\n",
    "        predicted_probs = model_recommendation.predict_proba(query_vec)[0]\n",
    "        top_5_indices = np.argsort(predicted_probs)[-5:][::-1]\n",
    "        recommended_items = new_data.iloc[top_5_indices]\n",
    "        st.subheader(f\"🍽️ Rekomendasi untuk: {food_name}\")\n",
    "        st.table(recommended_items[['name', 'category', 'ingredients', 'rating', 'cluster_label']])\n",
    "\n",
    "elif search_option == \"Cari berdasarkan kategori\":\n",
    "    category_name = st.selectbox(\"Pilih kategori:\", new_data['category'].unique())\n",
    "    if category_name:\n",
    "        matching_items = new_data[new_data['category'] == category_name].sample(5)\n",
    "        st.subheader(f\"Makanan dalam kategori '{category_name}':\")\n",
    "        st.table(matching_items[['id', 'name', 'ingredients', 'category', 'rating', 'cluster_label']])\n",
    "\n",
    "elif search_option == \"Cari berdasarkan cluster\":\n",
    "    cluster_label = st.selectbox(\"Pilih cluster:\", new_data['cluster_label'].unique())\n",
    "    if cluster_label:\n",
    "        matching_items = new_data[new_data['cluster_label'] == cluster_label].sample(5)\n",
    "        st.subheader(f\"Makanan dalam cluster '{cluster_label}':\")\n",
    "        st.table(matching_items[['id', 'name', 'ingredients', 'category', 'rating', 'cluster_label']])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "6e853501-5d2d-41da-ad33-bf0f5855c921",
   "metadata": {},
   "outputs": [
    {
     "ename": "SyntaxError",
     "evalue": "invalid syntax (2328615564.py, line 1)",
     "output_type": "error",
     "traceback": [
      "\u001b[1;36m  Cell \u001b[1;32mIn[7], line 1\u001b[1;36m\u001b[0m\n\u001b[1;33m    pip list | grep streamlit  # Jika Streamlit sudah terinstal, akan muncul dalam daftar\u001b[0m\n\u001b[1;37m        ^\u001b[0m\n\u001b[1;31mSyntaxError\u001b[0m\u001b[1;31m:\u001b[0m invalid syntax\n"
     ]
    }
   ],
   "source": [
    "pip list | grep streamlit  # Jika Streamlit sudah terinstal, akan muncul dalam daftar\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "ba3d0f67-590b-4b96-bc8d-99e134fa909a",
   "metadata": {},
   "outputs": [
    {
     "ename": "SyntaxError",
     "evalue": "invalid syntax (816758482.py, line 1)",
     "output_type": "error",
     "traceback": [
      "\u001b[1;36m  Cell \u001b[1;32mIn[8], line 1\u001b[1;36m\u001b[0m\n\u001b[1;33m    pip list\u001b[0m\n\u001b[1;37m        ^\u001b[0m\n\u001b[1;31mSyntaxError\u001b[0m\u001b[1;31m:\u001b[0m invalid syntax\n"
     ]
    }
   ],
   "source": [
    "pip list\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5159108b-0180-4fd5-a958-26fbdb95f8d5",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6ca4f623-130b-46c7-a3f8-fe483c4bfb12",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
