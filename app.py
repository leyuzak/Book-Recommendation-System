import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="book recommender", layout="wide")
st.title("book recommendation system")

# ---- yükleme (model + veri) ----
@st.cache_resource
def load_model():
    # model dosyaları root'taysa "." çalışır
    # eğer "all-minilm-l6-v2" klasörüne koyduysan: return SentenceTransformer("all-minilm-l6-v2")
    return SentenceTransformer(".")

@st.cache_data
def load_data():
    df = pd.read_csv("books.csv", on_bad_lines="skip", encoding="utf-8")
    df = df[["title", "authors"]].copy()
    df["title"] = df["title"].fillna("")
    df["authors"] = df["authors"].fillna("")
    df["text"] = (df["title"] + " " + df["authors"]).str.lower()
    return df

@st.cache_data
def build_embeddings(texts):
    model = load_model()
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return emb

df = load_data()
emb = build_embeddings(df["text"].tolist())

# ---- ui ----
q = st.text_input("kitap adı (ör: The Hobbit)", value="The Hobbit")
top_n = st.slider("kaç öneri", 3, 20, 10)

if st.button("öner"):
    m = df[df["title"].str.contains(q, case=False, na=False)]
    if len(m) == 0:
        st.error("kitap bulunamadı (başlığın bir kısmını yazmayı dene)")
    else:
        idx = int(m.index[0])

        sims = cosine_similarity(emb[idx:idx+1], emb).ravel()
        order = np.argsort(-sims)

        rec = []
        for j in order:
            if j == idx:
                continue
            rec.append([df.loc[j, "title"], df.loc[j, "authors"], float(sims[j])])
            if len(rec) >= top_n:
                break

        st.subheader("öneriler")
        st.dataframe(pd.DataFrame(rec, columns=["title", "authors", "score"]))
