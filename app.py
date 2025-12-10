
# app.py
import os
import io
import gridfs
import numpy as np
import streamlit as st
from PIL import Image
from pymongo import MongoClient
from bson import ObjectId
from skimage import color
from skimage.transform import resize
from skimage.feature import hog
from sklearn.metrics.pairwise import cosine_similarity

# ========== CONFIG ==========
# Substitua pela sua URI do Atlas:
MONGODB_URI = "mongodb+srv://root:root2@cluster0.ops04cu.mongodb.net/?retryWrites=true&w=majority&tls=true&appName=Cluster0"
DB_NAME = "faces_db"     # chumbado
COL_FILES = "fs.files"   # chumbado
COL_CHUNKS = "fs.chunks" # chumbado

# ========== CONEXÃO ==========
@st.cache_resource
def get_fs():
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=30000)
    client.admin.command("ping")  # valida conexão
    db = client[DB_NAME]
    fs = gridfs.GridFS(db)        # bucket padrão 'fs' (chumbado)
    return client, db, fs

client, db, fs = get_fs()

# ========== UTIL ==========
def pil_to_bytes(img_pil, format="JPEG"):
    buf = io.BytesIO()
    img_pil.save(buf, format=format)
    buf.seek(0)
    return buf.getvalue()

def compute_hog_embedding(img_pil, size=(160, 160)):
    # Converte para escala de cinza, redimensiona e extrai HOG (leve e robusto)
    img = np.array(img_pil)
    if img.ndim == 3:
        img_gray = color.rgb2gray(img)
    else:
        img_gray = img
    img_resized = resize(img_gray, size, anti_aliasing=True)
    feat = hog(
        img_resized,
        orientations=8, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
        block_norm="L2-Hys", feature_vector=True
    )
    # Normaliza
    feat = feat.astype(np.float32)
    norm = np.linalg.norm(feat) + 1e-8
    return (feat / norm).tolist()

def list_all_files():
    return list(db[COL_FILES].find({}).sort("uploadDate", -1))

def get_file_bytes(file_id):
    gf = fs.get(ObjectId(file_id))
    return gf.read()

def save_image_to_gridfs(img_pil, filename):
    emb = compute_hog_embedding(img_pil)
    data = pil_to_bytes(img_pil)
    file_id = fs.put(
        data,
        filename=filename,
        content_type="image/jpeg",
        embedding=emb  # salva o vetor no documento .files
    )
    return file_id

def delete_all_files():
    for doc in db[COL_FILES].find({}, {"_id": 1}):
        try:
            fs.delete(doc["_id"])
        except Exception:
            pass

def find_most_similar(img_pil, top_k=1):
    # embedding da consulta
    q_emb = np.array(compute_hog_embedding(img_pil)).reshape(1, -1)

    # carrega embeddings do banco
    docs = list_all_files()
    candidates = []
    for d in docs:
        emb = d.get("embedding")
        if emb is None:
            # se não existir, computa e atualiza
            try:
                data = get_file_bytes(d["_id"])
                img = Image.open(io.BytesIO(data)).convert("RGB")
                emb = compute_hog_embedding(img)
                db[COL_FILES].update_one({"_id": d["_id"]}, {"$set": {"embedding": emb}})
            except Exception:
                continue
        emb_vec = np.array(emb).reshape(1, -1)
        sim = cosine_similarity(q_emb, emb_vec)[0, 0]
        candidates.append((d, sim))

    # ordena por similaridade decrescente
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:top_k]

# ========== UI ==========
st.set_page_config(page_title="Faces GridFS (MongoDB Atlas)", layout="wide")

st.title("Galeria & Comparação de Faces (MongoDB Atlas + GridFS)")

# Barra de ações
col_a, col_b, col_c, col_d = st.columns(4)
with col_a:
    if st.button("🔄 Recarregar"):
        st.experimental_rerun()
with col_b:
    clear = st.button("🗑️ Limpar banco (perigoso)")
with col_c:
    st.write("")  # espaçador
with col_d:
    st.write("")  # espaçador

if clear:
    delete_all_files()
    st.success("Banco limpo (fs.files/fs.chunks).")
    st.experimental_rerun()

# Upload de novas imagens (salvar direto no GridFS)
st.subheader("Enviar novas imagens")
uploaded = st.file_uploader("Selecione uma ou mais imagens (JPG/PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded:
    for up in uploaded:
        try:
            img = Image.open(up).convert("RGB")
            fid = save_image_to_gridfs(img, filename=up.name)
            st.success(f"Enviada: {up.name} (id: {fid})")
        except Exception as e:
            st.error(f"Falha ao salvar {up.name}: {e}")

# Galeria (imagens do banco)
st.subheader("Galeria (imagens no GridFS)")
docs = list_all_files()
if not docs:
    st.info("Nenhuma imagem no GridFS.")
else:
    # grade 5 colunas
    cols = st.columns(5)
    for i, d in enumerate(docs[:50]):  # limite para performance
        try:
            data = get_file_bytes(d["_id"])
            img = Image.open(io.BytesIO(data)).convert("RGB")
            with cols[i % 5]:
                st.image(img, caption=d.get("filename"), use_container_width=True)
                st.caption(f"ID: {d['_id']}")
        except Exception:
            pass

# Comparação de faces
st.subheader("Comparar: encontre a face mais parecida")
query_up = st.file_uploader("Carregar a imagem de consulta (JPG/PNG)", type=["jpg", "jpeg", "png"])
btn_compare = st.button("🔎 Comparar")

if btn_compare and query_up:
    try:
        q_img = Image.open(query_up).convert("RGB")
        result = find_most_similar(q_img, top_k=3)

        if not result:
            st.warning("Não há candidatos no banco (ou embeddings indisponíveis).")
        else:
            st.write(f"Top {len(result)} mais parecidas (cosine):")
            res_cols = st.columns(len(result))
            for i, (doc, sim) in enumerate(result):
                data = get_file_bytes(doc["_id"])
                img = Image.open(io.BytesIO(data)).convert("RGB")
                with res_cols[i]:
                    st.image(img, caption=f"{doc.get('filename')} — sim={sim:.3f}", use_container_width=True)
                    st.caption(f"ID: {doc['_id']}")
    except Exception as e:
        st.error(f"Erro na comparação: {e}")
