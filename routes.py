from flask import Blueprint, render_template, request
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
import nltk
import numpy as np
import re
import os
from huggingface_hub import InferenceClient
import dotenv
dotenv.load_dotenv()

# NLTK Kontrolü
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

routes = Blueprint("routes", __name__)
MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

token = os.getenv("HF_TOKEN")
client = InferenceClient(token=token)


def get_embeddings_api(sentences):
    if not token:
        return "TOKEN_ERROR"

    try:
        # URL ile uğraşmak yerine resmi kütüphaneyi kullanıyoruz.
        # Bu fonksiyon modele "Ben sadece Feature Extraction istiyorum" der.
        response = client.feature_extraction(sentences, model=MODEL_ID)

        # Gelen veri bazen tensor, bazen liste olur; numpy dizisine çevirelim.
        return np.array(response)

    except Exception as e:
        print(f"API Hatası: {e}")
        return None

def clean_and_merge_text(text):
    """
    Metni PDF kırıklarından kurtarır ve tek bir blok haline getirir.
    """
    text = re.sub(r'-\s*\n\s*', '', text)
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'vd\.\s*\(', 'vd (', text)
    text = re.sub(r'vd\.\s*([a-z])', r'vd \1', text)
    text = re.sub(r'vb\.\s*([a-z])', r'vb \1', text)
    text = re.sub(r'Dr\.\s+', 'Dr ', text)
    return text.strip()


def group_sentences_logically(sentences):
    """
    Cümleleri 'Yazar (Yıl)' kalıbına göre mantıksal gruplara ayırır.
    """
    groups = []
    current_group = []
    year_pattern = re.compile(r'\(\d{4}\)')

    for sent in sentences:
        if year_pattern.search(sent):
            if current_group:
                groups.append(current_group)
            current_group = [sent]
        else:
            current_group.append(sent)

    if current_group:
        groups.append(current_group)

    return groups


def smart_summary_literature(text, target_limit):
    full_text = clean_and_merge_text(text)

    try:
        sentences = nltk.sent_tokenize(full_text)
    except:
        sentences = full_text.split(". ")

    if len(sentences) < 2: return text

    # --- API ÇAĞRISI BURADA ---
    try:
        # Modeli yerel çalıştırmak yerine API'den istiyoruz
        embeddings = get_embeddings_api(sentences)

        # Hata Kontrolü: API bazen liste yerine hata mesajı dönebilir
        if embeddings.ndim != 2:
            return "API Hatası: Model şu an yükleniyor olabilir, lütfen 10 saniye sonra tekrar deneyin."

    except Exception as e:
        return f"Bağlantı Hatası: {str(e)}"

    # --- KÜMELEME MANTIĞI (Aynı kalıyor) ---
    logical_groups = group_sentences_logically(sentences)
    sentence_map = {}
    flat_sentences = []
    global_idx = 0

    for g_id, group in enumerate(logical_groups):
        for s_id, sent in enumerate(group):
            flat_sentences.append(sent)
            sentence_map[global_idx] = {
                'group_id': g_id,
                'is_lead': (s_id == 0),
                'text': sent
            }
            global_idx += 1

    # Eğer API ile gelen embedding sayısı ile cümle sayısı uyuşmazsa (nadir durum)
    if len(embeddings) != len(flat_sentences):
        # Yeniden düz liste için embedding alalım (garanti olsun)
        embeddings = get_embeddings_api(flat_sentences)

    avg_sent_len = sum(len(s) for s in flat_sentences) / len(flat_sentences)
    if avg_sent_len == 0: avg_sent_len = 1

    num_clusters = int(target_limit / avg_sent_len)
    num_clusters = max(1, min(num_clusters, len(flat_sentences)))

    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    kmeans.fit(embeddings)

    selected_indices = set()
    for j in range(num_clusters):
        idx = np.where(kmeans.labels_ == j)[0]
        curr_center = kmeans.cluster_centers_[j].reshape(1, -1)
        curr_embeddings = embeddings[idx]
        closest, _ = pairwise_distances_argmin_min(curr_center, curr_embeddings)
        selected_indices.add(idx[closest[0]])

    final_indices = set(selected_indices)

    for idx in selected_indices:
        info = sentence_map[idx]
        if not info['is_lead']:
            target_group = info['group_id']
            for k, v in sentence_map.items():
                if v['group_id'] == target_group and v['is_lead']:
                    final_indices.add(k)
                    break

    sorted_indices = sorted(list(final_indices))

    final_summary = ""
    for i in sorted_indices:
        candidate = flat_sentences[i]
        if len(final_summary) + len(candidate) < target_limit + 150:
            final_summary += candidate + " "
        else:
            if not final_summary: final_summary = candidate
            break

    return final_summary.strip()


@routes.route("/", methods=["GET", "POST"])
def index():
    text = ""
    limit = ""
    semantic_ozet = ""
    kisaltma = None
    kelime_sayisi = 0
    karakter_sayisi = 0
    karakter_bosluk_yok = 0
    cumle_sayisi = 0

    if request.method == "POST":
        text = request.form.get("text", "").strip()
        limit = request.form.get("limit", "").strip()

        if text:
            cleaned_full = clean_and_merge_text(text)
            kelimeler = cleaned_full.split()
            kelime_sayisi = len(kelimeler)
            karakter_sayisi = len(cleaned_full)
            karakter_bosluk_yok = len(cleaned_full.replace(" ", ""))

            try:
                cumleler = nltk.sent_tokenize(cleaned_full)
            except:
                cumleler = cleaned_full.split(".")
            cumle_sayisi = len(cumleler)

            limit_int = 0
            if limit and limit.isdigit():
                limit_int = int(limit)
                if limit_int > 0 and len(cleaned_full) > limit_int:
                    kisaltma = cleaned_full[:limit_int] + "..."
                else:
                    kisaltma = cleaned_full
            else:
                kisaltma = cleaned_full
                limit_int = len(cleaned_full)

            if limit_int > 0 and limit_int < len(cleaned_full):
                semantic_ozet = smart_summary_literature(text, limit_int)
            else:
                semantic_ozet = cleaned_full

    return render_template(
        "index.html",
        text=text,
        limit=limit,
        kelime_sayisi=kelime_sayisi,
        karakter_sayisi=karakter_sayisi,
        karakter_bosluk_yok=karakter_bosluk_yok,
        cumle_sayisi=cumle_sayisi,
        kisaltma=kisaltma,
        semantic_ozet=semantic_ozet
    )