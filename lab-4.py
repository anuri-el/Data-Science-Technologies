import os
import re
import json
import spacy
import numpy as np
import pandas as pd
import feedparser, urllib.request
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from dotenv import load_dotenv
from collections import Counter
from datetime import datetime, timedelta, timezone

nltk.download('vader_lexicon', quiet=True)

SEP = "=" * 67

load_dotenv()
API_KEY = os.environ.get("NEWS_API_KEY")

nlp_en = spacy.load("en_core_web_md")
nlp_uk = spacy.load("uk_core_news_md")
sia = SentimentIntensityAnalyzer()


OUTPUT_DIR  = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RAW_ARTICLES = os.path.join(OUTPUT_DIR, "l4_articles.json")
OUT_CSV = os.path.join(OUTPUT_DIR, "l4_news_articles.csv")
OLAP_JSON = os.path.join(OUTPUT_DIR, "l4_olap.json")

TODAY = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
YESTERDAY = TODAY - timedelta(days=1)

SOURCES = [
    {"id": "ukrinform", "name": "Ukrinform", "lang": "uk", "url": "https://www.ukrinform.ua/rss/block-lastnews"},
    {"id": "bbc", "name": "BBC News", "lang": "en", "url": "https://feeds.bbci.co.uk/news/world/rss.xml"},
    {"id": "dw", "name": "Deutsche Welle", "lang": "en", "url": "https://rss.dw.com/rdf/rss-en-all"},
    {"id": "euronews", "name": "Euronews", "lang": "en", "url": "https://feeds.feedburner.com/euronews/en/news/"},
]

CATEGORIES = ["Politics", "Economy", "War/Security", "Technology", "Society", "Sports", "Culture"]

CATEGORY_DESCRIPTIONS = {
    "War/Security": "military conflict war attack defense army weapons security ceasefire frontline",
    "Economy": "economy business finance market trade investment budget inflation money banking",
    "Technology": "technology digital software hardware AI artificial intelligence innovation computer",
    "Sports": "sports football soccer basketball tennis olympics games competition athletic",
    "Culture": "culture arts music film theater museum festival exhibition heritage entertainment",
    "Society": "society health education climate environment social welfare population community",
    "Politics": "politics government election parliament president policy diplomatic governance voting"
}

CATEGORY_VECTORS = {}
for cat, desc in CATEGORY_DESCRIPTIONS.items():
    CATEGORY_VECTORS[cat] = nlp_en(desc).vector


def main():
    print(f"\n{SEP}")
    print(f"Range: {YESTERDAY.date()} - {TODAY.date()}")

    articles = load_or_save_articles(RAW_ARTICLES)

    df = pd.DataFrame(articles)
    for s in SOURCES:
        count = len(df[df['source_name'] == s['name']])
        print(f"  {s['name']:<15} - {count} articles")

    df.drop_duplicates(subset=["title"], inplace=True)
    df.sort_values("published", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(str(OUT_CSV), index=False, encoding="utf-8-sig")

    cube = build_olap_cube(df)
    cube_serial = {k: v.to_dict(orient="records") for k, v in cube.items()}
    with open(OLAP_JSON, "w", encoding="utf-8") as f:
        json.dump(cube_serial, f, ensure_ascii=False, indent=2, default=str)
    print(f"OLAP was saved to: {OLAP_JSON}")

    tfidf_kw = compute_tfidf_keywords(df)
    for title, terms in tfidf_kw.items():
        print(f"{title}")
        for term, val in terms[:5]:
            print(f"{term:>15} - {val:.5f}")


def detect_language(text: str):
    cyrillic_chars = sum(1 for c in text if '\u0400' <= c <= '\u04FF')
    return 'uk' if cyrillic_chars > len(text) * 0.3 else 'en'


def get_text_vector(text: str, language: str):
    if language == 'uk':
        doc = nlp_uk(text)
    else:
        doc = nlp_en(text)
    return doc.vector if doc.vector_norm else np.zeros(300)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray):
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def keyword_matching(text: str):
    text_l = text.lower()
    rules = {
        "War/Security": ["war","attack","military","missile","troops","drone",
                         "ceasefire","frontline","NATO","санкц","ракет","атак",
                         "ЗСУ","тривог","бій","оборон","security","weapon"],
        "Economy": ["economy","inflation","GDP","market","trade","budget",
                    "investment","bank","finance","price","oil","gas","EUR",
                    "економ","бюджет","курс","гривн","інфляц","ВВП"],
        "Technology": ["AI","tech","digital","software","robot","cyber","quantum",
                       "blockchain","satellite","5G","OpenAI","Apple","Google",
                       "цифров","технолог","штучн","інтелект"],
        "Sports": ["sport","football","olympic","tennis","basketball","FIFA",
                   "Champion","league","race","tournament","medal","збірна",
                   "футбол","олімп","чемпіон"],
        "Culture": ["film","music","art","festival","museum","book","theatre",
                    "cinema","heritage","культур","театр","мистецтв","кіно"],
        "Society": ["health","education","climate","migration","humanitarian",
                    "population","poverty","освіт","охорон","здоров","клімат",
                    "переміщен","громад"],
        "Politics": ["parliament","election","president","government","minister",
                     "policy","senate","vote","Рада","закон","вибор","уряд",
                     "президент","міністр","G7","EU","UN"],
    }
    
    scores = {cat: 0 for cat in rules}
    for cat, keywords in rules.items():
        for kw in keywords:
            if kw.lower() in text_l:
                scores[cat] += 1
    
    return scores


def assign_category(text: str):
    if not text or not isinstance(text, str):
        return "Politics"

    language = detect_language(text)
    
    text_vector = get_text_vector(text, language)
    
    similarities = {}
    for cat, cat_vector in CATEGORY_VECTORS.items():
        similarity = cosine_similarity(text_vector, cat_vector)
        similarities[cat] = similarity
    
    best_category = max(similarities, key=similarities.get)
    best_score = similarities[best_category]
    
    if best_score < 0.15:
        keyword_scores = keyword_matching(text)
        keyword_best = max(keyword_scores, key=keyword_scores.get)
        if keyword_scores[keyword_best] > 0:
            return keyword_best
    
    return best_category if best_score > 0 else "Politics"


def define_sentiment(text: str):
    if not text:
        return "neutral"
    
    scores = sia.polarity_scores(text)
    compound = scores['compound']
    
    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    else:
        return "neutral"


def fetch_rss():
    articles = []

    for src_info in SOURCES:
        req  = urllib.request.Request(src_info["url"])
        resp = urllib.request.urlopen(req, timeout=8)
        feed = feedparser.parse(resp.read())

        if not feed.entries:
            raise ValueError("empty feed")

        for idx, entry in enumerate(feed.entries):
            title  = entry.get("title", "")
            body = entry.get("summary", entry.get("description", ""))
            body = re.sub(r"<[^>]+>", " ", body)

            pub  = entry.get("published_parsed") or entry.get("updated_parsed")
            if pub:
                pub_dt = datetime(*pub[:6], tzinfo=timezone.utc)
            else:
                pub_dt = datetime.now(timezone.utc)

            day_str = pub_dt.strftime("%Y-%m-%d")
            if pub_dt.date() < YESTERDAY.date():
                continue

            cat = assign_category(title + " " + body)
            sentiment = define_sentiment(title + " " + body)

            articles.append(dict(
                id          = idx,
                source_id   = src_info["id"],
                source_name = src_info["name"],
                title       = title,
                body        = body[:400],
                category    = cat,
                sentiment   = sentiment,
                published   = pub_dt.isoformat(),
                day         = day_str,
                hour        = pub_dt.hour,
                week_day    = pub_dt.strftime("%A"),
            ))

    return articles


def load_or_save_articles(articles_file):
    
    if os.path.exists(articles_file):
        try:
            with open(articles_file, 'r', encoding='utf-8') as f:
                articles = json.load(f)
                print(f"Loaded from {articles_file}")
            return articles
        except Exception as e:
            print(f"Could not load from {articles_file}: {e}")
            articles = None
    else:
        articles = None
    
    if articles is None:
        articles = fetch_rss()
        
        try:
            with open(articles_file, 'w', encoding='utf-8') as f:
                json.dump(articles, f, ensure_ascii=False, indent=2)
            print(f"Saved {len(articles)} articles to {articles_file}")
        
        except Exception as e:
            print(f"Could not save to {articles_file}: {e}")

        return articles


def build_olap_cube(df: pd.DataFrame):
    df["sent_score"] = df["sentiment"].map({"positive": 1, "neutral": 0, "negative": -1})

    cube = {
        "source_date": df.groupby(["source_name", "day"])
                            .agg(count=("id","count"), avg_sentiment=("sent_score","mean"))
                            .reset_index(),
        "category_date": df.groupby(["category", "day"])
                            .agg(count=("id","count"), avg_sentiment=("sent_score","mean"))
                            .reset_index(),
        "source_category": df.groupby(["source_name", "category"])
                            .agg(count=("id","count"))
                            .reset_index(),
        "hour_source": df.groupby(["hour", "source_name"])
                            .agg(count=("id","count"))
                            .reset_index(),
        "sentiment_breakdown": df.groupby(["source_name", "day", "sentiment"])
                            .agg(count=("id","count"))
                            .reset_index(),
        "category_hour": df.groupby(["category", "hour"])
                            .agg(count=("id","count"))
                            .reset_index(),
    }

    return cube


STOP_WORDS = set(stopwords.words('english'))
STOP_WORDS.update("та","і","в","у","на","до","з","за","що","як","але","ще","вже",
    "це","цей","ця","ці","той","та","те","ті","вона","він","воно",
    "вони","ми","ви","ти","мене","тобі","його","її","нас","вас",
    "їх","свій","мій","твій","наш","ваш","так","ні","більше","менше",
    "дуже","трохи","багато","мало","де","куди","звідти","тут","там",)


def lemmatize_text(text: str) -> list[str]:
    if not text or not isinstance(text, str):
        return []
    
    language = detect_language(text)
    if language == 'uk':
        doc = nlp_uk(text.lower())
    else:
        doc = nlp_en(text.lower())
    
    lemmas = [
        token.lemma_ for token in doc 
        if token.is_alpha and not token.is_stop and len(token.lemma_) > 2
        and token.lemma_ not in STOP_WORDS
    ]
    return lemmas


def compute_tfidf_keywords(df: pd.DataFrame, top_n: int = 20):
    results = {}
    groups = df.groupby(["source_name", "day"])
    
    all_documents = []
    for _, row in df.iterrows():
        text = f"{row['title']} {row['body']}"
        tokens = lemmatize_text(text)
        all_documents.append(set(tokens))
    
    df_count = Counter()
    for doc_tokens in all_documents:
        for token in doc_tokens:
            df_count[token] += 1
    
    total_docs = len(df)
    
    for (src, day), group in groups:
        group_texts = (group["title"] + " " + group["body"]).tolist()
        
        tf = Counter()
        total_terms = 0
        for text in group_texts:
            tokens = lemmatize_text(text)
            for token in tokens:
                tf[token] += 1
                total_terms += 1
        
        tfidf = {}
        for word, freq in tf.items():
            tf_val = freq / max(total_terms, 1)
            idf_val = np.log((total_docs + 1) / (df_count.get(word, 0) + 1)) + 1  # Smoothing
            tfidf[word] = tf_val * idf_val
        
        top_keywords = sorted(tfidf.items(), key=lambda x: x[1], reverse=True)[:top_n]
        key_name = f"{src} | {day}"
        results[key_name] = top_keywords
            
    return results


if __name__ == "__main__":
    main()