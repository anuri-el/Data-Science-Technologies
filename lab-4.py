import os
import re
import json
import spacy
import numpy as np
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone
import feedparser, urllib.request


SEP = "=" * 67

load_dotenv()
API_KEY = os.environ.get("NEWS_API_KEY")

nlp_en = spacy.load("en_core_web_md")
nlp_uk = spacy.load("uk_core_news_md")


OUTPUT_DIR  = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CSV_FILENAME = "./outputs/l4_articles.json"

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

    articles = load_or_save_articles(CSV_FILENAME)


def detect_language(text: str) -> str:
    cyrillic_chars = sum(1 for c in text if '\u0400' <= c <= '\u04FF')
    return 'uk' if cyrillic_chars > len(text) * 0.3 else 'en'


def get_text_vector(text: str, language: str) -> np.ndarray:
    if language == 'uk':
        doc = nlp_uk(text)
    else:
        doc = nlp_en(text)
    return doc.vector if doc.vector_norm else np.zeros(300)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0.0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def keyword_matching(text: str) -> dict:
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


def assign_category(text: str) -> str:
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


def fetch_rss():
    articles = []

    for src_info in SOURCES:
        req  = urllib.request.Request(src_info["url"])
        resp = urllib.request.urlopen(req, timeout=8)
        feed = feedparser.parse(resp.read())

        if not feed.entries:
            raise ValueError("empty feed")

        source_article_count = 0

        for entry in feed.entries:
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

            articles.append(dict(
                source_id   = src_info["id"],
                source_name = src_info["name"],
                title       = title,
                body        = body[:400],
                category    = cat,
                published   = pub_dt.isoformat(),
                day         = day_str,
                hour        = pub_dt.hour,
                week_day    = pub_dt.strftime("%A"),
            ))
            source_article_count += 1

        print(f"{src_info['name']:<15}: {source_article_count} articles")

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


if __name__ == "__main__":
    main()