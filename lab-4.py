import os
import re
import json
import spacy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import feedparser, urllib.request
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from collections import Counter
from datetime import datetime, timedelta, timezone
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from scipy.ndimage import gaussian_filter


SEP = "=" * 67

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
    # {"id": "ukrinform", "name": "Ukrinform", "lang": "uk", "url": "https://www.ukrinform.ua/rss/block-lastnews"},
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

C = dict(
    bg="#0D1117", panel="#161B22", grid="#30363D",
    text="#E6EDF3", sub="#8B949E",
    pos="#4CAF50", neg="#EF5350", neu="#78909C",
)
CAT_COLORS = ["#2196F3","#FF9800","#EF5350","#00BCD4","#9C27B0","#4CAF50","#FF5722"]


def main():
    print(f"\n{SEP}")
    articles = load_or_save_articles(RAW_ARTICLES)

    df = pd.DataFrame(articles)
    for s in SOURCES:
        count = len(df[df['source_name'] == s['name']])
        print(f"  {s['name']:<15} - {count} articles")

    df.drop_duplicates(subset=["title"], inplace=True)
    df.sort_values("published", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(str(OUT_CSV), index=False, encoding="utf-8-sig")

    print(f"  Range: {df['day'].min()}  -  {df['day'].max()}")

    cube = build_olap_cube(df)
    cube_serial = {k: v.to_dict(orient="records") for k, v in cube.items()}
    with open(OLAP_JSON, "w", encoding="utf-8") as f:
        json.dump(cube_serial, f, ensure_ascii=False, indent=2, default=str)
    print(f"OLAP was saved to: {OLAP_JSON}")

    tfidf_kw = compute_tfidf_keywords(df)
    delta = compute_activity_delta(cube)

    print(f"\n{SEP}")
    print("  Activity by source and date:")
    print(f"  {'Source':<20} {'Date':<12} {'Articles':>7}  {'Avg Sentiment':>14}")
    for _, row in cube["source_date"].sort_values(["source_name","day"]).iterrows():
        print(f"  {row['source_name']:<20} {row['day']:<12} "
              f"{int(row['count']):>7}  {row['avg_sentiment']:>14.3f}")

    print(f"\n{SEP}")
    print("  Articles per category:")
    cat_total = df.groupby("category")["id"].count().sort_values(ascending=False)
    for cat, cnt in cat_total.items():
        bar = "=" * int(cnt / cat_total.max() * 10)
        print(f"  {cat:<20} {cnt:>4}  {bar}")

    if not delta.empty and "delta_pct" in delta.columns:
        print(f"\n{SEP}")
        print("  Activity change (Day 2 vs Day 1):")
        for _, row in delta.iterrows():
            sign = "+" if row["delta_pct"] >= 0 else "-"
            print(f"  {row['source_name']:<20} {sign} "
                  f"{abs(row['delta_pct']):.1f}%  "
                  f"(delta = {int(row.get('delta_abs',0)):+d} articles)")

    print(f"\n{SEP}")
    print("  TF-IDF top-10 terms per group:")
    for group_key, kws in tfidf_kw.items():
        print(f"\n  [{group_key}]")
        for word, score in kws[:10]:
            bar = "=" * int(score / kws[0][1] * 10)
            print(f"    {word:<20} {score:.4f}  {bar}")

    print(f"\n{SEP}")
    print("  Sentiment by source:")
    for src in df["source_name"].unique():
        sub = df[df["source_name"] == src]
        sc  = sub["sentiment"].value_counts()
        total = len(sub)
        pos_pct = sc.get("positive",0)/total*100
        neg_pct = sc.get("negative",0)/total*100
        neu_pct = sc.get("neutral",0)/total*100
        print(f"  {src:<20}  +: {pos_pct:.1f}%  0: {neu_pct:.1f}%  -: {neg_pct:.1f}%")


    print(f"\n{SEP}")

    plot_category_heatmap(cube, "l4_category_heatmap.png")
    plot_category_distribution(df, "l4_category_distribution.png")
    plot_sentiment_breakdown(cube, "l4_sentiment_breakdown.png")

    sources = sorted(df["source_name"].unique())
    days = sorted(df["day"].unique())
    
    plot_3d_bar_source_category_sentiment(df, cube, sources, "l4_bar_source_category.png")
    plot_3d_surface_category_hour(cube, "l4_surface_category_hour.png")
    plot_3d_waterfall_category_day(df, days, "l4_waterfall_category_day.png")


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


def lemmatize_text(text: str):
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


def compute_activity_delta(cube: dict):
    sd = cube["source_date"].copy()
    days = sorted(sd["day"].unique())
    if len(days) < 2:
        return pd.DataFrame()
    d1, d2 = days[0], days[1]

    pivot = sd.pivot(index="source_name", columns="day", values="count").fillna(0)
    if d1 in pivot.columns and d2 in pivot.columns:
        pivot["delta_pct"] = (pivot[d2] - pivot[d1]) / pivot[d1].replace(0, 1) * 100
        pivot["delta_abs"] = pivot[d2] - pivot[d1]
    return pivot.reset_index()


def plot_category_heatmap(cube: dict, filename):
    output_path = os.path.join(OUTPUT_DIR, filename)
    fig, ax = plt.subplots(figsize=(14, 6), facecolor=C["bg"])
    
    ch = cube["category_hour"].pivot(index="category", columns="hour", values="count").fillna(0)
    ch = ch.reindex(columns=range(24), fill_value=0)
    
    cmap_custom = LinearSegmentedColormap.from_list("olap", ["#0D1117", "#1565C0", "#FFA000", "#E53935"])

    im = ax.imshow(ch.values, cmap=cmap_custom, aspect="auto", interpolation="nearest")
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}" for h in range(24)], color=C["text"])
    ax.set_yticks(range(len(ch.index)))
    ax.set_yticklabels(ch.index.tolist(), color=C["text"])
    
    for i in range(len(ch.index)):
        for j in range(24):
            v = ch.values[i, j]
            if v > 0:
                ax.text(j, i, str(int(v)), color="white" if v > ch.values.max() * 0.4 else C["sub"])
    
    plt.colorbar(im, ax=ax, fraction=0.025, label="Number of Articles")
    
    title = f"Category Heatmap (Category * Hour)"
    ax.set_title(title, color=C["text"], pad=15)
    ax.set_xlabel("Hours", color=C["text"])
    ax.invert_yaxis()
    ax.set_facecolor(C["panel"])

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    print(f"'{title}' saved to: {output_path}")


def plot_category_distribution(df: pd.DataFrame, filename):
    output_path = os.path.join(OUTPUT_DIR, filename)
    fig, ax = plt.subplots(figsize=(8, 8), facecolor=C["bg"])
    
    cat_counts = df["category"].value_counts()
    wedges, texts, autotexts = ax.pie(cat_counts.values, labels=cat_counts.index, autopct="%1.1f%%", colors=CAT_COLORS[:len(cat_counts)], startangle=90)
    
    for t in texts:
        t.set_color(C["text"])
    for at in autotexts:
        at.set_color("white")
    
    title = f"Category Distribution"
    ax.set_title(title, color=C["text"], pad=15)
    ax.set_facecolor(C["panel"])

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    print(f"'{title}' saved to: {output_path}")
    

def plot_sentiment_breakdown(cube: dict, filename):
    output_path = os.path.join(OUTPUT_DIR, filename)
    fig, ax = plt.subplots(figsize=(15, 6), facecolor=C["bg"])
    
    sb = cube["sentiment_breakdown"].copy()
    sentiments = ["positive", "neutral", "negative"]
    sent_colors = [C["pos"], C["neu"], C["neg"]]
    
    x_labels = [f"{row.source_name} {row.day}" for _, row in sb.drop_duplicates(["source_name", "day"]).iterrows()]
    x_pos = np.arange(len(x_labels))
    bottoms = np.zeros(len(x_labels))
    
    for sent, sc in zip(sentiments, sent_colors):
        vals = []
        for _, row in sb.drop_duplicates(["source_name", "day"]).iterrows():
            mask = ((sb["source_name"] == row.source_name) & (sb["day"] == row.day) & (sb["sentiment"] == sent))
            vals.append(int(sb[mask]["count"].sum()))
        ax.bar(x_pos, vals, label=sent, color=sc, alpha=0.85, bottom=bottoms, edgecolor=C["bg"])
        bottoms += np.array(vals, dtype=float)
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, color=C["text"])
    ax.set_ylabel("Number of articles", color=C["text"])

    title = f"Sentiment Breakdown"
    ax.set_title(title, color=C["text"], pad=15)
    ax.legend(facecolor=C["panel"], edgecolor=C["grid"], labelcolor=C["text"])
    ax.set_facecolor(C["panel"])
    ax.tick_params(colors=C["text"])
    for spine in ax.spines.values():
        spine.set_color(C["grid"])

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    print(f"'{title}' saved to: {output_path}")


def setup_3d_axes(ax, title, xlabel="", ylabel="", zlabel="", elev=25, azim=-50):
    ax.set_facecolor(C["panel"])
    ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor(C["grid"])
    ax.yaxis.pane.set_edgecolor(C["grid"])
    ax.zaxis.pane.set_edgecolor(C["grid"])
    ax.grid(color=C["grid"], linestyle="--", linewidth=0.4, alpha=0.4)
    
    if xlabel:
        ax.set_xlabel(xlabel, color=C["sub"], labelpad=8)
    if ylabel:
        ax.set_ylabel(ylabel, color=C["sub"], labelpad=8)
    if zlabel:
        ax.set_zlabel(zlabel, color=C["sub"], labelpad=5)
    
    ax.set_title(title, color=C["text"], pad=10)
    ax.view_init(elev=elev, azim=azim)
    ax.tick_params(colors=C["sub"])


def plot_3d_bar_source_category_sentiment(df: pd.DataFrame, cube: dict, sources, filename):
    fig = plt.figure(figsize=(14, 10), facecolor=C["bg"])
    ax = fig.add_subplot(111, projection="3d")
    
    n_src = len(sources)
    n_cat = len(CATEGORIES)
    
    sc_pivot = cube["source_category"].pivot(index="source_name", columns="category", values="count").reindex(index=sources, columns=CATEGORIES).fillna(0)
    
    df["sent_score"] = df["sentiment"].map({"positive": 1, "neutral": 0, "negative": -1})
    sent_pivot = (df.groupby(["source_name", "category"])["sent_score"].mean().unstack(fill_value=0).reindex(index=sources, columns=CATEGORIES).fillna(0))
    
    cmap_sent = LinearSegmentedColormap.from_list("sent3d", ["#EF5350", "#78909C", "#4CAF50"])
    norm_sent = Normalize(vmin=-1, vmax=1)
    
    bar_width = 0.5
    bar_depth = 0.5
    
    for xi, src in enumerate(sources):
        for yi, cat in enumerate(CATEGORIES):
            height = float(sc_pivot.loc[src, cat]) if cat in sc_pivot.columns else 0
            if height == 0:
                continue
            
            sent_val = float(sent_pivot.loc[src, cat]) if (src in sent_pivot.index and cat in sent_pivot.columns) else 0
            color = cmap_sent(norm_sent(sent_val))
            
            x0, x1 = xi - bar_width/2, xi + bar_width/2
            y0, y1 = yi - bar_depth/2, yi + bar_depth/2
            z0, z1 = 0, height
            
            verts = [
                [(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)],  # front
                [(x0, y1, z0), (x1, y1, z0), (x1, y1, z1), (x0, y1, z1)],  # back
                [(x0, y0, z0), (x0, y1, z0), (x0, y1, z1), (x0, y0, z1)],  # left
                [(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)],  # right
                [(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)],  # top
            ]
            poly = Poly3DCollection(verts, alpha=0.8, facecolor=color, edgecolor=(*[c * 0.6 for c in color[:3]], 0.9), linewidth=0.4)
            ax.add_collection3d(poly)
            
            if height >= 3:
                ax.text(xi, yi, z1 + 0.5, str(int(height)), color=C["text"])
    
    ax.set_xticks(range(n_src))
    ax.set_xticklabels([s.replace(" ", "\n") for s in sources], color=C["text"])
    ax.set_yticks(range(n_cat))
    ax.set_yticklabels([c.split("/")[0][:9] for c in CATEGORIES], color=C["text"])
    

    title = f"Source * Category * Count"
    setup_3d_axes(ax, title, zlabel="Articles")
    
    sm = ScalarMappable(cmap=cmap_sent, norm=norm_sent)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, shrink=0.45, aspect=15, pad=0.08, orientation="vertical")
    cb.set_label("Sentiment Score", color=C["text"])
    cb.ax.yaxis.set_tick_params(color=C["sub"])
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=C["sub"])
    
    output_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(output_path)
    plt.show()
    print(f"'{title}' saved to: {output_path}")


def plot_3d_surface_category_hour(cube: dict, filename):
    fig = plt.figure(figsize=(14, 10), facecolor=C["bg"])
    ax = fig.add_subplot(111, projection="3d")
    n_cat = len(CATEGORIES)
    
    ch = cube["category_hour"].pivot(index="category", columns="hour", values="count").reindex(index=CATEGORIES, columns=range(24)).fillna(0)
    Z_raw = ch.values.astype(float)
    Z_smooth = gaussian_filter(Z_raw, sigma=1.2)
    
    X_h, Y_c = np.meshgrid(np.arange(24), np.arange(n_cat))
    norm_surf = Normalize(vmin=Z_smooth.min(), vmax=Z_smooth.max())
    
    cmap_heat = plt.cm.plasma
    surf = ax.plot_surface(X_h, Y_c, Z_smooth, cmap=cmap_heat, norm=norm_surf, alpha=0.88)
    
    ax.contourf(X_h, Y_c, Z_smooth, zdir="z", offset=Z_smooth.min() - 0.5, cmap=cmap_heat, alpha=0.4, levels=10)
    ax.set_xticks([0, 6, 12, 18, 23])
    ax.set_xticklabels(["00:00", "06:00", "12:00", "18:00", "23:00"], color=C["text"])
    ax.set_yticks(range(n_cat))
    ax.set_yticklabels([c.split("/")[0][:8] for c in CATEGORIES], color=C["text"])
    
    title = "Category * Hour Surface"
    setup_3d_axes(ax, title, xlabel="Hours", zlabel="Articles", elev=30, azim=225)
    
    cb = fig.colorbar(surf, ax=ax, shrink=0.45, aspect=15, pad=0.08)
    cb.set_label("Density", color=C["text"], fontsize=7.5)
    cb.ax.yaxis.set_tick_params(color=C["sub"])
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=C["sub"], fontsize=7)

        
    output_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(output_path)
    plt.show()
    print(f"'{title}' saved to: {output_path}")


def plot_3d_waterfall_category_day(df: pd.DataFrame, days, filename):
    fig = plt.figure(figsize=(14, 10), facecolor=C["bg"])
    ax = fig.add_subplot(111, projection="3d")
    
    n_cat = len(CATEGORIES)
    hours_bins = [0, 6, 9, 12, 15, 18, 21, 24]
    n_bins = len(hours_bins) - 1
    
    for di, day in enumerate(days[:2]):
        for ci, cat in enumerate(CATEGORIES):
            sub_h = df[(df["day"] == day) & (df["category"] == cat)]
            hist, _ = np.histogram(sub_h["hour"].values, bins=hours_bins)
            cumulative = np.cumsum(hist).astype(float)
            t_vals = np.array(hours_bins[:-1], dtype=float)
            
            verts_x = np.concatenate([[t_vals[0]], t_vals, [t_vals[-1]]])
            verts_z = np.concatenate([[0], cumulative, [0]])
            verts = list(zip(verts_x, verts_z))
            
            poly_verts = [[(x, ci + di * 0.25, z) for x, z in verts]]
            col = Poly3DCollection(poly_verts, alpha=0.65, facecolor=CAT_COLORS[ci % len(CAT_COLORS)])
            ax.add_collection3d(col)
            
            ax.plot(t_vals, [ci + di * 0.25] * n_bins, cumulative, color=CAT_COLORS[ci % len(CAT_COLORS)], alpha=0.9)
    
    ax.set_xticks([0, 6, 12, 18, 24])
    ax.set_xticklabels(["0:00", "6:00", "12:00", "18:00", "24:00"], color=C["text"])
    ax.set_yticks(range(n_cat))
    ax.set_yticklabels([c.split("/")[0][:9] for c in CATEGORIES], color=C["text"])
    
    title = "Waterfall Category * Day"
    setup_3d_axes(ax, title, xlabel="Hours", ylabel="Category", zlabel="Cumulative", elev=22, azim=-60)
    
    handles = [mpatches.Patch(color=CAT_COLORS[i], label=CATEGORIES[i], alpha=0.85) for i in range(n_cat)]
    fig.legend(handles=handles, loc="lower center", ncol=7, facecolor=C["panel"], edgecolor=C["grid"], labelcolor=C["text"], bbox_to_anchor=(0.5, 0.05))
    
    output_path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(output_path)
    plt.show()
    print(f"'{title}' saved to: {output_path}")


if __name__ == "__main__":
    main()