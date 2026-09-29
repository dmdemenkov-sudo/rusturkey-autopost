#!/usr/bin/env python3
"""
Автопостинг gastetv.com -> RusTurkey.com

Каждый запуск:
  1. забирает свежие статьи из разделов Gündem и Magazin через WordPress API gastetv;
  2. нейросеть оценивает заголовки (что интересно русскоязычному читателю);
  3. лучшие переводятся на русский — не больше DAILY_LIMIT в сутки, равномерно по дню;
  4. результат дописывается в RSS-ленту docs/feed.xml, которую забирает импорт Varient.

Запуск:  python autopost.py              — боевой режим (нужен ANTHROPIC_API_KEY)
         python autopost.py --mock       — без нейросети, для проверки механики
         python autopost.py --input f.json — брать статьи из файла, а не с сайта
"""
from __future__ import annotations

import argparse
import difflib
import html
import json
import math
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path
from xml.sax.saxutils import escape
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup

from prompts import SCORE_SYSTEM, TRANSLATE_SYSTEM

# ─────────────────────────── НАСТРОЙКИ ───────────────────────────
SOURCE = "https://gastetv.com"
CATEGORIES = {2: "Gündem", 9: "Magazin"}          # id рубрик на gastetv
DAILY_LIMIT = 10                                   # материалов в сутки
TOPIC_CAPS = {"celebrity": 4, "politics": 3, "life": 3, "incident": 1, "other": 1}
MIN_SCORE = 6                                      # ниже — не публикуем вообще
MIN_AGE_MIN = 40        # не брать статью моложе 40 минут: часто её ещё дописывают
MAX_AGE_H = 18          # старше — уже не новость
MIN_TEXT_CHARS = 450    # короче — заглушка, ждём дописывания
RUN_HOURS = list(range(8, 24))                     # часы запусков по Стамбулу
FEED_SIZE = 40                                     # сколько статей держать в ленте
TZ = ZoneInfo("Europe/Istanbul")

SCORE_MODEL = os.getenv("SCORE_MODEL", "claude-haiku-4-5")
TRANSLATE_MODEL = os.getenv("TRANSLATE_MODEL", "claude-sonnet-4-5")

FEED_TITLE = "RusTurkey — новости Турции (автоперевод)"
FEED_LINK = os.getenv("FEED_PUBLIC_URL", "https://example.github.io/rusturkey-autopost/feed.xml")

ROOT = Path(__file__).resolve().parent
STATE_FILE = ROOT / "state.json"
FEED_FILE = ROOT / "docs" / "feed.xml"
UA = {"User-Agent": "RusTurkeyAutopost/1.0 (+https://rusturkey.com)"}


def log(*a):
    print(datetime.now(TZ).strftime("%H:%M:%S"), *a, flush=True)


# ─────────────────────────── ИСТОЧНИК ───────────────────────────
def fetch_posts(hours: int) -> list[dict]:
    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%S")
    posts = []
    for cat_id, cat_name in CATEGORIES.items():
        r = requests.get(
            f"{SOURCE}/wp-json/wp/v2/posts",
            params={"categories": cat_id, "after": since, "per_page": 100, "_embed": "wp:featuredmedia"},
            headers=UA, timeout=30,
        )
        r.raise_for_status()
        for p in r.json():
            media = (p.get("_embedded") or {}).get("wp:featuredmedia") or [{}]
            posts.append({
                "id": p["id"],
                "cat": cat_name,
                "date_gmt": p["date_gmt"],
                "link": p["link"],
                "title": html.unescape(re.sub(r"<[^>]+>", "", p["title"]["rendered"])).strip(),
                "content": p["content"]["rendered"],
                "image": media[0].get("source_url"),
            })
    return posts


def clean_content(raw_html: str) -> tuple[str, list[str]]:
    """Текст для перевода (абзацы) + список картинок из тела статьи."""
    soup = BeautifulSoup(raw_html, "html.parser")
    for bad in soup(["script", "style", "iframe", "noscript", "form", "ins"]):
        bad.decompose()
    images = [i.get("src") for i in soup.find_all("img") if i.get("src", "").startswith("http")]
    parts = []
    for el in soup.find_all(["p", "h2", "h3", "h4", "li", "blockquote"]):
        t = el.get_text(" ", strip=True)
        if not t or re.match(r"^(ayrıca|ilgili haber|bunu da okuyun)", t, re.I):
            continue
        parts.append(t)
    return "\n\n".join(parts), images


# ─────────────────────────── НЕЙРОСЕТЬ ───────────────────────────
class LLM:
    def __init__(self, mock: bool):
        self.mock = mock
        if not mock:
            import anthropic
            self.client = anthropic.Anthropic()  # ключ из ANTHROPIC_API_KEY

    def _ask(self, model: str, system: str, user: str, max_tokens: int) -> dict:
        msg = self.client.messages.create(
            model=model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in msg.content if b.type == "text")
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise ValueError(f"Модель вернула не JSON: {text[:300]}")
        return json.loads(m.group(0))

    def score(self, candidates: list[dict], published_today: list[str]) -> list[dict]:
        if self.mock:
            return [_mock_score(c) for c in candidates]
        lines = [f'{c["id"]} | {c["cat"]} | {c["title"]}' for c in candidates]
        user = ("Уже опубликовано сегодня:\n" + ("\n".join(published_today) or "—") +
                "\n\nНовые заголовки (id | рубрика | заголовок):\n" + "\n".join(lines))
        return self._ask(SCORE_MODEL, SCORE_SYSTEM, user, 4000)["items"]

    def translate(self, post: dict, text: str) -> dict:
        if self.mock:
            return {"skip": False, "title": "[MOCK] " + post["title"],
                    "lead": "Тестовый лид.", "body_html": "".join(f"<p>{escape(p)}</p>" for p in text.split("\n\n")),
                    "tags": ["тест"]}
        user = f"Рубрика: {post['cat']}\nЗаголовок: {post['title']}\n\nТекст:\n{text}"
        return self._ask(TRANSLATE_MODEL, TRANSLATE_SYSTEM, user, 6000)


def _mock_score(c: dict) -> dict:
    t = c["title"].lower()
    if c["cat"] == "Magazin":
        return {"id": c["id"], "score": 2 if "jenner" in t else 8, "topic": "celebrity", "dup_of": None}
    if any(k in t for k in ["elektrik", "su kesintisi", "yağış", "fiyat", "havalimanı"]):
        return {"id": c["id"], "score": 7, "topic": "life", "dup_of": None}
    if any(k in t for k in ["bahçeli", "özel", "bakan", "soruşturma"]):
        return {"id": c["id"], "score": 7, "topic": "politics", "dup_of": None}
    return {"id": c["id"], "score": 3, "topic": "other", "dup_of": None}


# ─────────────────────────── СОСТОЯНИЕ ───────────────────────────
def load_state() -> dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text("utf-8"))
    return {"posts": {}, "published": [], "feed": []}


def save_state(state: dict):
    cutoff = (datetime.now(timezone.utc) - timedelta(days=5)).isoformat()
    state["posts"] = {k: v for k, v in state["posts"].items() if v["seen_at"] > cutoff}
    state["published"] = [p for p in state["published"] if p["at"] > cutoff]
    STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=1), "utf-8")


# ─────────────────────────── RSS ───────────────────────────
def build_item_html(tr: dict, post: dict, extra_images: list[str]) -> str:
    parts = [f"<p><strong>{escape(tr['lead'])}</strong></p>", tr["body_html"]]
    for src in extra_images[:6]:                       # галерея из тела статьи
        parts.append(f'<p><img src="{escape(src)}" alt="{escape(tr["title"])}"></p>')
    parts.append(f'<p>Источник: <a href="{escape(post["link"])}" rel="nofollow" target="_blank">gastetv.com</a></p>')
    return "\n".join(parts)


def write_feed(items: list[dict]):
    now = format_datetime(datetime.now(timezone.utc))
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/" '
           'xmlns:media="http://search.yahoo.com/mrss/" xmlns:atom="http://www.w3.org/2005/Atom">',
           "<channel>",
           f"<title>{escape(FEED_TITLE)}</title>",
           "<link>https://rusturkey.com</link>",
           f'<atom:link href="{escape(FEED_LINK)}" rel="self" type="application/rss+xml"/>',
           "<description>Переводы новостей gastetv.com для RusTurkey.com</description>",
           "<language>ru</language>",
           f"<lastBuildDate>{now}</lastBuildDate>"]
    for it in items:
        cdata = it["html"].replace("]]>", "]]]]><![CDATA[>")
        out.append("<item>")
        out.append(f"<title>{escape(it['title'])}</title>")
        out.append(f"<link>{escape(it['link'])}</link>")
        out.append(f'<guid isPermaLink="false">{escape(it["guid"])}</guid>')
        out.append(f"<pubDate>{it['pubDate']}</pubDate>")
        for tag in it.get("tags", []):
            out.append(f"<category>{escape(tag)}</category>")
        out.append(f"<description><![CDATA[{cdata}]]></description>")
        out.append(f"<content:encoded><![CDATA[{cdata}]]></content:encoded>")
        if it.get("image"):
            img = escape(it["image"])
            mime = "image/png" if img.lower().endswith(".png") else "image/webp" if img.lower().endswith(".webp") else "image/jpeg"
            out.append(f'<enclosure url="{img}" length="0" type="{mime}"/>')
            out.append(f'<media:content url="{img}" medium="image" type="{mime}"/>')
        out.append("</item>")
    out += ["</channel>", "</rss>"]
    FEED_FILE.parent.mkdir(parents=True, exist_ok=True)
    FEED_FILE.write_text("\n".join(out), "utf-8")


# ─────────────────────────── ОСНОВНОЙ ЦИКЛ ───────────────────────────
def allowed_this_run(now_local: datetime, published_today: int) -> int:
    left = DAILY_LIMIT - published_today
    if left <= 0:
        return 0
    runs_left = max(1, len([h for h in RUN_HOURS if h >= now_local.hour]))
    return math.ceil(left / runs_left)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="без нейросети")
    ap.add_argument("--input", help="JSON со статьями вместо запроса к сайту")
    ap.add_argument("--force", type=int, default=0, help="опубликовать N статей, игнорируя распределение по дню")
    args = ap.parse_args()

    now_utc = datetime.now(timezone.utc)
    now_local = now_utc.astimezone(TZ)
    today = now_local.date().isoformat()
    state = load_state()
    llm = LLM(args.mock)

    posts = json.loads(Path(args.input).read_text("utf-8")) if args.input else fetch_posts(MAX_AGE_H + 6)
    log(f"Получено статей: {len(posts)}")
    by_id = {str(p["id"]): p for p in posts}

    for pid, p in by_id.items():
        st = state["posts"].setdefault(pid, {"status": "new", "seen_at": now_utc.isoformat(), "title": p["title"], "cat": p["cat"]})
        st["title"] = p["title"]

    def age_min(p):
        return (now_utc - datetime.fromisoformat(p["date_gmt"]).replace(tzinfo=timezone.utc)).total_seconds() / 60

    # 1. Оценка новых заголовков
    to_score = [p for pid, p in by_id.items() if state["posts"][pid]["status"] == "new"]
    pub_today = [x for x in state["published"] if x["day"] == today]
    if to_score:
        log(f"Оцениваю заголовков: {len(to_score)}")
        for s in llm.score(to_score, [x["title_tr"] for x in pub_today]):
            st = state["posts"].get(str(s["id"]))
            if st:
                st.update(status="scored", score=int(s.get("score", 0)), topic=s.get("topic", "other"),
                          dup_of=s.get("dup_of"), why=s.get("why", ""))

    # 2. Сколько можно опубликовать сейчас
    n_allowed = min(args.force, DAILY_LIMIT - len(pub_today)) if args.force else allowed_this_run(now_local, len(pub_today))
    log(f"Сегодня опубликовано {len(pub_today)}/{DAILY_LIMIT}, в этот запуск можно: {n_allowed}")
    if n_allowed <= 0:
        save_state(state); return

    topic_used = {}
    for x in pub_today:
        topic_used[x["topic"]] = topic_used.get(x["topic"], 0) + 1

    candidates = sorted(
        [p for pid, p in by_id.items()
         if state["posts"][pid]["status"] == "scored"
         and state["posts"][pid]["score"] >= MIN_SCORE
         and not state["posts"][pid].get("dup_of")
         and (args.force or MIN_AGE_MIN <= age_min(p) <= MAX_AGE_H * 60)],
        key=lambda p: (-state["posts"][str(p["id"])]["score"], p["date_gmt"]), reverse=False,
    )

    # 3. Перевод и публикация
    used_src_titles = [state["posts"].get(str(x["id"]), {}).get("title", "") for x in pub_today]
    done = 0
    for p in candidates:
        if done >= n_allowed:
            break
        st = state["posts"][str(p["id"])]
        topic = st["topic"]
        if topic_used.get(topic, 0) >= TOPIC_CAPS.get(topic, 1):
            continue
        # страховка от дублей: почти одинаковые турецкие заголовки за сегодня
        if any(difflib.SequenceMatcher(None, p["title"].lower(), t.lower()).ratio() > 0.75 for t in used_src_titles):
            st["status"] = "duplicate"
            continue
        text, body_imgs = clean_content(p["content"])
        if len(text) < MIN_TEXT_CHARS:
            log(f"  заглушка, жду дописывания: {p['title']}")
            continue
        log(f"  перевожу [{topic} {st['score']}]: {p['title']}")
        try:
            tr = llm.translate(p, text)
        except Exception as e:
            log(f"  ошибка перевода: {e}")
            continue
        if tr.get("skip"):
            log(f"  модель пропустила: {tr.get('reason')}")
            st["status"] = "skipped"
            continue
        imgs = [i for i in body_imgs if i != p.get("image")]
        state["feed"].insert(0, {
            "guid": f"gastetv-{p['id']}",
            "title": tr["title"],
            "link": p["link"],
            "pubDate": format_datetime(now_utc),
            "html": build_item_html(tr, p, imgs),
            "image": p.get("image") or (body_imgs[0] if body_imgs else None),
            "tags": tr.get("tags", []),
        })
        state["published"].append({"id": p["id"], "day": today, "at": now_utc.isoformat(),
                                   "topic": topic, "title_tr": tr["title"]})
        st["status"] = "published"
        used_src_titles.append(p["title"])
        topic_used[topic] = topic_used.get(topic, 0) + 1
        done += 1

    state["feed"] = state["feed"][:FEED_SIZE]
    write_feed(state["feed"])
    save_state(state)
    log(f"Опубликовано в этот запуск: {done}. В ленте: {len(state['feed'])}")


if __name__ == "__main__":
    sys.exit(main())
