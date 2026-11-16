"""Turn the raw EDGAR press releases into short text passages: data/passages.csv.

Usage: python3 build/03_build_corpus.py build/edgar_raw
"""
import re, sys, warnings
from pathlib import Path
import pandas as pd
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

DATA = Path(__file__).resolve().parent.parent / "data"
RAW = Path(sys.argv[1])
BLOCK = ["p", "div", "li", "td", "h1", "h2", "h3", "h4", "h5", "h6"]
MAX_CHARS, PASSAGE = 4200, 700          # prose kept per release, target passage length

SKIP = re.compile(
    r"forward[- ]looking|safe harbor|non-gaap|conference call|webcast|investor relations|"
    r"media contact|press contact|contacts?:|@|www\.|http|exhibit 99|for immediate release|"
    r"news release|press release|all rights reserved|trademark|see (the )?(accompanying|attached)|"
    r"reconciliation|page \d+|\(\d{3}\) ?\d{3}|\d{3}-\d{3}-\d{4}|footnote|refer to|"
    r"definitions? of|basis of presentation|unaudited|^table \d|in millions|click here", re.I)
TITLE = re.compile(r"report|result|announce|quarter|earnings|fiscal|revenue|deliver", re.I)


def blocks(html):
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style", "ix:header", "head"]):
        t.decompose()
    out = []
    for el in soup.find_all(BLOCK):
        if el.find(BLOCK):
            continue                     # keep leaf blocks only
        s = re.sub(r"\s+", " ", el.get_text(" ", strip=True)).strip()
        s = s.replace("​", "").replace("\xa0", " ")
        if s and (not out or out[-1] != s):
            out.append(s)
    return out


def is_prose(s):
    if len(s) < 40:
        return False
    core = s.replace(" ", "")
    alpha = sum(c.isalpha() for c in core) / len(core)
    return alpha >= 0.62 and not SKIP.search(s)


def passages(bl):
    prose, total = [], 0
    pieces = []
    for s in bl:                         # very long blocks are filtered sentence by sentence
        pieces += re.split(r"(?<=[.!?])\s+(?=[A-Z])", s) if len(s) > 1100 else [s]
    for s in pieces:
        if is_prose(s):
            prose.append(s); total += len(s)
            if total >= MAX_CHARS:
                break
    out, cur = [], ""
    for s in prose:
        for sent in (re.split(r"(?<=[.!?])\s+(?=[A-Z])", s) if len(s) > 1100 else [s]):
            if cur and len(cur) + len(sent) > PASSAGE + 300:
                out.append(cur); cur = ""
            cur = (cur + " " + sent).strip()
            if len(cur) >= PASSAGE:
                out.append(cur); cur = ""
    if len(cur) >= 200:
        out.append(cur)
    return [p if len(p) <= 1200 else p[:1200].rsplit(" ", 1)[0] + " ..." for p in out if len(p) >= 200]


def title(bl, fallback):
    for s in bl[:40]:
        if 25 <= len(s) <= 220 and TITLE.search(s) and not SKIP.search(s) and sum(c.isalpha() for c in s) / len(s) > 0.6:
            return s
    return fallback


if __name__ == "__main__":
    comp = pd.read_csv(DATA / "companies.csv").set_index("ticker")
    fl = pd.read_csv(RAW / "filings.csv"); rows = []
    for r in fl.itertuples():
        bl = blocks((RAW / r.file).read_bytes().decode("utf-8", "ignore"))
        c = comp.loc[r.ticker]
        ttl = title(bl, f"{c.company} earnings release")
        for k, p in enumerate(passages(bl)):
            rows.append(dict(doc_id=f"{r.ticker}-{r.date}-{k+1}", ticker=r.ticker, company=c.company,
                             sector=c.sector, date=r.date, title=ttl, text=p, url=r.url))
    df = pd.concat([pd.DataFrame(rows), pd.read_csv(Path(__file__).parent / "planted.csv")], ignore_index=True)
    df.to_csv(DATA / "passages.csv", index=False)
    per = df.groupby(["ticker", "date"]).size()
    print(len(df), "passages from", len(per), "of", len(fl), "releases; chars/passage",
          int(df.text.str.len().mean()), "min/max per release", per.min(), per.max())
    miss = set(zip(fl.ticker, fl.date)) - set(per.index)
    print("releases with no passages:", sorted(miss)[:20])
