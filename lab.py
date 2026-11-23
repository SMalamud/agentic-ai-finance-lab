"""Helper code for the notebook "Agentic AI in Finance: build an analyst agent".

The notebook keeps the ideas on screen (the tools, the agent loop, the guardrail).
This file holds the plumbing: loading data, drawing charts, formatting output,
and replaying the recorded model. Nothing here calls a paid service.
"""
import difflib
import html
import inspect
import json
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import HTML, display

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FILES = ["companies.csv", "prices.csv", "passages.csv", "embeddings.npy", "recorded_brain.json"]

# Filled in by setup()
companies = prices = returns = market = news = vectors = None
_recorded = {}
_model = None

NAVY, GREEN, AMBER, RED, SLATE, BLUE = "#1F2A44", "#059669", "#B45309", "#B91C1C", "#64748B", "#2563EB"
SECTOR_COLORS = {"Technology": "#2563EB", "Financials": "#059669", "Health Care": "#DB2777",
                 "Consumer": "#D97706", "Industrials": "#64748B", "Energy": "#7C3AED", "Telecom": "#0891B2"}


# --------------------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------------------
def setup(repo=None, data_dir="data"):
    """Load the data. Files are downloaded from `repo` unless they are already in `data_dir`."""
    global companies, prices, returns, market, news, vectors, _recorded
    d = Path(data_dir)
    d.mkdir(exist_ok=True)
    for name in FILES:
        if not (d / name).exists():
            if not repo:
                raise FileNotFoundError(f"{d / name} is missing and no download address was given.")
            urllib.request.urlretrieve(f"{repo}/data/{name}", d / name)

    companies = pd.read_csv(d / "companies.csv").set_index("ticker")
    long = pd.read_csv(d / "prices.csv", parse_dates=["date"])
    prices = long.pivot(index="date", columns="ticker", values="close")
    returns = prices.pct_change().dropna(how="all")
    market = returns.mean(axis=1)                    # average move of the 30 stocks
    news = pd.read_csv(d / "passages.csv", parse_dates=["date"], keep_default_na=False)
    vectors = np.load(d / "embeddings.npy").astype(np.float32)
    _recorded = json.loads((d / "recorded_brain.json").read_text())

    real = news[~news.doc_id.str.startswith("PLANTED")]
    _card("Ready",
          f"<b>{prices.shape[1]} stocks</b>, daily prices from {prices.index[0]:%d %b %Y} to {prices.index[-1]:%d %b %Y}"
          f"<br><b>{len(news):,} text passages</b> from {real.groupby(['ticker', 'date']).ngroups} earnings press releases "
          f"filed with the US SEC<br><b>{len(_recorded['runs'])} recorded agent runs</b>. No API key, no cost.", GREEN)


# --------------------------------------------------------------------------------------
# Small display helpers
# --------------------------------------------------------------------------------------
def _esc(x):
    # each dollar sign gets its own element, so notebooks do not mistake "$5 to $6" for a formula
    return html.escape(str(x)).replace("$", "<span>$</span>")


def _card(label, body, color=NAVY, tag=""):
    tag = f"<span style='float:right;color:{SLATE};font-size:12px'>{tag}</span>" if tag else ""
    display(HTML(
        f"<div class='tex2jax_ignore mathjax_ignore' style='border-left:5px solid {color};background:#F8FAFC;color:#0F172A;padding:8px 14px;"
        f"margin:6px 0;border-radius:4px;font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;"
        f"font-size:14px;line-height:1.45'>{tag}<div style='color:{color};font-weight:700;font-size:12px;"
        f"letter-spacing:.06em;margin-bottom:3px'>{_esc(label).upper()}</div>{body}</div>"))


def _table(rows):
    cols = list(rows[0].keys())
    head = "".join(f"<th style='text-align:left;padding:2px 14px 2px 0;color:{SLATE}'>{_esc(c)}</th>" for c in cols)
    body = "".join("<tr>" + "".join(f"<td style='padding:2px 14px 2px 0'>{_esc(r[c])}</td>" for c in cols) + "</tr>"
                   for r in rows)
    return f"<table style='border-collapse:collapse;font-size:14px;color:#0F172A'><tr>{head}</tr>{body}</table>"


def _docs(rows, chars=380):
    out = ""
    for r in rows:
        text = r["text"] if len(r["text"]) <= chars else r["text"][:chars].rsplit(" ", 1)[0] + " ..."
        score = f" &nbsp;·&nbsp; similarity {r['score']}" if "score" in r else ""
        out += (f"<div style='margin:5px 0 9px 0'><b>{_esc(r['doc_id'])}</b> &nbsp;·&nbsp; {_esc(r['title'][:95])}{score}"
                f"<br><span style='color:#334155'>{_esc(text)}</span></div>")
    return out


def _format(result):
    if isinstance(result, list) and result and isinstance(result[0], dict):
        return _docs(result) if "text" in result[0] else _table(result)
    if isinstance(result, dict):
        return _table([result])
    return _esc(result)


# --------------------------------------------------------------------------------------
# Part 1: prices
# --------------------------------------------------------------------------------------
def plot_price(ticker, mark=None):
    """Price chart for one stock. `mark` is an optional list of dates to highlight."""
    import matplotlib.pyplot as plt
    s = prices[ticker]
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(s.index, s.values, color=NAVY, lw=1.4)
    for d in mark or []:
        d = pd.Timestamp(d)
        ax.scatter([d], [s[d]], color=RED, zorder=3, s=45)
        ax.annotate(f"{d:%d %b %Y}\n{returns.loc[d, ticker]:+.1%}", (d, s[d]), textcoords="offset points",
                    xytext=(8, 8), fontsize=9, color=RED)
    ax.set_title(f"{companies.loc[ticker, 'company']} ({ticker}), daily closing price in USD", loc="left", fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.25)
    plt.show()


# --------------------------------------------------------------------------------------
# Part 2: embeddings
# --------------------------------------------------------------------------------------
def embed(text):
    """Turn a sentence (or a list of sentences) into a vector of 384 numbers of length 1."""
    global _model
    cached = _recorded.get("query_vectors", {})
    if isinstance(text, str) and text in cached:       # queries of the recorded runs are stored
        return np.array(cached[text], dtype=np.float32)
    if _model is None:
        _model = _load_model()
    return _model.encode(text, normalize_embeddings=True, show_progress_bar=False)


def _load_model():
    """Load the small open embedding model, keeping the screen free of technical messages."""
    import contextlib, io, logging, os, warnings
    print("Loading the embedding model (once, about 20 seconds) ...")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    for name in ["huggingface_hub", "sentence_transformers", "transformers"]:
        logging.getLogger(name).setLevel(logging.ERROR)
    # the libraries print notices about access tokens in red; none of them matters here
    with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            import subprocess, sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "sentence-transformers"])
            from sentence_transformers import SentenceTransformer
        try:
            from transformers.utils import logging as hf_logging
            hf_logging.set_verbosity_error(); hf_logging.disable_progress_bar()
        except Exception:
            pass
        model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    print("Done.")
    return model


def compare(sentences):
    """Show how similar each pair of sentences is (1 = same meaning, 0 = unrelated)."""
    V = embed(list(sentences))
    S = V @ V.T
    rows = ""
    for i, a in enumerate(sentences):
        cells = ""
        for j in range(len(sentences)):
            shade = int(255 - 150 * max(0.0, min(1.0, float(S[i, j]))))
            cells += (f"<td style='background:rgb({shade},{shade + (255 - shade) // 3},255);color:#0F172A;"
                      f"padding:5px 12px;text-align:center'>{S[i, j]:.2f}</td>")
        rows += f"<tr><td style='padding:5px 12px 5px 0;color:#0F172A;text-align:left'><b>{i + 1}.</b> {_esc(a)}</td>{cells}</tr>"
    head = "".join(f"<th style='padding:5px 12px;color:{SLATE}'>{j + 1}</th>" for j in range(len(sentences)))
    display(HTML(f"<table style='border-collapse:collapse;font-size:14px;background:#FFFFFF'>"
                 f"<tr><th></th>{head}</tr>{rows}</table>"))


def in_period(dates, when):
    """True for dates in the year `when` ('2025') or within 5 days of the date `when` ('2025-04-17')."""
    if when is None:
        return pd.Series(True, index=dates.index)
    when = str(when)
    if re.fullmatch(r"\d{4}", when):
        return dates.dt.year == int(when)
    return (dates - pd.Timestamp(when)).abs().dt.days <= 5


def as_records(docs):
    """Turn rows of the passage table into the plain list a tool hands back to the model."""
    return [{"doc_id": r.doc_id, "date": f"{r.date:%Y-%m-%d}", "title": r.title, "text": r.text,
             "score": round(float(r.score), 2)} for r in docs.itertuples()]


def search(query, ticker=None, when=None, k=5):
    """Meaning-based search over all passages, shown as a list."""
    docs = news[in_period(news.date, when)]
    if ticker:
        docs = docs[docs.ticker == ticker]
    best = docs.assign(score=vectors[docs.index] @ embed(query)).nlargest(k, "score")
    _card(f"Meaning search: {query}", _docs(as_records(best)), BLUE)


def keyword_search(word, k=5):
    """Old-style search: passages that contain the exact word or phrase."""
    hits = news[news.text.str.contains(word, case=False, regex=False)]
    body = _docs(as_records(hits.head(k).assign(score=1.0))).replace(" &nbsp;·&nbsp; similarity 1.0", "")
    _card(f"Keyword search: {word}", f"<b>{len(hits)} passages contain this exact text.</b>{body}", SLATE)


def show_map(query=None, k=15):
    """Every passage as a dot. Dots that are close have similar meaning. A query lights up its best matches."""
    import plotly.graph_objects as go
    fig = go.Figure()
    for sector, g in news.groupby("sector"):
        fig.add_trace(go.Scattergl(
            x=g.x, y=g.y, mode="markers", name=sector,
            marker=dict(size=5, color=SECTOR_COLORS.get(sector, SLATE), opacity=0.25 if query else 0.7),
            text=[f"<b>{c}</b> {d:%Y-%m-%d}<br>{t[:110]}..." for c, d, t in zip(g.company, g.date, g.text)],
            hoverinfo="text"))
    if query:
        best = news.assign(score=vectors @ embed(query)).nlargest(k, "score")
        fig.add_trace(go.Scattergl(
            x=best.x, y=best.y, mode="markers", name="best matches",
            marker=dict(size=13, color=RED, symbol="star", line=dict(width=1, color="white")),
            text=[f"<b>{c}</b> {d:%Y-%m-%d}<br>{t[:110]}..." for c, d, t in zip(best.company, best.date, best.text)],
            hoverinfo="text"))
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor="white",
                      title=f"Map of meaning{': ' + query if query else ''}",
                      xaxis=dict(visible=False), yaxis=dict(visible=False), legend=dict(itemsizing="constant"))
    fig.show()


# --------------------------------------------------------------------------------------
# Part 3: the agent
# --------------------------------------------------------------------------------------
def describe_tools(tools):
    """Show the tool menu exactly as the model receives it: name, inputs, one-line description."""
    rows = "".join(f"<div style='margin:3px 0'><code style='color:{NAVY};background:#E2E8F0;padding:1px 5px;"
                   f"border-radius:3px'>{name}{_esc(inspect.signature(f))}</code><br>"
                   f"<span style='color:#334155'>{_esc((f.__doc__ or '').strip())}</span></div>"
                   for name, f in tools.items())
    _card("Tool menu given to the model", rows, NAVY)


def show_question(question):
    _card("Question", f"<b>{_esc(question)}</b>", NAVY)


def show_reply(step, reply):
    thought = f"<i>{_esc(reply.get('thought', ''))}</i>"
    if "answer" in reply:
        _card("Answer", f"{thought}<br><br>{_esc(reply['answer'])}".replace("\n", "<br>"), GREEN, f"step {step}")
    else:
        args = ", ".join(f"{k}={v!r}" for k, v in reply["args"].items())
        _card("Think, then act", f"{thought}<br><code style='color:{AMBER};background:#FEF3C7;padding:1px 5px;"
              f"border-radius:3px'>{_esc(reply['tool'])}({_esc(args)})</code>", AMBER, f"step {step}")


def show_result(result):
    stopped = isinstance(result, str) and result.startswith(("BLOCKED", "ERROR", "STOPPED"))
    _card(result.split()[0].rstrip(":") if stopped else "Observe", _format(result), RED if stopped else BLUE)


class RecordedBrain:
    """A stand-in for a language model. It replays replies that were written in advance.

    The replies were written by Claude, an Anthropic model, working through this same loop
    with these same tools and data. Replaying them costs nothing and gives the same result
    every time. The limits: it only knows the questions listed in `brain.questions`, and it
    does not read the tool results again, so it repeats its recorded moves whatever they say
    (except that it takes its recorded fallback when a move is blocked or fails).
    """

    def __init__(self):
        self.runs = _recorded["runs"]
        self.questions = list(self.runs)

    def menu(self):
        rows = "".join(f"<div style='margin:2px 0'><b>{i}.</b> {_esc(q)}"
                       + (f" <span style='color:{RED}'>(staged for Part 4)</span>" if self.runs[q].get("staged") else "")
                       + "</div>" for i, q in enumerate(self.questions))
        _card("Questions the recorded model can answer", rows, NAVY)

    def __call__(self, history):
        question = history[0]["content"]
        match = difflib.get_close_matches(question, self.questions, n=1, cutoff=0.8)
        if not match:
            raise ValueError("The recorded model has no reply for this question. "
                             "Run brain.menu() to see the questions it knows.")
        node = self.runs[match[0]]["start"]
        for message in history:
            if message["role"] == "tool":
                blocked = isinstance(message["content"], str) and message["content"].startswith(("BLOCKED", "ERROR"))
                node = node["next"]["blocked" if blocked and "blocked" in node["next"] else "ok"]
        return {k: v for k, v in node.items() if k != "next"}


class Game:
    """You play the model: you see the question and the tool menu, and you choose each move."""

    def __init__(self, question, tools):
        self.question, self.tools, self.steps = question, tools, 0
        show_question(question)
        describe_tools(tools)

    def act(self, tool, **args):
        self.steps += 1
        show_reply(self.steps, {"thought": "Your move.", "tool": tool, "args": args})
        try:
            show_result(self.tools[tool](**args))
        except Exception as e:                           # a wrong tool name or input is part of the game
            _card("Error", f"{type(e).__name__}: {_esc(e)}. The model would see this error and try again.", RED)

    def answer(self, text):
        show_reply(self.steps + 1, {"thought": "Your answer.", "answer": text})
        _card("Score", f"You used {self.steps} tool call{'s' if self.steps != 1 else ''}. "
              "Now compare with the recorded model in the next cell.", SLATE)


# --------------------------------------------------------------------------------------
# Part 4: a simulated action, so that nothing real can happen
# --------------------------------------------------------------------------------------
outbox = []


def send_email(to, body):
    """Send an email. (Simulated: the message is only stored in lab.outbox.)"""
    outbox.append({"to": to, "body": body})
    return f"Email sent to {to}."
