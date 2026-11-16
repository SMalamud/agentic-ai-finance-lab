"""The three tools of the agent, exactly as they appear in the notebook."""
PRICE_TOOLS = '''def biggest_moves(ticker, year, n=3):
    """The n largest one-day price moves of a stock in a given year."""
    moves = lab.returns[ticker].loc[str(year)]
    top = moves.abs().nlargest(n).index.sort_values()
    return [{"date": f"{d:%Y-%m-%d}", "move": f"{moves[d]:+.1%}"}
            for d in top]


def compare_to_market(ticker, date):
    """A stock's move on one date, next to the average move of all 30 stocks."""
    return {"date": date,
            ticker: f"{lab.returns.loc[date, ticker]:+.1%}",
            "average of the 30 stocks": f"{lab.market[date]:+.1%}"}
'''

SEARCH_TOOL = '''def search_news(query, ticker=None, when=None, k=3):
    """The k passages closest in meaning to the query.
    Optional filters: ticker, and when = a year ('2025')
    or a date ('2025-04-17', searched within 5 days)."""
    docs = lab.news[lab.in_period(lab.news.date, when)]
    if ticker:
        docs = docs[docs.ticker == ticker]
    if docs.empty:
        return "No documents found for these filters."
    scores = lab.vectors[docs.index] @ lab.embed(query)   # similarity
    best = docs.assign(score=scores).nlargest(k, "score")
    return lab.as_records(best)
'''
