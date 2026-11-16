"""Download daily adjusted closes for the Dow 30 and save data/prices.csv (long format)."""
import pandas as pd, yfinance as yf
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
COMPANIES = [  # ticker, name, sector, SEC CIK
    ("AAPL","Apple","Technology",320193), ("AMGN","Amgen","Health Care",318154),
    ("AMZN","Amazon","Consumer",1018724), ("AXP","American Express","Financials",4962),
    ("BA","Boeing","Industrials",12927), ("CAT","Caterpillar","Industrials",18230),
    ("CRM","Salesforce","Technology",1108524), ("CSCO","Cisco","Technology",858877),
    ("CVX","Chevron","Energy",93410), ("DIS","Walt Disney","Consumer",1744489),
    ("GS","Goldman Sachs","Financials",886982), ("HD","Home Depot","Consumer",354950),
    ("HON","Honeywell","Industrials",773840), ("IBM","IBM","Technology",51143),
    ("JNJ","Johnson & Johnson","Health Care",200406), ("JPM","JPMorgan Chase","Financials",19617),
    ("KO","Coca-Cola","Consumer",21344), ("MCD","McDonald's","Consumer",63908),
    ("MMM","3M","Industrials",66740), ("MRK","Merck","Health Care",310158),
    ("MSFT","Microsoft","Technology",789019), ("NKE","Nike","Consumer",320187),
    ("NVDA","Nvidia","Technology",1045810), ("PG","Procter & Gamble","Consumer",80424),
    ("SHW","Sherwin-Williams","Industrials",89800), ("TRV","Travelers","Financials",86312),
    ("UNH","UnitedHealth","Health Care",731766), ("V","Visa","Financials",1403161),
    ("VZ","Verizon","Telecom",732712), ("WMT","Walmart","Consumer",104169),
]
START, END = "2023-01-01", "2026-11-01"

if __name__ == "__main__":
    comp = pd.DataFrame(COMPANIES, columns=["ticker","company","sector","cik"])
    comp.to_csv(DATA/"companies.csv", index=False)
    px = yf.download(list(comp.ticker), start=START, end=END, auto_adjust=True, progress=False)["Close"]
    px = px.dropna(how="all")
    long = (px.stack().rename("close").reset_index()
              .rename(columns={"Date":"date","Ticker":"ticker"}))
    long["date"] = long["date"].dt.strftime("%Y-%m-%d")
    long["close"] = long["close"].round(2)
    long = long.sort_values(["ticker","date"])
    long.to_csv(DATA/"prices.csv", index=False)
    print(long.shape, long.date.min(), long.date.max(), long.ticker.nunique(), px.isna().sum().sum())
