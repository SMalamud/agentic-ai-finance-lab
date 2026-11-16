"""Download 8-K earnings press releases (Item 2.02, Exhibit 99.1) for the Dow 30 from SEC EDGAR.

Usage:  SEC_CONTACT="Your Name you@example.org" python3 build/02_edgar_download.py build/edgar_raw

The SEC requires a declared contact in the User-Agent (https://www.sec.gov/os/accessing-edgar-data).
Raw HTML is cached in the output directory, with an index in filings.csv. Reruns skip cached files.
"""
import os, sys, time
from pathlib import Path
import requests, pandas as pd
from bs4 import BeautifulSoup

DATA = Path(__file__).resolve().parent.parent / "data"
RAW = Path(sys.argv[1]); RAW.mkdir(parents=True, exist_ok=True)
START, END = "2023-01-01", "2026-10-31"
S = requests.Session(); S.headers["User-Agent"] = os.environ["SEC_CONTACT"]


def get(url, tries=4):
    for k in range(tries):
        r = S.get(url, timeout=30); time.sleep(0.13)
        if r.status_code == 200:
            return r
        time.sleep(2 * (k + 1))
    raise RuntimeError(f"{r.status_code} {url}")


def filings_8k(cik):
    sub = get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json").json()
    pages = [sub["filings"]["recent"]]
    for f in sub["filings"].get("files", []):
        if f["filingTo"] >= START:
            pages.append(get("https://data.sec.gov/submissions/" + f["name"]).json())
    out = []
    for p in pages:
        for form, d, acc, items in zip(p["form"], p["filingDate"], p["accessionNumber"], p["items"]):
            if form == "8-K" and "2.02" in items and START <= d <= END:
                out.append((d, acc))
    return sorted(set(out))


def exhibit_url(cik, acc):
    base = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc.replace('-', '')}/"
    soup = BeautifulSoup(get(base + acc + "-index.htm").text, "lxml")
    cands = []
    for tr in soup.select("table.tableFile tr"):
        td = [x.get_text(strip=True) for x in tr.find_all("td")]
        a = tr.find("a")
        if len(td) >= 4 and a and td[3].upper().startswith("EX-99") and a["href"].lower().endswith((".htm", ".html")):
            cands.append((td[3].upper(), "https://www.sec.gov" + a["href"].replace("/ix?doc=", "")))
    cands.sort()                      # EX-99.1 first
    return cands[0][1] if cands else None


if __name__ == "__main__":
    comp = pd.read_csv(DATA / "companies.csv"); rows = []
    for t, cik in zip(comp.ticker, comp.cik):
        fl = filings_8k(int(cik)); print(t, len(fl), "earnings 8-Ks", flush=True)
        for d, acc in fl:
            fn = RAW / f"{t}_{d}_{acc}.htm"
            if not fn.exists():
                url = exhibit_url(int(cik), acc)
                if not url:
                    print("  no exhibit", t, d, acc); continue
                fn.write_bytes(get(url).content); (RAW / (fn.name + ".url")).write_text(url)
            rows.append(dict(ticker=t, date=d, accession=acc, file=fn.name,
                             url=(RAW / (fn.name + ".url")).read_text()))
    pd.DataFrame(rows).to_csv(RAW / "filings.csv", index=False)
    print("DONE:", len(rows), "filings saved in", RAW)
