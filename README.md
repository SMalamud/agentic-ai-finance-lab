# Agentic AI in Finance: hands-on lab

A 40-minute Jupyter notebook for the SFI Master Class "Agentic AI in Finance". Participants build an agent that explains a large stock price move: it looks up the price, checks whether the whole market moved, searches the company's press releases, and answers with sources. The last part shows a prompt-injection attack and the guardrail that stops it.

The notebook needs no API key and costs nothing to run.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SMalamud/agentic-ai-finance-lab/blob/main/Agentic_AI_Finance_Lab.ipynb)

## Running it

**In Google Colab (recommended for the class).** Click the badge above and run the cells from top to bottom. The first cell downloads `lab.py` and the data from this repository.

**On your own machine.** Clone the repository, run `pip install -r requirements.txt`, and open `Agentic_AI_Finance_Lab.ipynb`. The notebook then reads the local `data/` folder. It still needs internet access for the diagrams and, once, for the embedding model (about 90 MB).

## What is in the repository

| Path | Contents |
|---|---|
| `Agentic_AI_Finance_Lab.ipynb` | The notebook. Its four diagrams are loaded from `img/` in this repository. |
| `lab.py` | Helper code: data loading, charts, output formatting, and the recorded model. |
| `data/prices.csv` | Daily adjusted closing prices of the 30 Dow Jones Industrial Average stocks, 3 January 2023 to 30 October 2026, from Yahoo Finance. |
| `data/passages.csv` | 2,860 text passages from 485 earnings press releases (Form 8-K, Item 2.02, Exhibit 99.1) downloaded from SEC EDGAR, plus one planted document for the security exercise. |
| `data/embeddings.npy` | One 384-number vector per passage, from the open model `all-MiniLM-L6-v2`. |
| `data/recorded_brain.json` | The recorded model: its replies for ten questions, and the vectors of its search queries. |
| `img/` | The four diagrams as PNG files. |
| `build/` | The scripts that produced everything above. |

## About the recorded model

A live language model needs a paid API key, so the notebook replays replies that were written in advance. Claude, a model made by Anthropic, worked through the agent loop with the same tools and data, and each reply was written from the actual tool results. The loop, the tools and the data run live in the notebook. Only the model's choice of the next step and its final answer are replayed, which is why it can answer the ten questions on its menu and no others, and why it would repeat the same steps even if a tool returned something new. Its one alternative path is a recorded fallback for a step that is blocked or fails.

One run is staged. In the question about Nike, the model is scripted to obey an instruction hidden in a planted document (`PLANTED-1`), so that participants can watch a prompt injection succeed and then block it. The notebook says so at that point.

## Rebuilding the data

The scripts in `build/` run in order and need the packages in both `requirements.txt` and `build/requirements-build.txt`. Steps 1 and 2 need internet access. Step 2 needs a contact address, which the SEC asks every automated client to declare.

```
python3 build/01_prices.py
SEC_CONTACT="Your Name you@example.org" python3 build/02_edgar_download.py build/edgar_raw
python3 build/03_build_corpus.py build/edgar_raw
python3 build/04_embed.py
python3 build/05_record.py
python3 build/06_diagrams.py
python3 build/07_make_notebook.py
```

Changing the data can change what the tools return. The recorded replies in `build/05_record.py` quote figures and passage identifiers, so reread them against the new tool output after any rebuild.

## Sources and terms

SEC filings are public documents. Price data comes from Yahoo Finance through the `yfinance` package and is included for teaching only. Nothing in this repository is investment advice.
