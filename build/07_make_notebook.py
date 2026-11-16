"""Assemble Agentic_AI_Finance_Lab.ipynb from the cells below. Diagrams are embedded in the file."""
import base64
from pathlib import Path
import nbformat as nbf
import tools_src

ROOT = Path(__file__).resolve().parent.parent
REPO = "SMalamud/agentic-ai-finance-lab"
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


def img(name, alt):
    data = base64.b64encode((ROOT / "img" / name).read_bytes()).decode()
    return f"![{alt}](data:image/png;base64,{data})"


TURN = "> **Your turn.**"

# ---------------------------------------------------------------------------------------
md(f"""
# Agentic AI in Finance: build an analyst agent

**SFI Master Class, hands-on lab (about 40 minutes)**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/{REPO}/blob/main/Agentic_AI_Finance_Lab.ipynb)

A client calls: *"UnitedHealth lost a fifth of its value in one day. What happened?"*
An analyst would look up the price, check whether the whole market fell, read the company's announcement, and answer with sources.
In this lab you build an AI agent that does the same four things.

{img("roadmap.png", "Roadmap of the lab: prices, embeddings, the agent, guardrails")}

**How to use this notebook**

- Grey boxes are code. Click one and press **Shift + Enter** to run it. Run them from top to bottom.
- You do not need to write code. Where a box is marked **Your turn**, change a word or a number and run it again.
- Nothing here can break anything. If you get lost, use *Runtime → Restart and run all*.

**No API key and no cost.** The data is public (stock prices and press releases filed with the US SEC).
The language model's replies were recorded in advance. Part 3 explains how that works and what it means.
""")

md("## Setup\nThis downloads the data and a small helper file. It takes a few seconds.")
code(f'''
REPO = ("https://raw.githubusercontent.com/"
        "{REPO}/main")

import os, urllib.request
if not os.path.exists("lab.py"):
    urllib.request.urlretrieve(f"{{REPO}}/lab.py", "lab.py")

import pandas as pd
import lab
lab.setup(REPO)
''')

# ---------------------------------------------------------------------------------------
md("""
---
## Part 1. Prices: a tool that reads numbers  (5 min)

An agent can only do what its **tools** let it do. A tool is an ordinary function: it takes inputs, returns a result, and gives the same result every time.

We have daily closing prices for 30 large US companies (the members of the Dow Jones Industrial Average). Here is one of them.
""")
code('lab.plot_price("UNH")')
md("""
Now we write two tools. Read them once: each is a few lines.

- `biggest_moves` finds the days on which a stock moved most.
- `compare_to_market` puts a stock's move next to the average move of all 30 stocks. If everything fell, the reason is probably not the company.

The sentence in triple quotes is important. Later, **it is all the language model will know about the tool.**
""")
code(tools_src.PRICE_TOOLS)
md("Try them.")
code('biggest_moves("UNH", 2025)')
code('compare_to_market("UNH", "2025-04-17")')
code('lab.plot_price("UNH", mark=["2025-04-17"])')
md(f"""
{TURN} Pick another company and year in the box below. The 30 tickers are:
`AAPL AMGN AMZN AXP BA CAT CRM CSCO CVX DIS GS HD HON IBM JNJ JPM KO MCD MMM MRK MSFT NKE NVDA PG SHW TRV UNH V VZ WMT`
""")
code('biggest_moves("JPM", 2025)')
md("""
**What we have so far.** The tools tell us *when* a stock moved and *whether the market moved too*. They cannot tell us *why*. For that we need text.
""")

# ---------------------------------------------------------------------------------------
md(f"""
---
## Part 2. Embeddings: a tool that reads text  (10 min)

A computer cannot compare meanings, but it can compare numbers. An **embedding model** turns a piece of text into a list of numbers
(here, 384 of them) in such a way that texts with similar meaning get similar numbers.

{img("embeddings.png", "Two sentences go through an embedding model and become nearby points")}

The model we use is small, free and runs inside this notebook. See it at work: the table shows how similar each pair of sentences is
(1 = same meaning, 0 = unrelated).
""")
code('''
lab.compare([
    "The bank raised its dividend.",
    "The lender increased its payout to shareholders.",
    "Credit losses rose sharply this quarter.",
    "The weather in Zurich is sunny today.",
])
''')
md(f"""
Sentences 1 and 2 share almost no words, yet they score highest. Sentence 4 is unrelated to all the others.

{TURN} Replace one sentence with your own (for example in German or French) and run the box again.

### Our library

We have 2,860 short passages from the earnings press releases of the same 30 companies, 2023 to 2026.
Each passage has already been turned into its 384 numbers. Searching by meaning is now one line of arithmetic.

Compare the two ways of searching. First the classic way, by exact words:
""")
code('lab.keyword_search("demand for artificial intelligence chips")')
md("No passage contains that exact phrase. Now by meaning:")
code('lab.search("demand for artificial intelligence chips")')
md(f"""
The search found Nvidia passages about "AI factories" and "accelerated computing". Nobody told it that Nvidia makes AI chips.

{TURN} Try your own query. Two suggestions: `"cyber attack"` and `"job cuts"`.
The second one works badly: the top results are not about job cuts at all. Meaning search **always returns something**, relevant or not. Remember this for Part 3.
""")
code('lab.search("cyber attack", k=3)')
md("""
### The map of meaning

Each dot is one passage, placed so that passages with similar meaning are close. Colours show the sector, which the model was never told.
Hover over a dot to read it.
""")
code('lab.show_map()')
code('lab.show_map("credit losses and loan provisions")')
md("""
### The third tool

We wrap the search in a tool, with two optional filters: the company, and a year or a date.
""")
code(tools_src.SEARCH_TOOL)
code('''
results = search_news("lower earnings outlook, reasons for the revision",
                      ticker="UNH", when="2025-04-17")
lab.show_result(results)
''')
md("""
**What we have so far.** Three tools: two read numbers, one reads text. Each is dumb on its own.
You just combined them by hand to explain the UnitedHealth drop. Next, a model does the combining.
""")

# ---------------------------------------------------------------------------------------
md(f"""
---
## Part 3. The agent: a model that chooses the tools  (12 min)

An agent is a language model placed inside a loop:

{img("agent_loop.png", "The agent loop: the model picks a tool, code runs it, the model reads the result, and repeats")}

Two points matter for a bank:

1. **The model never touches the data.** It only writes which tool it wants and with which inputs. Ordinary code runs the tool.
2. **Every step is visible.** You can log, check and block each one.

Here is the menu of tools the model receives. It is built from the sentences in triple quotes that you saw earlier.
""")
code('''
TOOLS = {
    "biggest_moves": biggest_moves,
    "compare_to_market": compare_to_market,
    "search_news": search_news,
}
lab.describe_tools(TOOLS)
''')
md("""
### First, you are the model

Before we let a model do it, play its role. You get a question and the tool menu, and nothing else. Choose one move at a time.
""")
code('''
DISNEY = "What was Disney's biggest one-day move in 2024, and what explains it?"
game = lab.Game(DISNEY, TOOLS)
''')
md("The first move is filled in for you.")
code('game.act("biggest_moves", ticker="DIS", year=2024)')
md(f"{TURN} Choose your second and third moves: replace each `...` and run the box. A mistake only produces an error message, exactly as it would for a model.")
code('game.act("compare_to_market", ticker="DIS", date="...")')
code('game.act("search_news", query="...", ticker="DIS", when="...")')
code('game.answer("Type your answer for the client here.")')
md("""
### Now the model

**About the model in this lab.** A live language model costs money per call and needs an API key. So we recorded one instead:
Claude, a model made by Anthropic, worked through this same loop with these same tools and data, and we stored its replies.
`RecordedBrain` replays them.

- What is real: the loop, the tools, the data. Every tool call below runs live on your machine.
- What is replayed: the model's choice of the next move, and its final answer.
- The limits: it only knows the questions on its menu. And it cannot react to anything new: if you changed a tool, it would still replay the moves it made when it was recorded. The one exception is a move that is blocked or fails, for which it has a recorded fallback. A live model reads each result afresh.
""")
code('''
brain = lab.RecordedBrain()
brain.menu()
''')
md("""
This is the whole agent. Read it line by line: it is the loop from the diagram.
""")
code('''
def run_agent(question, brain, tools, guardrail=None, max_steps=8):
    history = [{"role": "user", "content": question}]
    lab.show_question(question)

    for step in range(1, max_steps + 1):
        # THINK: the model reads the history and picks its next move
        reply = brain(history)
        lab.show_reply(step, reply)

        # the model decided it has enough: we are done
        if "answer" in reply:
            return

        # ACT: our code runs the tool, unless a guardrail says no (Part 4)
        tool, args = reply["tool"], reply["args"]
        if guardrail and not guardrail(tool, args):
            result = "BLOCKED by the guardrail."
        else:
            try:
                result = tools[tool](**args)
            except Exception as error:          # the model reads errors too
                result = f"ERROR: {error!r}"

        # OBSERVE: the result goes into the history for the next round
        lab.show_result(result)
        history += [{"role": "assistant", "content": reply},
                    {"role": "tool", "content": result}]

    lab.show_result("STOPPED: the agent used all its steps without an answer.")
''')
md("Run it on the question you just answered yourself, and compare.")
code('run_agent(DISNEY, brain, TOOLS)')
md(f"""
{TURN} Change the number (0 to 8) to run other questions from the menu. Do not miss **3** and **6**: watch what the agent does when its sources do not contain the answer. Question 9 is kept for Part 4.
""")
code('run_agent(brain.questions[3], brain, TOOLS)')
code('run_agent(brain.questions[6], brain, TOOLS)')
md("""
**What to notice**

- No line of our code fixes the order of the steps. The model chose it when it was recorded, and chose differently for different questions (compare questions 0, 7 and 8).
- In question 8 the first search returned the wrong quarter. The model saw that in the result and searched again with narrower dates.
- In questions 3 and 6 the honest answer is "my sources do not say". A model without tools, asked the same thing, would be tempted to produce a fluent guess.
- Every answer that gives a reason cites passages you can open and check.
""")

# ---------------------------------------------------------------------------------------
md(f"""
---
## Part 4. Guardrails: limits on what the agent may do  (10 min)

So far the agent could only *read*. Real agents also *act*: they send messages, book orders, move money.
We give ours one action tool, `send_email`. It is simulated: nothing leaves this notebook, the message is only stored in a list.
""")
code('''
TOOLS_WITH_EMAIL = {**TOOLS, "send_email": lab.send_email}
lab.describe_tools(TOOLS_WITH_EMAIL)
''')
md("Now a harmless question about Nike. Watch the steps closely.")
code('''
NIKE = "What is Nike saying about inventory and tariffs in 2026?"
run_agent(NIKE, brain, TOOLS_WITH_EMAIL)
''')
code('lab.outbox')
md(f"""
### What just happened

The client's question went to an outside address, and the answer given to the client says nothing about it.

Nobody hacked the model or the code. One of the *documents* contained a sentence addressed to AI assistants, and the agent obeyed it.
This is called **prompt injection**. To a language model, a retrieved document and an instruction are both just text.

Two honest remarks:

- We planted that document ourselves (look at its label, `PLANTED-1`), and this run is **staged**: we scripted the model to fall for it. Well-trained models resist many such attempts, but not all of them, and you cannot audit which ones.
- That is why the protection must sit **outside the model**, in code that the model cannot talk its way around.

{img("guardrail.png", "A guardrail sits between the model and the tools")}

### A guardrail in five lines

The rule below lets every read-only tool through, and allows emails only to addresses inside the bank.
""")
code('''
def guardrail(tool, args):
    """Return True to allow the move, False to block it."""
    if tool == "send_email":
        return args["to"].endswith("@mybank.example")
    return True
''')
code('''
lab.outbox.clear()
run_agent(NIKE, brain, TOOLS_WITH_EMAIL, guardrail=guardrail)
''')
code('lab.outbox')
md(f"""
The outbox is empty. The move was blocked, the model was told, and it went on to answer from the genuine release and to flag the suspicious document.

{TURN} A second kind of guardrail asks a person. Remove the `#` in the last line of the box below and run it. The agent will stop and wait for your decision.
""")
code('''
def ask_human(tool, args):
    """Read-only tools pass. Any action needs a human yes."""
    if tool in ["biggest_moves", "compare_to_market", "search_news"]:
        return True
    print(f"The agent wants to call {tool} with {args}")
    return input("Allow? (y/n) ") == "y"

# run_agent(NIKE, brain, TOOLS_WITH_EMAIL, guardrail=ask_human)
''')
md("""
**What to take back to the bank**

- Give an agent the fewest tools it needs. An agent that can only read cannot send anything anywhere.
- Put the rules in code between the model and the tools. Do not rely on instructions to the model.
- Require a human decision for actions that are hard to undo.
- Log every step. You have seen that the loop makes this easy.
""")

# ---------------------------------------------------------------------------------------
md("""
---
## Wrap-up

In 40 minutes you built the three parts of every agent:

| Part | In this lab | In production |
|---|---|---|
| Tools | three small functions over prices and press releases | market data, document stores, core banking systems |
| Loop | about twenty lines of Python | a framework such as LangGraph, which adds persistence and monitoring around the same loop |
| Model | recorded replies | a live model behind an API or hosted inside the bank |

To use a live model, only `brain` changes: it becomes a function that sends `history` to the model and translates the model's reply into the same format. The tools, the loop and the guardrails stay as they are.

**Questions for the discussion**

1. In which of your processes does someone look things up in several systems and then write a summary?
2. Which tools would that agent need, and which of them are actions?
3. Where would you put the human decision?
""")

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}, "colab": {"provenance": [], "toc_visible": True}})
out = ROOT / "Agentic_AI_Finance_Lab.ipynb"
nbf.write(nb, out)
print(out.name, len(cells), "cells,", out.stat().st_size // 1024, "KB")
