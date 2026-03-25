# NBA Player Twitter Sentiment Analysis

A Python project that fetches tweets from NBA players' official Twitter accounts, runs BERT-based sentiment analysis on each tweet, and visualizes the correlation between a player's points per game (PPG) and the sentiment of their social media presence.

---

## What It Does

1. Logs into Twitter via `twikit` and fetches the most recent tweets for each player
2. Runs each tweet through a HuggingFace BERT sentiment analysis pipeline (POSITIVE / NEGATIVE)
3. Aggregates sentiment counts per player
4. Plots two scatter charts:
   - **PPG vs. Number of Positive Tweets**
   - **PPG vs. Number of Negative Tweets**

---

## Players Tracked (configurable)

| Player | Twitter Handle | PPG |
|---|---|---|
| LeBron James | @KingJames | 25.7 |
| Stephen Curry | @StephenCurry30 | 32.0 |
| Kevin Durant | @KDTrey5 | 22.0 |

Players, handles, and PPG values can all be updated in the `players` dictionary at the top of the script.

---

## Tech Stack

| Concern | Library |
|---|---|
| Twitter API client | `twikit` |
| Sentiment analysis | `transformers` (HuggingFace BERT) |
| Numerical aggregation | `numpy` |
| Visualization | `matplotlib` |
| Credential management | `python-dotenv` |

---

## Setup

### 1. Clone the repo

```bash
git clone <repo-url>
cd <repo-folder>
```

### 2. Install dependencies

```bash
pip install twikit transformers numpy matplotlib python-dotenv
```

### 3. Configure credentials

Create a `.env` file in the project root:

```
TWITTER_USERNAME=your_twitter_username
TWITTER_PASSWORD=your_twitter_password
```

> **Never commit your `.env` file.** Add it to `.gitignore`.

### 4. Run the script

```bash
python main.py
```

On first run, the script logs in and saves session cookies to `cookies.json` so subsequent runs don't require re-authentication.

---

## Output

- Console logs showing fetched tweet text and its sentiment label for each player
- Sentiment distribution summary per player
- Two matplotlib scatter plots displayed at the end of the run

---

## Project Structure

```
.
├── main.py          # Main script
├── .env             # Twitter credentials (not committed)
├── cookies.json     # Auto-generated session cookies (not committed)
└── README.md
```

---

## .gitignore Recommendation

```
.env
cookies.json
__pycache__/
```

---

## Notes

- The BERT model used is `distilbert-base-uncased-finetuned-sst-2-english` by default (HuggingFace default for `sentiment-analysis`), which outputs `POSITIVE` or `NEGATIVE` only — not `NEUTRAL`. The script accounts for this by initializing all three categories to `0`.
- `tweet_count` is set to `20` by default and can be adjusted in the script.
- Twitter rate limits and account restrictions may affect how many tweets are returned.
