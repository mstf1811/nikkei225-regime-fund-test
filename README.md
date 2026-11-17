# Does a regime-aware long-short fund beat cash in Japan?

Code, frozen specification and complete run log for the paper
*Does a regime-aware long-short fund beat cash in Japan? A pre-specified test on Nikkei 225 stocks, 2005–2025*.

## What is here

| Path | Contents |
|---|---|
| `pipeline/` | Data pipeline (membership, JPX parsing, Yahoo corporate actions, adjusted total returns) and the backtest (`backtest.py`, `bt_*.py`, `evaluation.py`) |
| `analysis/` | Pre-test design checks, signal study, figures and report scripts |
| `results/FREEZE.json` | Freeze record: date, decisions covered, SHA-256 of the backtest code, the 23 planned runs |
| `results/trials.csv` | Every test-period run, with its code hash and whether it matches the freeze |
| `results/report/` | Evaluation output (metrics, H2 and H3 tables, summary) |
| `data/membership/` | Nikkei 225 membership parsed from Nikkei Inc.'s component-change history, and the successor chaining table |

## Verify the freeze

The code hash is computed over the backtest modules listed in `pipeline/backtest.py` (`CODE`):

```
python -c "import sys; sys.path.insert(0, 'pipeline'); import backtest; print(backtest.code_hash())"
```

It should print `484b657a55c94636a3a39834f245cf94b9324fcd0d9ff1092d64047cffe0b151`, the value in `results/FREEZE.json` and in every row of `results/trials.csv`.
Line-ending conversion is disabled in `.gitattributes` so that a clone has the same bytes.

## Rebuild the data

Raw files are not redistributed. Download them to the paths in `pipeline/config.py`:

- **JPX daily reports** (stock prices, past issues), 2000-01 to 2025-09, as the monthly zip files: https://www.jpx.co.jp/markets/statistics-equities/daily/03.html → `JPX Data/`
- **JPX monthly quotation tables, detailed version** (`st_YYYYMM.pdf`), 2020-01 onward: https://www.jpx.co.jp/markets/statistics-equities/price/index.html → `JPX Monthly/`
- **Nikkei 225 component-change history**, downloaded by `pipeline/membership.py`
- **Yahoo Finance** corporate actions, downloaded by `pipeline/yahoo.py`
- **Bank of Japan call rate** (FM01 STRDCLUCON) and `^N225`, downloaded by `pipeline/market.py`
- **Kenneth French Japan factors**, downloaded by `pipeline/ff_japan.py`

Then, in order:

```
python pipeline/membership.py
python pipeline/yahoo.py
python pipeline/jpx.py
python pipeline/jpx_monthly.py
python pipeline/build_prices.py
python pipeline/market.py
python pipeline/ff_japan.py
python pipeline/price_limits.py
python pipeline/successors.py
```

## Run the backtest

```
pip install -r requirements.txt
python pipeline/backtest.py --runs all --jobs 6
python pipeline/bt_report.py
```

Each run takes about an hour. `backtest.py` refuses test-period runs before a freeze and logs every run to
`analysis/output/backtest/trials.csv`.

## Data licences

JPX, Nikkei, Yahoo Finance, Bank of Japan and Kenneth French data remain subject to their providers' terms.
Only code, the parsed index membership and the run outputs are published here.
The code is released under the MIT License (see `LICENSE`).
