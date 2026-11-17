"""Predecessor -> successor links, to lengthen short return histories (D15 (6)).

A holding-company formation or a merger gives the company a new code, so its own history restarts:
3.1% of the 2005-01 -> 2025-09 member month-ends have returns on fewer than 90% of the days of their
5-year window (D8). Most are such successors, and their predecessors are in the price file.

LINKS is built by hand. Each link is checked here against
  - the prices: every predecessor stops trading before the successor's first day, at most
    MAX_GAP_DAYS calendar days earlier;
  - the Nikkei change history: the predecessor's deletion and the successor's addition near the
    event. A predecessor that was not a member then (e.g. Nippon Metal Industry in 2012) has no row.
PARTIAL marks successors that also absorbed a company not in the file (a non-member); their chained
history comes from the in-file part only. NOT_CHAINED lists the short histories left as they are.

Several predecessors are combined with weights equal to their average daily trading value (close x
volume) over the WEIGHT_DAYS trading days before each one's last day, a proxy for relative market
value (shares outstanding are not in the free data). All such events are before 2020, when the daily
volume exists.

The chained history of a successor is its own rows from its first trading day, and before that the
weighted predecessor rows, recursively (Mizuho FG <- Mizuho Holdings <- IBJ, DKB, Fuji Bank).
Daily simple returns and dividend returns are averaged with the weights (renormalised over the
predecessors that traded that day); trading values are summed. The return across the delisting gap
(predecessor's last day -> successor's first, about a week) stays missing: filling it would need the
exact exchange ratio. The successor's first own day has no return either, as in prices_daily.

Outputs (MEMBERSHIP_DIR):
    successors.csv           one row per (successor, predecessor), with the checks and the weight
    successor_returns.parquet  the chained rows before each successor's first day: code (the
                             successor), date, log_return, div_return, trading_value, source
    successors_report.txt    checks, the cases not chained, and the coverage gain

Run:  python pipeline/successors.py
"""
import numpy as np
import pandas as pd

from config import DATA_DIR, MEMBERSHIP_DIR

MAX_GAP_DAYS = 14
WEIGHT_DAYS = 250
NIKKEI_WINDOW = pd.Timedelta(days=21)
COVERAGE_MIN = 0.90          # a window with returns on fewer days than this counts as short
TEST_START, TEST_END = "2005-01-01", "2025-09-30"
WINDOW_YEARS = 5             # D8

# (successor, its first trading day, predecessors in the file, what happened)
LINKS = [
    ("8305", "2000-09-28", ["8302", "8311", "8317"], "Mizuho Holdings: joint holding company of IBJ, Dai-Ichi Kangyo Bank and Fuji Bank"),
    ("3893", "2001-03-29", ["3863"], "Nippon Unipac Holding (later Nippon Paper Group): Nippon Paper Industries + Daishowa Paper"),
    ("8306", "2001-04-02", ["8315", "8402"], "Mitsubishi Tokyo Financial Group: Bank of Tokyo-Mitsubishi + Mitsubishi Trust (+ Nippon Trust)"),
    ("8307", "2001-04-02", ["8320", "8321", "8407"], "UFJ Holdings: Sanwa Bank + Tokai Bank + Toyo Trust"),
    ("8308", "2001-12-11", ["8319"], "Daiwa Bank Holdings (Resona): Daiwa Bank + Kinki Osaka Bank + Nara Bank"),
    ("8766", "2002-04-01", ["8751"], "Millea Holdings (Tokio Marine): Tokio Marine & Fire + Nichido Fire"),
    ("5016", "2002-09-26", ["5014"], "Nippon Mining Holdings: Japan Energy + Nikko Materials"),
    ("5411", "2002-09-26", ["5403", "5404"], "JFE Holdings: Kawasaki Steel + NKK"),
    ("9205", "2002-10-01", ["9201"], "Japan Airlines System: Japan Airlines + Japan Air System"),
    ("8316", "2002-12-02", ["8318"], "Sumitomo Mitsui Financial Group: holding company of Sumitomo Mitsui Banking"),
    ("8411", "2003-03-12", ["8305"], "Mizuho Financial Group: Mizuho Holdings reorganised"),
    ("2768", "2003-04-01", ["8063"], "Nissho Iwai-Nichimen Holdings (Sojitz): Nissho Iwai + Nichimen"),
    ("2779", "2003-09-01", ["8231"], "Mitsukoshi (new): Mitsukoshi + four regional Mitsukoshi companies"),
    ("1721", "2003-09-29", ["1947"], "Comsys Holdings: Nippon Comsys + Sanwa Densetsu + Tokyo Tsuushin Network"),
    ("6674", "2004-04-01", ["6933"], "GS Yuasa: Yuasa + Japan Storage Battery"),
    ("3382", "2005-09-01", ["8183", "8264"], "Seven & i Holdings: Seven-Eleven Japan + Ito-Yokado + Denny's Japan"),
    ("4568", "2005-09-28", ["4501", "4505"], "Daiichi Sankyo: Sankyo + Daiichi Pharmaceutical"),
    ("4188", "2005-10-03", ["4010"], "Mitsubishi Chemical Holdings: Mitsubishi Chemical + Mitsubishi Pharma"),
    ("9412", "2007-04-02", ["4795"], "Sky Perfect JSAT Holdings: Sky Perfect Communications + JSAT"),
    ("3099", "2008-04-01", ["8238", "2779"], "Isetan Mitsukoshi Holdings: Isetan + Mitsukoshi"),
    ("8725", "2008-04-01", ["8752"], "Mitsui Sumitomo Insurance Group Holdings (MS&AD): holding company of Mitsui Sumitomo Insurance"),
    ("2269", "2009-04-01", ["2202", "2261"], "Meiji Holdings: Meiji Seika Kaisha + Meiji Dairies"),
    ("5020", "2010-04-01", ["5001", "5016"], "JX Holdings (ENEOS): Nippon Oil + Nippon Mining Holdings"),
    ("8630", "2010-04-01", ["8755"], "NKSJ Holdings (Sompo): Sompo Japan + Nipponkoa"),
    ("5413", "2012-10-01", ["5407", "5479"], "Nisshin Steel Holdings: Nisshin Steel + Nippon Metal Industry"),
    ("5703", "2012-10-01", ["5701"], "Nippon Light Metal Holdings: holding company of Nippon Light Metal"),
    ("3863", "2013-04-01", ["3893"], "Nippon Paper Industries (relisted under its old code): Nippon Paper Group merged into it"),
    ("3289", "2013-10-01", ["8815"], "Tokyu Fudosan Holdings: Tokyu Land + Tokyu Livable + Tokyu Community"),
    ("1333", "2014-04-01", ["1334"], "Maruha Nichiro: Maruha Nichiro Holdings merged into its subsidiary"),
    ("7186", "2016-04-01", ["8332"], "Concordia Financial Group: Bank of Yokohama + Higashi-Nippon Bank"),
    ("9147", "2022-01-04", ["9062"], "Nippon Express Holdings: holding company of Nippon Express"),
    ("5831", "2022-10-03", ["8355"], "Shizuoka Financial Group: holding company of Shizuoka Bank"),
]

# Successors that also absorbed a company not in the file (not a Nikkei 225 member since 1996).
PARTIAL = {
    "3893": "Daishowa Paper", "8306": "Nippon Trust Bank", "8308": "Kinki Osaka Bank, Nara Bank",
    "8766": "Nichido Fire", "5016": "Nikko Materials", "9205": "Japan Air System", "2768": "Nichimen",
    "2779": "regional Mitsukoshi companies", "1721": "Sanwa Densetsu, Tokyo Tsuushin Network",
    "6674": "Japan Storage Battery", "3382": "Denny's Japan", "4188": "Mitsubishi Pharma",
    "9412": "JSAT", "8630": "Nipponkoa", "3289": "Tokyu Livable, Tokyu Community",
    "7186": "Higashi-Nippon Bank",
}

# Short histories left as they are (no usable predecessor in the file).
NOT_CHAINED = {
    "8309": "Mitsui Trust Holdings (2002): its predecessor Chuo Mitsui Trust is in the file only to 2000-03",
    "8795": "T&D Holdings (2004): Taiyo Life and Daido Life were never members",
    "1605": "INPEX Holdings (2006): the larger predecessor, INPEX Corp, was never a member; Teikoku Oil alone is not representative",
    "3086": "J. Front Retailing (2007): Daimaru and Matsuzakaya were never members (since 1996)",
    "8354": "Fukuoka Financial Group (2007): Bank of Fukuoka was never a member",
    "4324": "Dentsu: IPO 2001", "8303": "Shinsei Bank: IPO 2004", "3436": "SUMCO: IPO 2005",
    "8729": "Sony Financial: IPO 2007", "8750": "Dai-ichi Life: demutualised, IPO 2010",
    "6098": "Recruit: IPO 2014", "6178": "Japan Post Holdings: IPO 2015",
    "9434": "SoftBank Corp: IPO 2018", "6526": "Socionext: IPO 2022",
    "4704": "Trend Micro: moved from JASDAQ in 2000 (earlier prices are not in the TSE daily reports)",
    "4689": "Yahoo Japan (LY): moved from JASDAQ in 2003",
    "4755": "Rakuten: moved from JASDAQ in 2013",
}


def load_prices():
    px = pd.read_parquet(DATA_DIR / "prices_daily.parquet",
                         columns=["date", "code", "name", "in_index", "close", "volume", "log_return",
                                  "dividend", "split"])
    px["date"] = pd.to_datetime(px["date"])
    px = px.sort_values(["code", "date"]).reset_index(drop=True)
    prev = px.groupby("code")["close"].shift(1)
    px["div_return"] = (px["split"].fillna(1) * px["dividend"].fillna(0) / prev).where(px["log_return"].notna())
    px["trading_value"] = px["close"] * px["volume"]
    return px


def link_table(px, changes):
    traded = px.dropna(subset=["close"])
    rows = []
    for succ, listed, preds, what in LINKS:
        listed = pd.Timestamp(listed)
        s = traded[traded["code"] == succ]
        first = s.loc[s["date"] >= listed, "date"].min()
        before = s[(s["date"] < listed) & (s["date"] >= listed - pd.Timedelta(days=30))]
        added = changes[(changes["added"] == succ) & (changes["date"] >= listed - pd.Timedelta(days=5))
                        & (changes["date"] <= listed + NIKKEI_WINDOW)]["date"].min()
        tv = {}
        for p in preds:
            g = traded[(traded["code"] == p) & (traded["date"] < listed)]
            tv[p] = g["trading_value"].tail(WEIGHT_DAYS).mean()
        total = sum(tv.values())
        for p in preds:
            g = traded[(traded["code"] == p) & (traded["date"] < listed)]
            last = g["date"].max()
            deleted = changes[(changes["deleted"] == p) & (changes["date"] >= listed - NIKKEI_WINDOW)
                              & (changes["date"] <= listed + pd.Timedelta(days=5))]["date"].min()
            ratio = (s.loc[s["date"] == first, "close"].iloc[0] / g["close"].iloc[-1]
                     if len(preds) == 1 else np.nan)
            rows.append({
                "successor": succ, "successor_name": s["name"].iloc[-1], "successor_first_day": first.date(),
                "predecessor": p, "predecessor_name": g["name"].iloc[-1], "predecessor_last_day": last.date(),
                "gap_days": (first - last).days, "weight": tv[p] / total if total > 0 else 1 / len(preds),
                "nikkei_deleted": deleted.date() if pd.notna(deleted) else None,
                "nikkei_added": added.date() if pd.notna(added) else None,
                "price_ratio": ratio, "partial": PARTIAL.get(succ, ""), "event": what,
                "ok_prices": bool(first == listed and len(before) == 0 and 0 < (first - last).days <= MAX_GAP_DAYS),
            })
    return pd.DataFrame(rows)


def own_rows(px, code, until=None, since=None):
    g = px[px["code"] == code]
    if since is not None:
        g = g[g["date"] >= since]
    if until is not None:
        g = g[g["date"] <= until]
    return g[["date", "log_return", "div_return", "trading_value"]].assign(source=code)


def chained(px, table, code, until=None):
    """Daily rows of `code` up to `until`, with its predecessors' rows before its first day.
    A link applies only if the history asked for reaches the successor's first day: a reused code
    (3863 before 2001 and from 2013) is then not chained to itself."""
    links = table[table["successor"] == code]
    if until is not None:
        links = links[pd.to_datetime(links["successor_first_day"]) <= until]
    if links.empty:
        return own_rows(px, code, until=until)
    first = pd.Timestamp(links["successor_first_day"].iloc[0])
    own = own_rows(px, code, until=until, since=first)
    parts = []
    for _, l in links.iterrows():
        h = chained(px, table, l["predecessor"], until=pd.Timestamp(l["predecessor_last_day"]))
        parts.append(h.assign(w=l["weight"]))
    pre = pd.concat(parts)
    if len(links) == 1:
        pre = pre.drop(columns="w")
    else:
        pre = pre.assign(simple=np.expm1(pre["log_return"]))

        def combine(d):
            k = d[d["log_return"].notna()]
            if k.empty:
                return pd.Series({"log_return": np.nan, "div_return": np.nan,
                                  "trading_value": d["trading_value"].sum(min_count=1)})
            w = k["w"] / k["w"].sum()
            return pd.Series({"log_return": np.log1p((w * k["simple"]).sum()),
                              "div_return": (w * k["div_return"].fillna(0)).sum(),
                              "trading_value": d["trading_value"].sum(min_count=1)})
        src = "+".join(links["predecessor"])
        pre = pre.groupby("date")[["log_return", "simple", "div_return", "trading_value", "w"]].apply(combine).reset_index().assign(source=src)
    return pd.concat([pre, own], ignore_index=True).sort_values("date")


def coverage(px, series):
    """Share of member month-ends whose 5-year window has returns on < COVERAGE_MIN of its days."""
    days = pd.DatetimeIndex(np.sort(px["date"].unique()))
    member = px[px["in_index"]]
    me = member.groupby(member["date"].dt.to_period("M"))["date"].max()
    me = me[(me >= TEST_START) & (me <= TEST_END)]
    mm = member[member["date"].isin(me.values)][["date", "code"]]
    short = []
    for code, g in mm.groupby("code"):
        r = series(code)
        have = pd.Series(1, index=pd.DatetimeIndex(r.loc[r["log_return"].notna(), "date"]))
        have = have[~have.index.duplicated()].reindex(days, fill_value=0).cumsum()
        for d in g["date"]:
            start = d - pd.DateOffset(years=WINDOW_YEARS)
            n_days = ((days > start) & (days <= d)).sum()
            i0 = days.searchsorted(start, side="right") - 1
            n = have.loc[d] - (have.iloc[i0] if i0 >= 0 else 0)
            short.append((code, d, n / n_days < COVERAGE_MIN))
    return pd.DataFrame(short, columns=["code", "date", "short"])


def main():
    px = load_prices()
    changes = pd.read_csv(MEMBERSHIP_DIR / "nk225_changes.csv", dtype=str)
    changes["date"] = pd.to_datetime(changes["date"])
    table = link_table(px, changes)
    table.to_csv(MEMBERSHIP_DIR / "successors.csv", index=False, float_format="%.6g")

    pre = []
    for succ in table["successor"].unique():
        h = chained(px, table, succ)
        first = pd.Timestamp(table.loc[table["successor"] == succ, "successor_first_day"].iloc[0])
        pre.append(h[h["date"] < first].assign(code=succ))
    pre = pd.concat(pre, ignore_index=True)[["code", "date", "log_return", "div_return", "trading_value", "source"]]
    pre.to_parquet(MEMBERSHIP_DIR / "successor_returns.parquet", index=False)

    before = coverage(px, lambda c: own_rows(px, c))
    after = coverage(px, lambda c: chained(px, table, c))
    left = after[after["short"]].groupby("code").size().rename("short_month_ends")

    t = table.copy()
    t["weight"] = t["weight"].round(3)
    t["price_ratio"] = t["price_ratio"].round(4)
    lines = [
        "Predecessor -> successor links (D15 (6)). One row per (successor, predecessor).",
        f"Checks: prices (successor's first day = the listed date, no trade in the 30 days before, the "
        f"predecessor stops 1-{MAX_GAP_DAYS} days earlier): {int(t.groupby('successor')['ok_prices'].all().sum())} "
        f"of {t['successor'].nunique()} successors pass.",
        f"Nikkei change history: successor added within {NIKKEI_WINDOW.days} days: "
        f"{t.groupby('successor')['nikkei_added'].first().notna().sum()} of {t['successor'].nunique()}; "
        f"predecessor deleted within {NIKKEI_WINDOW.days} days: {t['nikkei_deleted'].notna().sum()} of {len(t)} "
        "(the others were not members at the time).",
        "",
        t[["successor", "successor_name", "successor_first_day", "predecessor", "predecessor_name",
           "predecessor_last_day", "gap_days", "weight", "nikkei_deleted", "nikkei_added", "price_ratio",
           "ok_prices"]].to_string(index=False),
        "",
        "Partial (also absorbed a company not in the file; chained from the in-file part):",
        *[f"  {k} {v}" for k, v in PARTIAL.items()],
        "",
        "Not chained:",
        *[f"  {k} {v}" for k, v in NOT_CHAINED.items()],
        "",
        f"Member month-ends {TEST_START[:7]} -> {TEST_END[:7]} whose {WINDOW_YEARS}-year window has returns on "
        f"< {COVERAGE_MIN:.0%} of its days:",
        f"  own history only: {before['short'].sum():,} of {len(before):,} = {100 * before['short'].mean():.2f}%",
        f"  with chaining:    {after['short'].sum():,} of {len(after):,} = {100 * after['short'].mean():.2f}%",
        "  still short, by code: " + ", ".join(f"{c} ({n})" for c, n in left.sort_values(ascending=False).items()),
        f"  codes still short that are not in NOT_CHAINED: {sorted(set(left.index) - set(NOT_CHAINED))}",
        "",
        f"Chained rows written: {len(pre):,} (successor_returns.parquet).",
    ]
    (MEMBERSHIP_DIR / "successors_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
