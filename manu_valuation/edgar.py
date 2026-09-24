"""Manchester United plc financials from SEC EDGAR (XBRL). Free and structured.

Manchester United plc (NYSE: MANU, CIK 1549107) files a 20-F under IFRS every September,
in GBP. Two EDGAR endpoints are used:

* companyfacts API (data.sec.gov/api/xbrl/companyfacts): every *undimensioned* fact the
  company has ever tagged. Gives total revenue, employee benefits (wages), amortisation,
  depreciation, operating profit, borrowings and cash.
* the XBRL instance of each 20-F (www.sec.gov/Archives/...): needed for the revenue split,
  because matchday / broadcasting / commercial are tagged as *members of a dimension*
  (e.g. ProductsAndServicesAxis = BroadcastingMember), and companyfacts leaves those out.

The SEC asks for a User-Agent naming you and an e-mail address, and at most 10 requests
per second: https://www.sec.gov/os/accessing-edgar-data
"""
from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd
import requests

CIK = 1549107
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/"
ANNUAL_FORMS = {"20-F", "20-F/A", "10-K", "10-K/A"}

# Concept names to try, in order of preference (IFRS taxonomy first, then US GAAP).
DURATION_CONCEPTS = {
    "revenue": ["Revenue", "RevenueFromContractsWithCustomers", "Revenues"],
    "wages": ["EmployeeBenefitsExpense", "WagesAndSalaries"],
    "amortisation": ["AmortisationExpense", "AmortisationIntangibleAssetsOtherThanGoodwill"],
    "depreciation": ["DepreciationExpense", "DepreciationPropertyPlantAndEquipment",
                     "DepreciationAndImpairmentLossReversalOfImpairmentLossRecognisedInProfitOrLossPropertyPlantAndEquipment"],
    "operating_profit": ["ProfitLossFromOperatingActivities", "OperatingIncomeLoss"],
    "player_disposals": ["GainsLossesOnDisposalsOfIntangibleAssets",
                         "GainsLossesOnDisposalsOfIntangibleAssetsOtherThanGoodwill"],
}
INSTANT_CONCEPTS = {
    "borrowings": ["Borrowings", "LongtermBorrowings"],
    "cash": ["CashAndCashEquivalents", "CashAndCashEquivalentsAtCarryingValue"],
}
# Company-specific concepts are matched by keyword when no standard concept exists.
KEYWORD_FALLBACK = {"amortisation": re.compile(r"amorti[sz]ation.*(?:player|registration)", re.I)}
SEGMENTS = {"matchday": re.compile(r"matchday", re.I),
            "broadcasting": re.compile(r"broadcast", re.I),
            "commercial": re.compile(r"commercial", re.I)}
COMMERCIAL_PARTS = re.compile(r"sponsorship|retail|merchandis|licens", re.I)


class Edgar:
    def __init__(self, user_agent: str, session=None, pause: float = 0.15):
        if "@" not in user_agent:
            raise ValueError("SEC requires a User-Agent with a contact e-mail, e.g. 'Jane Doe jane@x.com'")
        self.s = session or requests.Session()
        self.s.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
        self.pause = pause

    def get(self, url: str):
        time.sleep(self.pause)
        r = self.s.get(url, timeout=60)
        r.raise_for_status()
        return r

    def company_facts(self, cik: int = CIK) -> dict:
        return self.get(FACTS_URL.format(cik=cik)).json()

    def annual_filings(self, cik: int = CIK) -> pd.DataFrame:
        recent = self.get(SUBMISSIONS_URL.format(cik=cik)).json()["filings"]["recent"]
        f = pd.DataFrame(recent)
        return f[f["form"].isin(ANNUAL_FORMS)][["accessionNumber", "form", "filingDate", "reportDate"]]

    def instance(self, accession: str, cik: int = CIK) -> bytes:
        """The XBRL instance document of one filing (the *_htm.xml file for inline XBRL)."""
        base = ARCHIVE_URL.format(cik=cik, acc=accession.replace("-", ""))
        items = self.get(base + "index.json").json()["directory"]["item"]
        names = [i["name"] for i in items]
        pick = [n for n in names if n.endswith("_htm.xml")] or [
            n for n in names if n.endswith(".xml") and not re.search(r"(_cal|_def|_lab|_pre|FilingSummary)", n)]
        if not pick:
            raise FileNotFoundError(f"no XBRL instance in {base}")
        return self.get(base + pick[0]).content


# ---------------------------------------------------------------- companyfacts


def facts_frame(facts: dict) -> pd.DataFrame:
    rows = []
    for taxonomy, concepts in facts.get("facts", {}).items():
        for concept, body in concepts.items():
            for unit, values in body.get("units", {}).items():
                for v in values:
                    rows.append(dict(taxonomy=taxonomy, concept=concept, unit=unit, value=v["val"],
                                     start=v.get("start"), end=v["end"], form=v.get("form"),
                                     filed=v.get("filed"), accn=v.get("accn")))
    df = pd.DataFrame(rows)
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c])
    return df


def _fiscal_year(end: pd.Series) -> pd.Series:
    """United's year ends 30 June; FY2024 = year to 30 June 2024."""
    return end.dt.year.where(end.dt.month >= 6, end.dt.year - 1)


def annual_values(df: pd.DataFrame, unit: str = "GBP") -> pd.DataFrame:
    """One value per (item, fiscal year), latest filing wins (restatements included)."""
    df = df[df["form"].isin(ANNUAL_FORMS) & (df["unit"] == unit)].copy()
    days = (df["end"] - df["start"]).dt.days
    out = {}
    for item, names in DURATION_CONCEPTS.items():
        cands = df[df["concept"].isin(names) & days.between(350, 380)]
        if cands.empty and item in KEYWORD_FALLBACK:
            cands = df[df["concept"].str.contains(KEYWORD_FALLBACK[item]) & days.between(350, 380)]
        out[item] = _latest(cands, names)
    for item, names in INSTANT_CONCEPTS.items():
        out[item] = _latest(df[df["concept"].isin(names) & df["start"].isna()], names)
    res = pd.DataFrame(out)
    res.index.name = "fy"
    return res / 1e6


def _latest(cands: pd.DataFrame, names: list) -> pd.Series:
    if cands.empty:
        return pd.Series(dtype=float)
    cands = cands.assign(rank=cands["concept"].map({n: i for i, n in enumerate(names)}).fillna(len(names)),
                         fy=_fiscal_year(cands["end"]))
    cands = cands.sort_values(["fy", "rank", "filed"], ascending=[True, True, False])
    return cands.groupby("fy")["value"].first().astype(float)


# ---------------------------------------------------------------- XBRL instance (segments)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].split(":")[-1]


def parse_instance(xml: bytes) -> pd.DataFrame:
    """All numeric facts in an XBRL instance with their period and explicit dimensions."""
    root = ET.fromstring(xml)
    contexts = {}
    for ctx in root.iter():
        if _local(ctx.tag) != "context":
            continue
        start = end = None
        dims = {}
        for el in ctx.iter():
            name = _local(el.tag)
            if name == "startDate":
                start = el.text.strip()
            elif name in ("endDate", "instant"):
                end = el.text.strip()
            elif name == "explicitMember":
                dims[_local(el.get("dimension", ""))] = _local(el.text.strip())
        contexts[ctx.get("id")] = (start, end, dims)
    rows = []
    for el in root.iter():
        ref = el.get("contextRef")
        if ref is None or ref not in contexts or el.text is None:
            continue
        try:
            value = float(el.text.strip())
        except ValueError:
            continue
        start, end, dims = contexts[ref]
        rows.append(dict(concept=_local(el.tag), value=value, start=start, end=end,
                         unit=el.get("unitRef"), dims=dims))
    df = pd.DataFrame(rows)
    if not df.empty:
        df["start"] = pd.to_datetime(df["start"])
        df["end"] = pd.to_datetime(df["end"])
    return df


def revenue_segments(facts: pd.DataFrame) -> pd.DataFrame:
    """Matchday / broadcasting / commercial revenue per fiscal year from instance facts."""
    rev = facts[facts["concept"].isin(DURATION_CONCEPTS["revenue"])
                & ((facts["end"] - facts["start"]).dt.days.between(350, 380))
                & (facts["dims"].map(len) == 1)].copy()
    if rev.empty:
        return pd.DataFrame(columns=list(SEGMENTS))
    rev["member"] = rev["dims"].map(lambda d: next(iter(d.values())))
    rev["fy"] = _fiscal_year(rev["end"])
    out = {}
    for seg, pat in SEGMENTS.items():
        m = rev[rev["member"].str.contains(pat)]
        out[seg] = m.groupby("fy")["value"].max()
    res = pd.DataFrame(out)
    parts = rev[rev["member"].str.contains(COMMERCIAL_PARTS)]
    if not parts.empty:  # commercial only tagged as sponsorship + retail etc.
        parts = parts.drop_duplicates(["fy", "member"]).groupby("fy")["value"].sum()
        res["commercial"] = res.get("commercial", pd.Series(dtype=float)).combine_first(parts)
    res.index.name = "fy"
    return res / 1e6


# ---------------------------------------------------------------- assembly


def build(client: Edgar, cik: int = CIK, max_filings: int = 8) -> pd.DataFrame:
    """Annual financials in GBP m, one row per fiscal year, from EDGAR only."""
    base = annual_values(facts_frame(client.company_facts(cik)))
    segs = []
    for acc in client.annual_filings(cik)["accessionNumber"].head(max_filings):
        try:
            segs.append(revenue_segments(parse_instance(client.instance(acc, cik))))
        except (requests.HTTPError, FileNotFoundError, ET.ParseError) as e:
            print(f"  skipped {acc}: {e}")
    if segs:
        s = pd.concat(segs)
        s = s[~s.index.duplicated(keep="first")]  # newest filing first = restated figures
        base = base.join(s, how="outer")
    base = base.sort_index()
    # Adjusted EBITDA as the club defines it excludes player-sale profits and exceptional items;
    # this approximation leaves exceptional items in.
    if {"operating_profit", "amortisation", "depreciation"} <= set(base):
        disp = base.get("player_disposals", pd.Series(0.0, index=base.index)).fillna(0)
        base["adj_ebitda"] = base["operating_profit"] + base["amortisation"] + base["depreciation"] - disp
    return base


COLUMNS = ["revenue", "matchday", "broadcasting", "commercial", "wages", "adj_ebitda",
           "amortisation", "depreciation"]


def merge_with_snapshot(edgar: pd.DataFrame, snapshot: pd.DataFrame) -> pd.DataFrame:
    """EDGAR values win; the snapshot fills whatever XBRL did not provide."""
    snap = snapshot.set_index("fy")
    e = edgar.reindex(columns=[c for c in COLUMNS + ["borrowings", "cash"] if c in edgar.columns])
    out = e.combine_first(snap[COLUMNS])
    complete = e.reindex(index=out.index, columns=COLUMNS).notna().all(axis=1)
    source = snap["source"].reindex(out.index).fillna("")
    source[out.index.isin(e.index)] = "EDGAR XBRL + snapshot"
    source[complete] = "EDGAR XBRL"
    return out.assign(source=source).reset_index()


def load_financials(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = {"fy", "revenue", "matchday", "broadcasting", "commercial", "wages", "adj_ebitda"} - set(df)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return df.sort_values("fy", ignore_index=True)


def shares_outstanding(instance_facts: pd.DataFrame) -> float | None:
    """dei:EntityCommonStockSharesOutstanding in millions, summed over share classes."""
    s = instance_facts[instance_facts["concept"] == "EntityCommonStockSharesOutstanding"]
    if s.empty:
        return None
    by_class = s[s["dims"].map(len) > 0]
    total = by_class["value"].sum() if not by_class.empty else s["value"].max()
    return float(total) / 1e6
