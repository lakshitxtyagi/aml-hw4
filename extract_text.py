"""Step 2 of building external.parquet: pull the business description out of each cached filing.

Annual reports: "Item 1. Business" (10-K) / "Item 4. Information on the Company" (20-F).
Registration statements: "Prospectus Summary" followed by the "Business" section.
Table-of-contents hits are skipped by taking, for each start marker, the occurrence with the
longest run of text before the next end marker.
"""
import gzip
import os
import re
import warnings

import pandas as pd
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

from fetch_edgar import CACHE, get

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
MAX_WORDS = 2500


def html_text(html):
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style"]):
        t.decompose()
    for t in soup.find_all(style=re.compile(r"display:\s*none", re.I)):
        t.decompose()  # inline-XBRL hidden header
    txt = soup.get_text(" ")
    txt = re.sub(r"[\u200b\u200c\u200d\u2060\ufeff\xad]", "", txt).replace("\xa0", " ")
    return re.sub(r"\s+", " ", txt)


TOC = re.compile(r"\bitem\s*\d+[a-c]?\b|\.{4,}|\bpage\b|\b(?:risk factors|use of proceeds|dividend policy|"
                 r"capitalization|dilution|underwriting|legal matters|experts)\b\s*\d+|\bnote\s+\d+\b", re.I)


def from_heading(txt, start, n_words, max_toc=2, window=600):
    """Text after the first heading match that is not followed by table-of-contents material."""
    for ms in re.finditer(start, txt, flags=re.I):
        if len(TOC.findall(txt[ms.end():ms.end() + window])) <= max_toc:
            return " ".join(txt[ms.end():ms.end() + 40 * n_words].split()[:n_words])
    return ""


ANNUAL = [r"i\s?tem\s*1\s*[\.:\-–—]?\s*b\s?usiness\b", r"i\s?tem\s*4\s*[\.:\-–—]?\s*information\s*on\s*the\s*company",
          r"\bbusiness\s+overview\b"]
# first self-description sentence ("We are a ...", "X is a leading ...") outside the cover pages
SELF = re.compile(r"(?:\.\s+|^)((?:we|[A-Z][\w&.,' -]{1,60}?)\s+(?:are|is)\s+(?:a|an|one of|the world)\b)")
REG = [r"prospectus\s+summary(?!\s*[—–-])", r"\bsummary\b(?=\s+(?:overview|our company|who we are|this summary))",
       r"\bbusiness\b(?=\s+(?:overview|our\s+(?:company|mission)|who\s+we\s+are))", r"\boverview\b"]


def clip(s, n):
    return " ".join(s.split()[:n])


def main():
    idx = pd.read_csv(os.path.join(CACHE, "filing_index.csv"))
    out = []
    for r in idx.itertuples():
        if not isinstance(r.url, str):
            out.append({"stock_id": r.stock_id, "text": "", "n_words": 0, "method": "missing"})
            continue
        txt = html_text(get(r.url))
        pats = ANNUAL if r.form in ("10-K", "20-F", "40-F") else REG
        body, method = "", "fallback"
        for k, pat in enumerate(pats):
            body = from_heading(txt, pat, MAX_WORDS)
            if len(body.split()) >= 300:
                method = f"heading{k}"
                break
        if len(body.split()) < 300:
            for ms in SELF.finditer(txt, len(txt) // 50):
                if len(TOC.findall(txt[ms.start(1):ms.start(1) + 600])) <= 2:
                    body = " ".join(txt[ms.start(1):ms.start(1) + 40 * MAX_WORDS].split()[:MAX_WORDS])
                    method = "self-description"
                    break
        if len(body.split()) < 300:  # fall back to the text after the cover page
            words = txt.split()
            body = " ".join(words[min(1500, len(words) // 10):][:MAX_WORDS])
        body = clip(body, MAX_WORDS)
        out.append({"stock_id": r.stock_id, "text": body, "n_words": len(body.split()), "method": method})
        print(r.stock_id, r.ticker, r.form, method, len(body.split()), "|", body[:140].replace("\n", " "), flush=True)
    pd.DataFrame(out).to_parquet(os.path.join(CACHE, "texts.parquet"), index=False)


if __name__ == "__main__":
    main()
