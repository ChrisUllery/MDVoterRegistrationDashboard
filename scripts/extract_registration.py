from pathlib import Path
import re

import pandas as pd
import pdfplumber


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"

JURISDICTIONS = {
    "ALLEGANY": "Allegany County",
    "ANNE ARUNDEL": "Anne Arundel County",
    "BALTIMORE CITY": "Baltimore City",
    "BALTIMORE CO.": "Baltimore County",
    "CALVERT": "Calvert County",
    "CAROLINE": "Caroline County",
    "CARROLL": "Carroll County",
    "CECIL": "Cecil County",
    "CHARLES": "Charles County",
    "DORCHESTER": "Dorchester County",
    "FREDERICK": "Frederick County",
    "GARRETT": "Garrett County",
    "HARFORD": "Harford County",
    "HOWARD": "Howard County",
    "KENT": "Kent County",
    "MONTGOMERY": "Montgomery County",
    "PR. GEORGE'S": "Prince George's County",
    "QUEEN ANNE'S": "Queen Anne's County",
    "ST. MARY'S": "St. Mary's County",
    "SOMERSET": "Somerset County",
    "TALBOT": "Talbot County",
    "WASHINGTON": "Washington County",
    "WICOMICO": "Wicomico County",
    "WORCESTER": "Worcester County",
}


def number(value):
    return int(value.replace(",", ""))


def find_latest_report():
    reports = sorted(RAW_DIR.glob("MSR-*.pdf"))

    if not reports:
        raise FileNotFoundError(
            f"No MSR PDF files found in {RAW_DIR}"
        )

    return reports[-1]


def report_date_from_filename(path):
    match = re.search(r"MSR-(\d{4})_(\d{2})\.pdf$", path.name)

    if not match:
        raise ValueError(
            f"Expected filename like MSR-2026_07.pdf, got {path.name}"
        )

    return int(match.group(1)), int(match.group(2))


def extract_registration(pdf_path):
    rows = []
    statewide = None

    with pdfplumber.open(pdf_path) as pdf:
        text = pdf.pages[0].extract_text()

    if not text:
        raise RuntimeError("Could not extract text from page 1.")

    for line in text.splitlines():

        if line.startswith("TOTAL "):
            values = [number(x) for x in line.split()[1:]]

            if len(values) == 18:
                statewide = {
                    "dem": values[9],
                    "rep": values[10],
                    "grn": values[11],
                    "wcp": values[12],
                    "una": values[13],
                    "oth": values[14],
                    "total": values[15],
                }

            continue

        for raw_name, clean_name in JURISDICTIONS.items():

            prefix = raw_name + " "

            if not line.startswith(prefix):
                continue

            values = [
                number(x)
                for x in line[len(prefix):].split()
            ]

            if len(values) != 18:
                raise ValueError(
                    f"{raw_name}: expected 18 numeric fields, "
                    f"found {len(values)}\n{line}"
                )

            # PDF layout:
            # 0-1   activity fields
            # 2-8   party-affiliation changes
            # 9-15  total active registration
            # 16-17 confirmation/inactive fields

            dem = values[9]
            rep = values[10]
            grn = values[11]
            wcp = values[12]
            una = values[13]
            oth = values[14]
            total = values[15]

            calculated_total = dem + rep + grn + wcp + una + oth

            if calculated_total != total:
                raise ValueError(
                    f"{clean_name}: party totals sum to "
                    f"{calculated_total:,}, but PDF reports {total:,}"
                )

            rows.append(
                {
                    "jurisdiction": clean_name,
                    "sbe_name": raw_name,
                    "dem": dem,
                    "rep": rep,
                    "grn": grn,
                    "wcp": wcp,
                    "una": una,
                    "oth": oth,
                    "total": total,
                }
            )

            break

    return pd.DataFrame(rows), statewide


def main():
    pdf_path = find_latest_report()
    year, month = report_date_from_filename(pdf_path)

    print(f"Reading: {pdf_path.name}")

    df, statewide = extract_registration(pdf_path)

    if len(df) != 24:
        raise ValueError(
            f"Expected 24 Maryland jurisdictions, found {len(df)}"
        )

    jurisdiction_total = int(df["total"].sum())

    if statewide is None:
        raise ValueError("Could not find statewide TOTAL row.")

    if jurisdiction_total != statewide["total"]:
        raise ValueError(
            f"Jurisdictions sum to {jurisdiction_total:,}, "
            f"but statewide total is {statewide['total']:,}"
        )

    df.insert(0, "year", year)
    df.insert(1, "month", month)

    # Dashboard-ready calculated fields
    df["third_party_unaffiliated"] = (
        df["grn"] + df["wcp"] + df["una"] + df["oth"]
    )

    df["dem_pct"] = (df["dem"] / df["total"] * 100).round(2)
    df["rep_pct"] = (df["rep"] / df["total"] * 100).round(2)
    df["third_party_unaffiliated_pct"] = (
        df["third_party_unaffiliated"] / df["total"] * 100
    ).round(2)

    df["dr_margin"] = df["dem"] - df["rep"]
    df["dr_margin_pct"] = (df["dem_pct"] - df["rep_pct"]).round(2)

    output_path = (
        PROCESSED_DIR
        / f"registration_{year}_{month:02d}.csv"
    )

    df.to_csv(output_path, index=False)

    print()
    print(df.to_string(index=False))
    print()
    print(f"Jurisdictions: {len(df)}")
    print(f"Statewide active registration: {statewide['total']:,}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()


