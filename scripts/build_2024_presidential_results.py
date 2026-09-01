from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

SOURCE_FILE = (
    ROOT
    / "data"
    / "raw"
    / "presidential_2024_by_jurisdiction.html"
)

OUTPUT_FILE = (
    ROOT
    / "data"
    / "reference"
    / "presidential_2024_jurisdiction_results.csv"
)


JURISDICTION_MAP = {
    "Allegany": "Allegany County",
    "Anne Arundel": "Anne Arundel County",
    "Baltimore City": "Baltimore City",
    "Baltimore County": "Baltimore County",
    "Calvert": "Calvert County",
    "Caroline": "Caroline County",
    "Carroll": "Carroll County",
    "Cecil": "Cecil County",
    "Charles": "Charles County",
    "Dorchester": "Dorchester County",
    "Frederick": "Frederick County",
    "Garrett": "Garrett County",
    "Harford": "Harford County",
    "Howard": "Howard County",
    "Kent": "Kent County",
    "Montgomery": "Montgomery County",
    "Prince George's": "Prince George's County",
    "Queen Anne's": "Queen Anne's County",
    "Saint Mary's": "St. Mary's County",
    "Somerset": "Somerset County",
    "Talbot": "Talbot County",
    "Washington": "Washington County",
    "Wicomico": "Wicomico County",
    "Worcester": "Worcester County",
}


def numeric(series):
    values = (
        series.astype(str)
        .str.extract(r"^\s*([\d,]+)", expand=False)
        .str.replace(",", "", regex=False)
    )

    return pd.to_numeric(
        values,
        errors="coerce",
    ).fillna(0).astype(int)


def find_column(columns, text):
    matches = [
        column
        for column in columns
        if text.lower() in str(column).lower()
    ]

    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one column containing "
            f"{text!r}; found {matches}"
        )

    return matches[0]


def main():
    print(f"Reading: {SOURCE_FILE.name}")

    tables = pd.read_html(SOURCE_FILE)

    if len(tables) != 6:
        raise ValueError(
            f"Expected 6 result tables, found {len(tables)}"
        )

    # All six tables should contain the same
    # 24 jurisdictions plus the statewide Totals row.
    expected_jurisdictions = set(
        tables[0]["Jurisdiction"].astype(str)
    )

    for index, table in enumerate(tables):
        jurisdictions = set(
            table["Jurisdiction"].astype(str)
        )

        if jurisdictions != expected_jurisdictions:
            raise ValueError(
                f"Jurisdiction rows differ in table {index}."
            )

    main_table = tables[0].copy()

    harris_col = find_column(
        main_table.columns,
        "Kamala D. Harris",
    )

    trump_col = find_column(
        main_table.columns,
        "Donald J. Trump",
    )

    oliver_col = find_column(
        main_table.columns,
        "Chase Oliver",
    )

    stein_col = find_column(
        main_table.columns,
        "Jill Ellen Stein",
    )

    kennedy_col = find_column(
        main_table.columns,
        "Robert F. Kennedy",
    )

    # Start with all five named ballot candidates.
    results = pd.DataFrame(
        {
            "sbe_jurisdiction": (
                main_table["Jurisdiction"].astype(str)
            ),
            "harris_votes": numeric(
                main_table[harris_col]
            ),
            "trump_votes": numeric(
                main_table[trump_col]
            ),
            "oliver_votes": numeric(
                main_table[oliver_col]
            ),
            "stein_votes": numeric(
                main_table[stein_col]
            ),
            "kennedy_votes": numeric(
                main_table[kennedy_col]
            ),
        }
    )

    # Sum every write-in column across all six
    # tables, including the write-in column that
    # appears on table 0.
    write_in_votes = pd.Series(
        0,
        index=results.index,
        dtype="int64",
    )

    for table_index, table in enumerate(tables):
        for column in table.columns:
            if column == "Jurisdiction":
                continue

            column_text = str(column).lower()

            if (
                "write in" in column_text
                or "write-in" in column_text
                or "other write-ins" in column_text
            ):
                write_in_votes += numeric(
                    table[column]
                )

    results["write_in_votes"] = write_in_votes

    # Separate statewide totals before building
    # the 24-jurisdiction output.
    statewide_row = results[
        results["sbe_jurisdiction"] == "Totals"
    ].copy()

    if len(statewide_row) != 1:
        raise ValueError(
            "Expected exactly one statewide Totals row."
        )

    results = results[
        results["sbe_jurisdiction"] != "Totals"
    ].copy()

    results["jurisdiction"] = (
        results["sbe_jurisdiction"]
        .map(JURISDICTION_MAP)
    )

    if results["jurisdiction"].isna().any():
        missing = results.loc[
            results["jurisdiction"].isna(),
            "sbe_jurisdiction",
        ].tolist()

        raise ValueError(
            f"Unmapped jurisdictions: {missing}"
        )

    if len(results) != 24:
        raise ValueError(
            f"Expected 24 jurisdictions, found {len(results)}"
        )

    vote_columns = [
        "harris_votes",
        "trump_votes",
        "oliver_votes",
        "stein_votes",
        "kennedy_votes",
        "write_in_votes",
    ]

    # Validate jurisdiction totals against the
    # statewide Totals row supplied by SBE.
    for column in vote_columns:
        jurisdiction_sum = int(
            results[column].sum()
        )

        official_total = int(
            statewide_row[column].iloc[0]
        )

        if jurisdiction_sum != official_total:
            raise ValueError(
                f"{column}: jurisdiction sum "
                f"{jurisdiction_sum:,} does not match "
                f"SBE statewide total {official_total:,}"
            )

    results["total_presidential_votes"] = (
        results[vote_columns].sum(axis=1)
    )

    results["winner"] = results.apply(
        lambda row:
            "Trump"
            if row["trump_votes"] > row["harris_votes"]
            else "Harris",
        axis=1,
    )

    results["winner_votes"] = results[
        [
            "trump_votes",
            "harris_votes",
        ]
    ].max(axis=1)

    results["margin_votes"] = (
        results["trump_votes"]
        - results["harris_votes"]
    ).abs()

    results["winner_pct"] = (
        results["winner_votes"]
        / results["total_presidential_votes"]
        * 100
    ).round(2)

    results["margin_pct"] = (
        results["margin_votes"]
        / results["total_presidential_votes"]
        * 100
    ).round(2)

    results = results[
        [
            "jurisdiction",
            "harris_votes",
            "trump_votes",
            "oliver_votes",
            "stein_votes",
            "kennedy_votes",
            "write_in_votes",
            "total_presidential_votes",
            "winner",
            "winner_votes",
            "margin_votes",
            "winner_pct",
            "margin_pct",
        ]
    ].sort_values("jurisdiction")

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(f"Saved: {OUTPUT_FILE}")
    print(f"Jurisdictions: {len(results)}")
    print(
        "Total presidential votes: "
        f"{results['total_presidential_votes'].sum():,}"
    )

    print()
    print(
        results[
            [
                "jurisdiction",
                "winner",
                "margin_votes",
                "margin_pct",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()

