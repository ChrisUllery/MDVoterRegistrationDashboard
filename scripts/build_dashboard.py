from pathlib import Path
import calendar
import json
import re
import unicodedata
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import requests


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
REFERENCE_DIR = ROOT / "data" / "reference"
RAW_DIR = ROOT / "data" / "raw"

PRESIDENTIAL_RESULTS_FILE = (
    REFERENCE_DIR
    / "presidential_2024_jurisdiction_results.csv"
)

SHAPE_ZIP = RAW_DIR / "cb_2025_us_county_500k.zip"
SHAPE_DIR = RAW_DIR / "cb_2025_us_county_500k"

OUTPUT_DIR = ROOT / "docs"
OUTPUT_FILE = OUTPUT_DIR / "index.html"


# ---------------------------------------------------------
# Sources
# ---------------------------------------------------------

COUNTY_SHAPE_URL = (
    "https://www2.census.gov/geo/tiger/"
    "GENZ2025/shp/"
    "cb_2025_us_county_500k.zip"
)

MD_VOTER_STATS_URL = (
    "https://elections.maryland.gov/"
    "voter_registration/stats.html"
)


# ---------------------------------------------------------
# Map configuration -- mirrors PA dashboard
# ---------------------------------------------------------

RED = "#b2182b"
PURPLE = "#7b3294"
BLUE = "#2166ac"

COLOR_SCALE = [
    [0.0, RED],
    [0.5, PURPLE],
    [1.0, BLUE],
]

ASINH_SCALE = 5.0


# ---------------------------------------------------------
# File/date helpers
# ---------------------------------------------------------

def find_latest_stats_file():
    files = sorted(PROCESSED_DIR.glob("registration_*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No registration CSV files found in {PROCESSED_DIR}"
        )

    return files[-1]


def source_date_display(stats):
    years = stats["year"].dropna().unique()
    months = stats["month"].dropna().unique()

    if len(years) != 1 or len(months) != 1:
        raise ValueError(
            "Expected one year and one month in registration data."
        )

    year = int(years[0])
    month = int(months[0])

    return f"{calendar.month_name[month]} {year}"


# ---------------------------------------------------------
# Jurisdiction-name normalization
# ---------------------------------------------------------

def normalize_jurisdiction_name(value):
    text = str(value).strip()

    text = unicodedata.normalize(
        "NFKD",
        text,
    )

    text = text.casefold()

    text = re.sub(
        r"\bcounty\b",
        "",
        text,
    )

    text = re.sub(
        r"[^a-z0-9]",
        "",
        text,
    )

    return text


# ---------------------------------------------------------
# Download Census boundaries
# ---------------------------------------------------------

def download_shapes():
    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if SHAPE_ZIP.exists():
        print(
            f"Boundary file already downloaded: {SHAPE_ZIP}"
        )
        return

    print("Downloading Census county boundaries...")

    response = requests.get(
        COUNTY_SHAPE_URL,
        timeout=120,
    )

    response.raise_for_status()
    SHAPE_ZIP.write_bytes(response.content)

    print(f"Saved: {SHAPE_ZIP}")


# ---------------------------------------------------------
# Extract Census boundaries
# ---------------------------------------------------------

def extract_shapes():
    SHAPE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    existing = list(SHAPE_DIR.glob("*.shp"))

    if existing:
        print("Boundary shapefile already extracted.")
        return

    print("Extracting Census county boundaries...")

    with zipfile.ZipFile(SHAPE_ZIP) as archive:
        archive.extractall(SHAPE_DIR)

    print(f"Extracted to: {SHAPE_DIR}")


# ---------------------------------------------------------
# Load voter-registration data
# ---------------------------------------------------------

def load_stats():
    stats_file = find_latest_stats_file()

    print("\nLoading Maryland voter-registration data...")
    print(f"Registration file: {stats_file.name}")

    stats = pd.read_csv(stats_file)

    required_columns = {
        "year",
        "month",
        "jurisdiction",
        "dem",
        "rep",
        "third_party_unaffiliated",
        "total",
        "dem_pct",
        "rep_pct",
        "third_party_unaffiliated_pct",
        "dr_margin",
        "dr_margin_pct",
    }

    missing = required_columns - set(stats.columns)

    if missing:
        raise ValueError(
            "Registration file is missing required columns: "
            + ", ".join(sorted(missing))
        )

    stats["jurisdiction_key"] = (
        stats["jurisdiction"]
        .apply(normalize_jurisdiction_name)
    )

    if len(stats) != 24:
        raise ValueError(
            f"Expected 24 Maryland jurisdictions, found {len(stats)}."
        )

    print(f"Registration jurisdictions: {len(stats)}")

    return stats


# ---------------------------------------------------------
# Load 2024 presidential jurisdiction results
# ---------------------------------------------------------

def load_presidential_results():
    print("\nLoading 2024 presidential jurisdiction results...")

    if not PRESIDENTIAL_RESULTS_FILE.exists():
        raise FileNotFoundError(
            "Presidential results file not found: "
            f"{PRESIDENTIAL_RESULTS_FILE}"
        )

    presidential = pd.read_csv(PRESIDENTIAL_RESULTS_FILE)

    required_columns = {
        "jurisdiction",
        "harris_votes",
        "trump_votes",
        "total_presidential_votes",
        "winner",
        "winner_votes",
        "margin_votes",
        "winner_pct",
        "margin_pct",
    }

    missing = required_columns - set(presidential.columns)

    if missing:
        raise ValueError(
            "Presidential results file is missing required columns: "
            + ", ".join(sorted(missing))
        )

    presidential["jurisdiction_key"] = (
        presidential["jurisdiction"]
        .apply(normalize_jurisdiction_name)
    )

    duplicates = presidential[
        presidential["jurisdiction_key"].duplicated(keep=False)
    ]

    if not duplicates.empty:
        raise ValueError(
            "Duplicate jurisdiction keys found in presidential results."
        )

    if len(presidential) != 24:
        raise ValueError(
            "Expected 24 presidential-result jurisdictions, "
            f"found {len(presidential)}."
        )

    print(
        "2024 presidential-result jurisdictions: "
        f"{len(presidential)}"
    )

    return presidential


# ---------------------------------------------------------
# Load Maryland geography
# ---------------------------------------------------------

def load_jurisdictions():
    shape_files = list(SHAPE_DIR.glob("*.shp"))

    if not shape_files:
        raise FileNotFoundError(
            "No Census shapefile was found."
        )

    counties = gpd.read_file(shape_files[0])

    # Maryland FIPS = 24. This includes 23 counties + Baltimore City.
    md = counties[
        counties["STATEFP"] == "24"
    ].copy()

    md = md.to_crs(epsg=4326)

    md["jurisdiction_key"] = (
        md["NAMELSAD"]
        .apply(normalize_jurisdiction_name)
    )

    print(
        f"Census Maryland jurisdiction shapes: {len(md)}"
    )

    return md


# ---------------------------------------------------------
# Validate geography/data matching
# ---------------------------------------------------------

def validate_names(stats, jurisdictions):
    print("\nChecking jurisdiction names...")

    stats_duplicates = stats[
        stats["jurisdiction_key"].duplicated(keep=False)
    ]

    census_duplicates = jurisdictions[
        jurisdictions["jurisdiction_key"].duplicated(keep=False)
    ]

    if not stats_duplicates.empty:
        raise ValueError(
            "Duplicate jurisdiction keys found in registration data."
        )

    if not census_duplicates.empty:
        raise ValueError(
            "Duplicate jurisdiction keys found in Census data."
        )

    stats_keys = set(stats["jurisdiction_key"])
    census_keys = set(jurisdictions["jurisdiction_key"])

    missing_from_map = stats_keys - census_keys
    missing_from_stats = census_keys - stats_keys

    if missing_from_map:
        print("\nRegistration jurisdictions not found in Census map:")
        for key in sorted(missing_from_map):
            name = stats.loc[
                stats["jurisdiction_key"] == key,
                "jurisdiction",
            ].iloc[0]
            print(f"  - {name}")

    if missing_from_stats:
        print("\nCensus jurisdictions not found in registration data:")
        for key in sorted(missing_from_stats):
            name = jurisdictions.loc[
                jurisdictions["jurisdiction_key"] == key,
                "NAME",
            ].iloc[0]
            print(f"  - {name}")

    if missing_from_map or missing_from_stats:
        raise ValueError(
            "Jurisdiction-name matching failed. Dashboard was not built."
        )

    print("All 24 jurisdictions matched.")


# ---------------------------------------------------------
# Merge registration data with geography
# ---------------------------------------------------------

def merge_data(jurisdictions, stats):
    merged = jurisdictions.merge(
        stats,
        on="jurisdiction_key",
        how="left",
        validate="one_to_one",
    )

    merged["display_jurisdiction"] = merged["jurisdiction"]

    print(f"Merged jurisdiction rows: {len(merged)}")

    return merged


# ---------------------------------------------------------
# Merge 2024 presidential results
# ---------------------------------------------------------

def merge_presidential_results(merged, presidential):
    print(
        "\nMatching 2024 presidential results "
        "to dashboard jurisdictions..."
    )

    dashboard_keys = set(merged["jurisdiction_key"])
    presidential_keys = set(presidential["jurisdiction_key"])

    missing_results = dashboard_keys - presidential_keys
    extra_results = presidential_keys - dashboard_keys

    if missing_results:
        print("\nDashboard jurisdictions missing presidential results:")
        for key in sorted(missing_results):
            name = merged.loc[
                merged["jurisdiction_key"] == key,
                "jurisdiction",
            ].iloc[0]
            print(f"  - {name}")

    if extra_results:
        print("\nPresidential-result jurisdictions not in dashboard:")
        for key in sorted(extra_results):
            name = presidential.loc[
                presidential["jurisdiction_key"] == key,
                "jurisdiction",
            ].iloc[0]
            print(f"  - {name}")

    if missing_results or extra_results:
        raise ValueError(
            "Presidential jurisdiction matching failed. "
            "Dashboard was not built."
        )

    presidential_for_merge = presidential.drop(
        columns=["jurisdiction"]
    )

    merged = merged.merge(
        presidential_for_merge,
        on="jurisdiction_key",
        how="left",
        validate="one_to_one",
    )

    if merged["winner"].isna().any():
        raise ValueError(
            "One or more jurisdictions are missing presidential "
            "results after merge."
        )

    print(
        "All 24 jurisdictions matched to 2024 presidential results."
    )

    return merged


# ---------------------------------------------------------
# Calculate map color metric -- same logic as PA dashboard
# ---------------------------------------------------------

def calculate_map_metric(merged):
    print("\nCalculating map color metric...")

    # Third-party/unaffiliated share pulls the D-R margin toward purple.
    merged["adjusted_margin"] = (
        merged["dr_margin_pct"]
        * (
            1
            - (
                merged["third_party_unaffiliated_pct"]
                / 100
            )
        )
    )

    max_abs_margin = merged["adjusted_margin"].abs().max()

    if max_abs_margin == 0:
        merged["color_value"] = 0.0
    else:
        max_transformed = np.arcsinh(
            max_abs_margin / ASINH_SCALE
        )

        merged["color_value"] = (
            np.arcsinh(
                merged["adjusted_margin"]
                / ASINH_SCALE
            )
            / max_transformed
        )

    print(
        f"Maximum adjusted margin: {max_abs_margin:.2f}"
    )
    print(f"Asinh scale: {ASINH_SCALE:g}")

    return merged


# ---------------------------------------------------------
# Reader-friendly margin labels
# ---------------------------------------------------------

def add_margin_labels(merged):
    def describe_margin(row):
        margin_count = int(row["dr_margin"])
        margin_pct = float(row["dr_margin_pct"])

        if margin_count > 0:
            return (
                "Democratic registration edge: "
                f"{margin_count:,} voters "
                f"({abs(margin_pct):.2f} points)"
            )

        if margin_count < 0:
            return (
                "Republican registration edge: "
                f"{abs(margin_count):,} voters "
                f"({abs(margin_pct):.2f} points)"
            )

        return "Democratic and Republican registration is even"

    merged["margin_label"] = merged.apply(
        describe_margin,
        axis=1,
    )

    return merged


# ---------------------------------------------------------
# Statewide summary
# ---------------------------------------------------------

def calculate_statewide_stats(stats):
    total = int(stats["total"].sum())
    dem = int(stats["dem"].sum())
    rep = int(stats["rep"].sum())
    third = int(stats["third_party_unaffiliated"].sum())

    dem_pct = dem / total * 100
    rep_pct = rep / total * 100
    third_pct = third / total * 100

    margin = dem - rep
    margin_pct = dem_pct - rep_pct

    if margin > 0:
        margin_party = "Democratic"
        margin_count = margin
    elif margin < 0:
        margin_party = "Republican"
        margin_count = abs(margin)
    else:
        margin_party = "Even"
        margin_count = 0

    return {
        "total": total,
        "dem": dem,
        "rep": rep,
        "third": third,
        "dem_pct": dem_pct,
        "rep_pct": rep_pct,
        "third_pct": third_pct,
        "margin_party": margin_party,
        "margin_count": margin_count,
        "margin_pct": abs(margin_pct),
    }


# ---------------------------------------------------------
# Build Plotly jurisdiction map -- PA dashboard style
# ---------------------------------------------------------

def build_map(merged):
    print("\nBuilding Maryland jurisdiction map...")

    geojson = merged.__geo_interface__

    fig = px.choropleth(
        merged,
        geojson=geojson,
        locations="GEOID",
        featureidkey="properties.GEOID",
        color="color_value",
        range_color=(-1, 1),
        color_continuous_scale=COLOR_SCALE,
        custom_data=[
            "display_jurisdiction",
            "dem",
            "dem_pct",
            "rep",
            "rep_pct",
            "third_party_unaffiliated",
            "third_party_unaffiliated_pct",
            "total",
            "margin_label",
        ],
    )

    fig.update_traces(
        marker_line_color="white",
        marker_line_width=0.7,
        hovertemplate=(
            "<b>%{customdata[0]}</b>"
            "<br><br>"
            "Democratic: "
            "%{customdata[1]:,} "
            "(%{customdata[2]:.2f}%)"
            "<br>"
            "Republican: "
            "%{customdata[3]:,} "
            "(%{customdata[4]:.2f}%)"
            "<br>"
            "Third-party/unaffiliated: "
            "%{customdata[5]:,} "
            "(%{customdata[6]:.2f}%)"
            "<br>"
            "Total active registration: "
            "%{customdata[7]:,}"
            "<br><br>"
            "%{customdata[8]}"
            "<extra></extra>"
        ),
    )

    fig.update_geos(
        fitbounds="locations",
        visible=False,
    )

    fig.update_layout(
        height=500,
        dragmode=False,
        margin=dict(
            l=0,
            r=0,
            t=10,
            b=20,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False,
    )

    return fig


# ---------------------------------------------------------
# Neighbor lookup
# ---------------------------------------------------------

def build_neighbors(jurisdiction_stats):
    neighbors_by_jurisdiction = {}

    for idx, row in jurisdiction_stats.iterrows():
        name = str(row["jurisdiction"])
        neighbors = []

        for other_idx, other in jurisdiction_stats.iterrows():
            if idx == other_idx:
                continue

            shared_boundary = (
                row.geometry.boundary
                .intersection(other.geometry.boundary)
            )

            if (
                not shared_boundary.is_empty
                and shared_boundary.length > 1e-9
            ):
                neighbors.append(str(other["jurisdiction"]))

        neighbors_by_jurisdiction[name] = sorted(neighbors)

    return neighbors_by_jurisdiction


# ---------------------------------------------------------
# Build complete dashboard HTML -- mirrors PA structure
# ---------------------------------------------------------

def build_dashboard(fig, statewide, source_date, jurisdiction_stats):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    map_html = pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=True,
        config={
            "responsive": True,
            "displaylogo": False,
            "displayModeBar": False,
        },
    )

    region_data = {
        "Statewide": {
            "display": "Statewide",
            "total": int(statewide["total"]),
            "dem": int(statewide["dem"]),
            "dem_pct": float(statewide["dem_pct"]),
            "rep": int(statewide["rep"]),
            "rep_pct": float(statewide["rep_pct"]),
            "third": int(statewide["third"]),
            "third_pct": float(statewide["third_pct"]),
            "margin_party": statewide["margin_party"],
            "margin_count": int(statewide["margin_count"]),
            "margin_pct": float(statewide["margin_pct"]),
            "signed_margin_pct": float(
                statewide["dem_pct"] - statewide["rep_pct"]
            ),
        }
    }

    for _, row in jurisdiction_stats.sort_values("jurisdiction").iterrows():
        name = str(row["jurisdiction"])
        margin = int(row["dr_margin"])

        if margin > 0:
            margin_party = "Democratic"
        elif margin < 0:
            margin_party = "Republican"
        else:
            margin_party = "Even"

        region_data[name] = {
            "display": name,
            "total": int(row["total"]),
            "dem": int(row["dem"]),
            "dem_pct": float(row["dem_pct"]),
            "rep": int(row["rep"]),
            "rep_pct": float(row["rep_pct"]),
            "third": int(row["third_party_unaffiliated"]),
            "third_pct": float(row["third_party_unaffiliated_pct"]),
            "margin_party": margin_party,
            "margin_count": abs(margin),
            "margin_pct": abs(float(row["dr_margin_pct"])),
            "signed_margin_pct": float(row["dr_margin_pct"]),
            "presidential_winner": str(row["winner"]),
            "presidential_winner_votes": int(row["winner_votes"]),
            "presidential_winner_pct": float(row["winner_pct"]),
            "presidential_margin_votes": int(row["margin_votes"]),
            "presidential_margin_pct": float(row["margin_pct"]),
            "trump_votes": int(row["trump_votes"]),
            "harris_votes": int(row["harris_votes"]),
            "total_presidential_votes": int(
                row["total_presidential_votes"]
            ),
        }

    neighbors_by_jurisdiction = build_neighbors(jurisdiction_stats)

    for name, neighbors in neighbors_by_jurisdiction.items():
        if name in region_data:
            region_data[name]["neighbors"] = neighbors

    region_data_json = json.dumps(
        region_data,
        ensure_ascii=False,
    )

    selector_options = "\n".join(
        f'<option value="{name}">{data["display"]}</option>'
        for name, data in region_data.items()
    )

    selector_css = """
    .region-selector {
        max-width: 1040px;
        margin: 18px auto;
        display: flex;
        align-items: center;
        gap: 10px;
        flex-wrap: wrap;
    }

    .region-selector label {
        font-weight: 700;
    }

    .region-selector select {
        min-height: 44px;
        min-width: 220px;
        padding: 8px 36px 8px 12px;
        border: 1px solid #c8c8c8;
        border-radius: 8px;
        background: #fff;
        font: inherit;
        font-size: 16px;
    }
    """

    selector_script = """
<script>
(() => {
    const regionData = __REGION_DATA__;
    const selector = document.getElementById("region-select");
    const cards = document.querySelectorAll(".summary-card");

    if (!selector || cards.length < 5) return;

    const numberFormat = new Intl.NumberFormat("en-US");

    function updateCards() {
        const data = regionData[selector.value];
        if (!data) return;

        const [registered, dem, rep, third, edge] = cards;

        registered.querySelector(".summary-value").textContent =
            numberFormat.format(data.total);
        registered.querySelector(".summary-detail").textContent =
            data.display;

        dem.querySelector(".summary-value").textContent =
            numberFormat.format(data.dem);
        dem.querySelector(".summary-detail").textContent =
            `${data.dem_pct.toFixed(2)}% of voters`;

        rep.querySelector(".summary-value").textContent =
            numberFormat.format(data.rep);
        rep.querySelector(".summary-detail").textContent =
            `${data.rep_pct.toFixed(2)}% of voters`;

        third.querySelector(".summary-value").textContent =
            numberFormat.format(data.third);
        third.querySelector(".summary-detail").textContent =
            `${data.third_pct.toFixed(2)}% of voters`;

        edge.querySelector(".summary-label").textContent =
            data.display === "Statewide"
                ? "Statewide registration edge"
                : `${data.display} registration edge`;

        edge.querySelector(".summary-value").textContent =
            data.margin_party;

        edge.querySelector(".summary-detail").textContent =
            `${numberFormat.format(data.margin_count)} voters · ${data.margin_pct.toFixed(2)} points`;
    }

    selector.addEventListener("change", updateCards);
})();
</script>
""".replace("__REGION_DATA__", region_data_json)

    analysis_css = """
    .analysis-panel {
        max-width: 1040px;
        margin: 18px auto 0;
        padding: 20px;
        background: #ffffff;
        border: 1px solid #d9e0e7;
        border-radius: 12px;
    }

    .analysis-heading-row {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 16px;
        flex-wrap: wrap;
        margin-bottom: 12px;
    }

    .analysis-heading-row h2 {
        margin: 0;
        font-size: 1.3rem;
    }

    .analysis-summary {
        margin: 0 0 18px;
        line-height: 1.6;
    }

    .analysis-subhead {
        margin: 20px 0 10px;
        font-size: 1.05rem;
    }

    .analysis-highlights {
        margin: 8px 0 0;
        padding-left: 22px;
        line-height: 1.6;
    }

    .neighbor-intro {
        line-height: 1.6;
        margin: 0 0 12px;
    }

    .neighbor-note {
        color: #5d6976;
        font-size: 0.82rem;
        margin: 8px 0 0;
    }

    .neighbor-table-wrap {
        width: 100%;
        overflow-x: auto;
    }

    .neighbor-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.88rem;
        white-space: nowrap;
    }

    .neighbor-table th,
    .neighbor-table td {
        padding: 9px 10px;
        border-bottom: 1px solid #e0e5ea;
        text-align: right;
    }

    .neighbor-table th:first-child,
    .neighbor-table td:first-child {
        text-align: left;
    }

    .neighbor-table th {
        color: #5d6976;
        font-size: 0.78rem;
    }

    .neighbor-table .selected-row {
        font-weight: 700;
        background: #f3f6f9;
    }

    .copy-analysis {
        min-height: 44px;
        padding: 9px 15px;
        border: 1px solid #aeb8c2;
        border-radius: 8px;
        background: #ffffff;
        font: inherit;
        font-weight: 600;
        cursor: pointer;
    }

    .copy-analysis:hover {
        background: #f4f6f8;
    }

    .copy-status {
        min-height: 1em;
        margin: 10px 0 0;
        color: #5d6976;
        font-size: 0.82rem;
    }
    """

    analysis_script = """
<script>
(() => {
    const regionData = __REGION_DATA__;
    const selector = document.getElementById("region-select");
    const title = document.getElementById("analysis-title");
    const summary = document.getElementById("analysis-summary");
    const comparison = document.getElementById("analysis-comparison");
    const neighborsBox = document.getElementById("analysis-neighbors");
    const copyButton = document.getElementById("copy-analysis");
    const copyStatus = document.getElementById("copy-status");

    if (!selector || !title || !summary || !comparison ||
        !neighborsBox || !copyButton) {
        return;
    }

    const numberFormat = new Intl.NumberFormat("en-US");
    const jurisdictionEntries = Object.entries(regionData).filter(
        ([name]) => name !== "Statewide"
    );

    let copyText = "";

    function escapeHtml(value) {
        return String(value)
            .replaceAll("&", "&amp;")
            .replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;")
            .replaceAll('"', "&quot;")
            .replaceAll("'", "&#039;");
    }

    function ordinal(number) {
        const n = Number(number);
        const mod100 = n % 100;

        if (mod100 >= 11 && mod100 <= 13) return `${n}th`;

        switch (n % 10) {
            case 1: return `${n}st`;
            case 2: return `${n}nd`;
            case 3: return `${n}rd`;
            default: return `${n}th`;
        }
    }

    function rankDescending(metric, value) {
        return 1 + jurisdictionEntries.filter(
            ([, data]) => Number(data[metric]) > Number(value)
        ).length;
    }

    function partySubject(party) {
        if (party === "Democratic") return "Democrats";
        if (party === "Republican") return "Republicans";
        return null;
    }

    function edgeSentence(data, statewide=false) {
        if (Number(data.margin_count) === 0) {
            return statewide
                ? "Democratic and Republican registration is even statewide."
                : "Democratic and Republican registration is even.";
        }

        const subject = partySubject(data.margin_party);
        const location = statewide ? " statewide" : "";

        return (
            `${subject} hold a registration edge${location} ` +
            `of ${numberFormat.format(data.margin_count)} voters, ` +
            `or ${Number(data.margin_pct).toFixed(2)} percentage points.`
        );
    }

    function shortEdge(data) {
        if (Number(data.margin_count) === 0) return "Even";

        const letter = data.margin_party === "Democratic" ? "D" : "R";
        return `${letter} +${Number(data.margin_pct).toFixed(2)}`;
    }

    function presidentialWinnerName(winner) {
        if (winner === "Trump") return "Donald Trump";
        if (winner === "Harris") return "Kamala Harris";
        return String(winner);
    }

    function shortPresidentialResult(data) {
        return (
            `${data.presidential_winner} +` +
            `${Number(data.presidential_margin_pct).toFixed(2)}`
        );
    }

    function maxBy(metric) {
        return jurisdictionEntries.reduce(
            (best, current) =>
                Number(current[1][metric]) > Number(best[1][metric])
                    ? current
                    : best
        );
    }

    function closestMargin() {
        return jurisdictionEntries.reduce(
            (best, current) =>
                Math.abs(Number(current[1].signed_margin_pct)) <
                Math.abs(Number(best[1].signed_margin_pct))
                    ? current
                    : best
        );
    }

    function renderStatewide(data) {
        title.textContent = "Statewide analysis";

        const text =
            `Maryland has ${numberFormat.format(data.total)} active registered voters. ` +
            `Democrats account for ${numberFormat.format(data.dem)} voters ` +
            `(${Number(data.dem_pct).toFixed(2)}%), Republicans account for ` +
            `${numberFormat.format(data.rep)} (${Number(data.rep_pct).toFixed(2)}%), ` +
            `and ${numberFormat.format(data.third)} voters ` +
            `(${Number(data.third_pct).toFixed(2)}%) are registered with a third party ` +
            `or are unaffiliated. ${edgeSentence(data, true)}`;

        summary.textContent = text;

        const largest = maxBy("total");
        const demHigh = maxBy("dem_pct");
        const repHigh = maxBy("rep_pct");
        const thirdHigh = maxBy("third_pct");
        const closest = closestMargin();

        comparison.innerHTML = `
            <h3 class="analysis-subhead">Jurisdiction highlights</h3>
            <ul class="analysis-highlights">
                <li><strong>Largest registered electorate:</strong>
                    ${escapeHtml(largest[1].display)}
                    (${numberFormat.format(largest[1].total)} voters)</li>
                <li><strong>Highest Democratic registration share:</strong>
                    ${escapeHtml(demHigh[1].display)}
                    (${Number(demHigh[1].dem_pct).toFixed(2)}%)</li>
                <li><strong>Highest Republican registration share:</strong>
                    ${escapeHtml(repHigh[1].display)}
                    (${Number(repHigh[1].rep_pct).toFixed(2)}%)</li>
                <li><strong>Highest third-party/unaffiliated share:</strong>
                    ${escapeHtml(thirdHigh[1].display)}
                    (${Number(thirdHigh[1].third_pct).toFixed(2)}%)</li>
                <li><strong>Closest Democratic-Republican split:</strong>
                    ${escapeHtml(closest[1].display)}
                    (${shortEdge(closest[1])} points)</li>
            </ul>
        `;

        neighborsBox.innerHTML = "";

        copyText = `${text}\n\nJurisdiction highlights:\n` +
            `Largest registered electorate: ${largest[1].display} ` +
            `(${numberFormat.format(largest[1].total)} voters)\n` +
            `Highest Democratic registration share: ${demHigh[1].display} ` +
            `(${Number(demHigh[1].dem_pct).toFixed(2)}%)\n` +
            `Highest Republican registration share: ${repHigh[1].display} ` +
            `(${Number(repHigh[1].rep_pct).toFixed(2)}%)\n` +
            `Highest third-party/unaffiliated share: ${thirdHigh[1].display} ` +
            `(${Number(thirdHigh[1].third_pct).toFixed(2)}%)\n` +
            `Closest Democratic-Republican split: ${closest[1].display} ` +
            `(${shortEdge(closest[1])} points)`;
    }

    function renderJurisdiction(name, data) {
        title.textContent = `${data.display} analysis`;

        const presidentialWinner =
            presidentialWinnerName(data.presidential_winner);

        const text =
            `${data.display} has ${numberFormat.format(data.total)} active registered voters. ` +
            `Democrats account for ${numberFormat.format(data.dem)} voters ` +
            `(${Number(data.dem_pct).toFixed(2)}%), Republicans account for ` +
            `${numberFormat.format(data.rep)} (${Number(data.rep_pct).toFixed(2)}%), ` +
            `and ${numberFormat.format(data.third)} voters ` +
            `(${Number(data.third_pct).toFixed(2)}%) are registered with a third party ` +
            `or are unaffiliated. ${edgeSentence(data)} ` +
            `${presidentialWinner} won ${data.display} in the 2024 presidential ` +
            `election with ${numberFormat.format(data.presidential_winner_votes)} votes, ` +
            `${Number(data.presidential_winner_pct).toFixed(2).replace(/\.?0+$/, "")}% ` +
            `of the ${numberFormat.format(data.total_presidential_votes)} votes cast ` +
            `in the race.`;

        summary.textContent = text;

        const totalRank = rankDescending("total", data.total);
        const demRank = rankDescending("dem_pct", data.dem_pct);
        const repRank = rankDescending("rep_pct", data.rep_pct);
        const thirdRank = rankDescending("third_pct", data.third_pct);
        const balanceRank = rankDescending(
            "signed_margin_pct",
            data.signed_margin_pct
        );

        comparison.innerHTML = `
            <h3 class="analysis-subhead">How this jurisdiction compares statewide</h3>
            <ul class="analysis-highlights">
                <li><strong>Registered voters:</strong>
                    ${ordinal(totalRank)}-largest electorate in Maryland</li>
                <li><strong>Democratic registration:</strong>
                    ${ordinal(demRank)}-highest percentage among Maryland's 24 jurisdictions</li>
                <li><strong>Republican registration:</strong>
                    ${ordinal(repRank)}-highest percentage among Maryland's 24 jurisdictions</li>
                <li><strong>Third-party/unaffiliated registration:</strong>
                    ${ordinal(thirdRank)}-highest percentage among Maryland's 24 jurisdictions</li>
                <li><strong>Registration margin:</strong>
                    ${ordinal(balanceRank)}-most Democratic among Maryland's 24 jurisdictions</li>
            </ul>
        `;

        const neighborNames = Array.isArray(data.neighbors)
            ? data.neighbors
            : [];

        const neighbors = neighborNames
            .map(neighborName => [neighborName, regionData[neighborName]])
            .filter(([, neighborData]) => Boolean(neighborData));

        if (!neighbors.length) {
            neighborsBox.innerHTML = `
                <h3 class="analysis-subhead">Maryland neighbors</h3>
                <p class="neighbor-intro">
                    No neighboring Maryland jurisdictions were identified.
                </p>
            `;

            copyText = `${text}\n\nStatewide rankings:\n` +
                `• Registered voters: ${ordinal(totalRank)}-largest electorate in Maryland\n` +
                `• Democratic share: ${ordinal(demRank)}-highest among Maryland's 24 jurisdictions\n` +
                `• Republican share: ${ordinal(repRank)}-highest among Maryland's 24 jurisdictions\n` +
                `• Third-party/unaffiliated share: ${ordinal(thirdRank)}-highest among Maryland's 24 jurisdictions\n` +
                `• Registration margin: ${ordinal(balanceRank)}-most Democratic among Maryland's 24 jurisdictions`;
            return;
        }

        const selectedMargin = Number(data.signed_margin_pct);

        const moreDemocraticThan = neighbors.filter(
            ([, neighbor]) => selectedMargin > Number(neighbor.signed_margin_pct)
        ).length;

        const moreRepublicanThan = neighbors.filter(
            ([, neighbor]) => selectedMargin < Number(neighbor.signed_margin_pct)
        ).length;

        const rows = [[name, data], ...neighbors].map(
            ([jurisdictionName, jurisdiction]) => {
                const selected = jurisdictionName === name;
                return `
                    <tr class="${selected ? "selected-row" : ""}">
                        <td>${escapeHtml(jurisdiction.display)}</td>
                        <td>${numberFormat.format(jurisdiction.total)}</td>
                        <td>${Number(jurisdiction.dem_pct).toFixed(2)}%</td>
                        <td>${Number(jurisdiction.rep_pct).toFixed(2)}%</td>
                        <td>${Number(jurisdiction.third_pct).toFixed(2)}%</td>
                        <td>${shortEdge(jurisdiction)}</td>
                        <td>${shortPresidentialResult(jurisdiction)}</td>
                    </tr>
                `;
            }
        ).join("");

        const neighborDisplays = neighbors.map(([, n]) => n.display);

        function naturalList(values) {
            if (values.length === 0) return "";
            if (values.length === 1) return values[0];

            if (values.length === 2) {
                return `${values[0]} and ${values[1]}`;
            }

            return (
                `${values.slice(0, -1).join(", ")}, and ` +
                `${values.at(-1)}`
            );
        }

        function jurisdictionList(values) {
            if (values.length === 0) return "";

            const allCounties = values.every(
                value => value.endsWith(" County")
            );

            if (!allCounties) {
                return naturalList(values);
            }

            const countyNames = values.map(
                value => value.slice(0, -7)
            );

            if (countyNames.length === 1) {
                return `${countyNames[0]} County`;
            }

            if (countyNames.length === 2) {
                return (
                    `${countyNames[0]} and ` +
                    `${countyNames[1]} counties`
                );
            }

            return (
                `${countyNames.slice(0, -1).join(", ")}, and ` +
                `${countyNames.at(-1)} counties`
            );
        }

        const neighborList = naturalList(
            neighborDisplays
        );

        const democraticNeighbors = neighbors.filter(
            ([, neighbor]) =>
                neighbor.margin_party === "Democratic"
        );

        const republicanNeighbors = neighbors.filter(
            ([, neighbor]) =>
                neighbor.margin_party === "Republican"
        );

        const evenNeighbors = neighbors.filter(
            ([, neighbor]) =>
                neighbor.margin_party === "Even"
        );

        function registrationGroupSentence(
            entries,
            party
        ) {
            const names = entries.map(
                ([, jurisdiction]) =>
                    jurisdiction.display
            );

            const verb =
                entries.length === 1 ? "has" : "have";

            const advantage =
                entries.length === 1
                    ? "advantage"
                    : "advantages";

            const article =
                entries.length === 1
                    ? "a "
                    : "";

            return (
                `${jurisdictionList(names)} ${verb} ` +
                `${article}${party} registration ${advantage}`
            );
        }

        let registrationObservation = "";

        if (
            democraticNeighbors.length > 0 &&
            republicanNeighbors.length > 0
        ) {
            const demText =
                registrationGroupSentence(
                    democraticNeighbors,
                    "Democratic"
                );

            const repText =
                registrationGroupSentence(
                    republicanNeighbors,
                    "Republican"
                );

            registrationObservation =
                `By voter registration, its neighbors are ` +
                `politically mixed: ${demText}, while ` +
                `${repText}.`;

        } else if (
            democraticNeighbors.length ===
            neighbors.length
        ) {
            registrationObservation =
                `By voter registration, all of its ` +
                `Maryland neighbors have Democratic ` +
                `registration advantages.`;

        } else if (
            republicanNeighbors.length ===
            neighbors.length
        ) {
            registrationObservation =
                `By voter registration, all of its ` +
                `Maryland neighbors have Republican ` +
                `registration advantages.`;

        } else if (evenNeighbors.length > 0) {
            const pieces = [];

            if (democraticNeighbors.length) {
                pieces.push(
                    registrationGroupSentence(
                        democraticNeighbors,
                        "Democratic"
                    )
                );
            }

            if (republicanNeighbors.length) {
                pieces.push(
                    registrationGroupSentence(
                        republicanNeighbors,
                        "Republican"
                    )
                );
            }

            const evenNames = evenNeighbors.map(
                ([, jurisdiction]) =>
                    jurisdiction.display
            );

            pieces.push(
                `${jurisdictionList(evenNames)} ` +
                `${evenNeighbors.length === 1 ? "has" : "have"} ` +
                `an even Democratic-Republican split`
            );

            registrationObservation =
                `By voter registration, its neighboring ` +
                `jurisdictions differ: ` +
                `${naturalList(pieces)}.`;
        }

        const closestNeighbor = neighbors.reduce(
            (best, current) => {
                const bestDifference = Math.abs(
                    Number(best[1].signed_margin_pct) -
                    selectedMargin
                );

                const currentDifference = Math.abs(
                    Number(current[1].signed_margin_pct) -
                    selectedMargin
                );

                return currentDifference < bestDifference
                    ? current
                    : best;
            }
        );

        const closestData = closestNeighbor[1];

        const closestObservation =
            `Among those neighbors, ${closestData.display} ` +
            `is closest to ${data.display} in ` +
            `Democratic-Republican registration balance.`;

        const localJurisdictions = [
            [name, data],
            ...neighbors
        ];

        const harrisJurisdictions =
            localJurisdictions
                .filter(
                    ([, jurisdiction]) =>
                        jurisdiction.presidential_winner ===
                        "Harris"
                )
                .map(
                    ([, jurisdiction]) =>
                        jurisdiction.display
                );

        const trumpJurisdictions =
            localJurisdictions
                .filter(
                    ([, jurisdiction]) =>
                        jurisdiction.presidential_winner ===
                        "Trump"
                )
                .map(
                    ([, jurisdiction]) =>
                        jurisdiction.display
                );

        let presidentialObservation = "";

        if (
            harrisJurisdictions.length > 0 &&
            trumpJurisdictions.length > 0
        ) {
            presidentialObservation =
                `Kamala Harris won ` +
                `${jurisdictionList(harrisJurisdictions)} ` +
                `in the 2024 presidential election; ` +
                `Donald Trump won ` +
                `${jurisdictionList(trumpJurisdictions)}.`;

        } else if (harrisJurisdictions.length > 0) {
            presidentialObservation =
                `Kamala Harris won ` +
                `${jurisdictionList(harrisJurisdictions)} ` +
                `in the 2024 presidential election.`;

        } else if (trumpJurisdictions.length > 0) {
            presidentialObservation =
                `Donald Trump won ` +
                `${jurisdictionList(trumpJurisdictions)} ` +
                `in the 2024 presidential election.`;
        }

        const neighborNarrative = [
            `${data.display} borders ${neighborList}.`,
            registrationObservation,
            closestObservation,
            presidentialObservation
        ]
            .filter(Boolean)
            .join(" ");
        neighborsBox.innerHTML = `
            <h3 class="analysis-subhead">Compared with neighboring jurisdictions</h3>
            <p class="neighbor-intro">${escapeHtml(neighborNarrative)}</p>
            <div class="neighbor-table-wrap">
                <table class="neighbor-table">
                    <thead>
                        <tr>
                            <th>Jurisdiction</th>
                            <th>Registered</th>
                            <th>Dem.</th>
                            <th>Rep.</th>
                            <th>Third/unaff.</th>
                            <th>D-R edge</th>
                            <th>2024 president</th>
                        </tr>
                    </thead>
                    <tbody>${rows}</tbody>
                </table>
            </div>
            <p class="neighbor-note">
                Maryland neighbors are determined from the same U.S. Census
                boundaries used for the map. Out-of-state jurisdictions are not included.
            </p>
        `;

        copyText = `${text}\n\nStatewide rankings:\n` +
            `• Registered voters: ${ordinal(totalRank)}-largest electorate in Maryland\n` +
            `• Democratic share: ${ordinal(demRank)}-highest among Maryland's 24 jurisdictions\n` +
            `• Republican share: ${ordinal(repRank)}-highest among Maryland's 24 jurisdictions\n` +
            `• Third-party/unaffiliated share: ${ordinal(thirdRank)}-highest among Maryland's 24 jurisdictions\n` +
            `• Registration margin: ${ordinal(balanceRank)}-most Democratic among Maryland's 24 jurisdictions\n\n` +
            `Neighbor comparison:\n${neighborNarrative}`;
    }

    function renderAnalysis() {
        const name = selector.value;
        const data = regionData[name];
        if (!data) return;

        if (name === "Statewide") {
            renderStatewide(data);
        } else {
            renderJurisdiction(name, data);
        }
    }

    async function copyAnalysis() {
        if (!copyText) return;

        try {
            if (navigator.clipboard && window.isSecureContext) {
                await navigator.clipboard.writeText(copyText);
            } else {
                const textarea = document.createElement("textarea");
                textarea.value = copyText;
                textarea.style.position = "absolute";
                textarea.style.left = "-9999px";
                document.body.appendChild(textarea);
                textarea.select();
                document.execCommand("copy");
                textarea.remove();
            }

            copyStatus.textContent = "Analysis copied.";
        } catch (error) {
            copyStatus.textContent = "Could not copy automatically.";
        }

        window.setTimeout(() => {
            copyStatus.textContent = "";
        }, 2500);
    }

    selector.addEventListener("change", renderAnalysis);
    copyButton.addEventListener("click", copyAnalysis);
    renderAnalysis();
})();
</script>
""".replace("__REGION_DATA__", region_data_json)

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Maryland Voter Registration Dashboard</title>

<style>
* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    background: #f4f5f7;
    color: #1f2933;
    font-family: Arial, Helvetica, sans-serif;
}}

.page {{
    width: 100%;
    max-width: 1180px;
    margin: 0 auto;
    padding: 28px 20px 40px;
}}

.header {{
    margin-bottom: 22px;
}}

.header h1 {{
    margin: 0 0 8px;
    font-size: clamp(1.65rem, 4vw, 2.35rem);
    line-height: 1.15;
}}

.subtitle {{
    margin: 0;
    color: #586574;
    font-size: 1rem;
    line-height: 1.5;
}}

.data-date {{
    margin: 7px 0 0;
    color: #647180;
    font-size: 0.9rem;
    font-weight: 600;
}}

.summary-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(175px, 1fr));
    gap: 12px;
    margin-bottom: 20px;
}}

.summary-card {{
    background: white;
    border: 0;
    border-radius: 10px;
    padding: 15px 16px;
}}

.summary-label {{
    color: #647180;
    font-size: 0.85rem;
    margin-bottom: 7px;
}}

.summary-value {{
    font-size: 1.5rem;
    font-weight: 700;
    line-height: 1.1;
}}

.summary-detail {{
    margin-top: 6px;
    color: #4e5b68;
    font-size: 0.88rem;
}}

.explainer {{
    background: #ffffff;
    border: 0;
    border-radius: 10px;
    padding: 16px 18px;
    margin-bottom: 20px;
}}

.explainer h2 {{
    margin: 0 0 8px;
    font-size: 1.05rem;
}}

.explainer p {{
    margin: 0;
    line-height: 1.55;
    color: #465463;
}}

.map-panel {{
    max-width: 1040px;
    margin: 0 auto;
    background: white;
    border: 0;
    border-radius: 12px;
    padding: 18px 18px 10px;
}}

.map-header {{
    margin-bottom: 4px;
}}

.map-header h2 {{
    margin: 0 0 5px;
    font-size: 1.25rem;
}}

.map-header p {{
    margin: 0;
    color: #647180;
    font-size: 0.9rem;
    line-height: 1.4;
}}

.map-wrap {{
    width: 100%;
    min-width: 0;
}}

.map-wrap .plotly-graph-div {{
    width: 100% !important;
    height: clamp(360px, 55vw, 500px) !important;
}}

.map-content {{
    position: relative;
    width: 100%;
}}

.map-legend {{
    position: absolute;
    right: -22px;
    top: 15%;
    z-index: 3;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 7px 6px;
    background: transparent;
    pointer-events: none;
}}

.map-legend-title {{
    margin-bottom: 7px;
    font-size: 0.66rem;
    font-weight: 600;
    text-align: center;
    white-space: nowrap;
}}

.map-legend-scale {{
    display: flex;
    gap: 6px;
    height: 250px;
}}

.map-legend-bar {{
    width: 10px;
    height: 100%;
    border-radius: 2px;
    background: linear-gradient(
        to bottom,
        {BLUE} 0%,
        {PURPLE} 50%,
        {RED} 100%
    );
}}

.map-legend-labels {{
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    font-size: 0.66rem;
    line-height: 1;
    white-space: nowrap;
}}

.methodology {{
    max-width: 1040px;
    margin: 14px auto 0;
    color: #5d6976;
    font-size: 0.84rem;
    line-height: 1.5;
}}

.methodology p {{
    margin: 5px 0;
}}

.methodology a {{
    color: #315f94;
}}

@media (max-width: 700px) {{
    .page {{
        padding: 18px 12px 28px;
    }}

    .summary-grid {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }}

    .summary-card {{
        padding: 13px;
    }}

    .summary-value {{
        font-size: 1.25rem;
    }}

    .map-panel {{
        padding: 14px 8px 8px;
    }}

    .map-legend {{
        position: static;
        width: min(92%, 520px);
        margin: 0 auto 6px;
        padding: 0;
    }}

    .map-legend-title {{
        margin-bottom: 5px;
    }}

    .map-legend-scale {{
        width: 100%;
        height: auto;
        flex-direction: column;
        gap: 5px;
    }}

    .map-legend-bar {{
        width: 100%;
        height: 12px;
        background: linear-gradient(
            to right,
            {RED} 0%,
            {PURPLE} 50%,
            {BLUE} 100%
        );
    }}

    .map-legend-labels {{
        width: 100%;
        flex-direction: row-reverse;
        justify-content: space-between;
    }}
}}

@media (max-width: 430px) {{
    .summary-grid {{
        grid-template-columns: 1fr;
    }}

    .explainer {{
        padding: 14px;
    }}

    .map-wrap .plotly-graph-div {{
        height: 360px !important;
    }}
}}

{selector_css}
{analysis_css}
</style>
</head>

<body>
<main class="page">

    <header class="header">
        <h1>Maryland voter registration</h1>

        <p class="subtitle">
            Current active voter registration totals and party balance
            across Maryland's 24 jurisdictions.
        </p>

        <p class="data-date">
            Maryland State Board of Elections monthly report: {source_date}
        </p>
    </header>

    <section class="explainer">
        <h2>How to read the map</h2>
        <p>
            Blue jurisdictions have a Democratic registration advantage and
            red jurisdictions have a Republican advantage. Jurisdictions
            closer to purple have a more balanced Democratic-Republican
            registration split or a larger share of third-party and
            unaffiliated voters. Hover over a jurisdiction for the actual
            registration totals and Democratic-Republican margin.
        </p>
    </section>

    <section class="map-panel">
        <div class="map-header">
            <h2>Jurisdiction registration balance</h2>
            <p>
                Color shows registration balance; it does not represent
                election results or a forecast.
            </p>
        </div>

        <div class="map-content">
            <div class="map-wrap">
                {map_html}
            </div>

            <div class="map-legend" aria-label="Registration balance legend">
                <div class="map-legend-title">Registration balance</div>
                <div class="map-legend-scale">
                    <div class="map-legend-bar"></div>
                    <div class="map-legend-labels">
                        <span>Strong D</span>
                        <span>D lean</span>
                        <span>Even</span>
                        <span>R lean</span>
                        <span>Strong R</span>
                    </div>
                </div>
            </div>
        </div>
    </section>

    <section class="region-selector" aria-label="Registration area">
        <label for="region-select">View registration for:</label>
        <select id="region-select">
            {selector_options}
        </select>
    </section>

    <section class="summary-grid">
        <div class="summary-card">
            <div class="summary-label">Active registered voters</div>
            <div class="summary-value">{statewide["total"]:,}</div>
            <div class="summary-detail">Statewide</div>
        </div>

        <div class="summary-card">
            <div class="summary-label">Democratic</div>
            <div class="summary-value">{statewide["dem"]:,}</div>
            <div class="summary-detail">{statewide["dem_pct"]:.2f}% of voters</div>
        </div>

        <div class="summary-card">
            <div class="summary-label">Republican</div>
            <div class="summary-value">{statewide["rep"]:,}</div>
            <div class="summary-detail">{statewide["rep_pct"]:.2f}% of voters</div>
        </div>

        <div class="summary-card">
            <div class="summary-label">Third-party/unaffiliated</div>
            <div class="summary-value">{statewide["third"]:,}</div>
            <div class="summary-detail">{statewide["third_pct"]:.2f}% of voters</div>
        </div>

        <div class="summary-card">
            <div class="summary-label">Statewide registration edge</div>
            <div class="summary-value">{statewide["margin_party"]}</div>
            <div class="summary-detail">
                {statewide["margin_count"]:,} voters ·
                {statewide["margin_pct"]:.2f} points
            </div>
        </div>
    </section>

    <section class="analysis-panel" aria-labelledby="analysis-title">
        <div class="analysis-heading-row">
            <h2 id="analysis-title">Statewide analysis</h2>
            <button id="copy-analysis" class="copy-analysis" type="button">
                Copy analysis
            </button>
        </div>

        <p id="analysis-summary" class="analysis-summary"></p>
        <div id="analysis-comparison"></div>
        <div id="analysis-neighbors"></div>
        <p id="copy-status" class="copy-status" aria-live="polite"></p>
    </section>

    <section class="methodology">
        <p>
            <strong>About the colors:</strong>
            Democratic-Republican registration determines the red or blue
            direction. The third-party/unaffiliated share pulls the color
            toward purple. A nonlinear color scale gives greater visual
            separation to jurisdictions with relatively close registration
            margins.
        </p>

        <p>
            Data as of {source_date}. Source:
            <a href="{MD_VOTER_STATS_URL}" target="_blank" rel="noopener noreferrer">
                Maryland State Board of Elections
            </a>.
            Jurisdiction boundaries are from the U.S. Census Bureau.
        </p>
    </section>

</main>

{selector_script}
{analysis_script}

</body>
</html>
"""

    OUTPUT_FILE.write_text(
        html,
        encoding="utf-8",
    )

    print(f"\nDashboard saved: {OUTPUT_FILE}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":
    download_shapes()
    extract_shapes()

    stats = load_stats()
    presidential = load_presidential_results()
    source_date = source_date_display(stats)

    jurisdictions = load_jurisdictions()

    validate_names(
        stats,
        jurisdictions,
    )

    merged = merge_data(
        jurisdictions,
        stats,
    )

    merged = merge_presidential_results(
        merged,
        presidential,
    )

    merged = calculate_map_metric(merged)
    merged = add_margin_labels(merged)

    statewide = calculate_statewide_stats(stats)
    fig = build_map(merged)

    build_dashboard(
        fig,
        statewide,
        source_date,
        merged,
    )

    print("\nDone.")




