#!/usr/bin/env python3
"""
Download the current Atlantic HURDAT2 dataset from NOAA/NHC, extract all
North Atlantic storms for a selected season, and write the results in a
JSON format that can be consumed directly by the hurricane-tracker HTML page.

Fixes vs. earlier version:
- correct NHC HURDAT2 URL discovery (no duplicate /data/ in the path)
- more robust href matching on the NHC archive page
- direct fallback to the hurdat directory index if the archive page markup changes
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

ARCHIVE_PAGE = "https://www.nhc.noaa.gov/data/"
HURDAT_DIR = "https://www.nhc.noaa.gov/data/hurdat/"
USER_AGENT = "Mozilla/5.0 (compatible; HurricaneTrackExporter/1.1; +https://openai.com)"
ATLANTIC_FILENAME_RE = re.compile(r"hurdat2(?:-atl)?-1851-\d{4}-\d{8}\.txt", re.IGNORECASE)
HREF_RE = re.compile(
    r'''href=["'](?P<href>[^"']*(?:hurdat/)?(?P<file>hurdat2(?:-atl)?-1851-\d{4}-\d{8}\.txt))['"]''',
    re.IGNORECASE,
)
HEADER_RE = re.compile(r"^(AL\d{6}),\s*([^,]+),\s*(\d+),\s*$")


@dataclass
class StormHeader:
    storm_id: str
    name: str
    entry_count: int


def fetch_text(url: str, timeout: int = 60) -> str:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:
        charset = resp.headers.get_content_charset() or "utf-8"
        return resp.read().decode(charset, errors="replace")


def discover_latest_atlantic_hurdat2_url() -> str:
    # First try the official NHC data page.
    html = fetch_text(ARCHIVE_PAGE)
    candidates: list[str] = []

    for match in HREF_RE.finditer(html):
        href = match.group("href")
        candidates.append(urljoin(ARCHIVE_PAGE, href))

    # Fallback: if the page content changes, inspect the hurdat directory index.
    if not candidates:
        index_html = fetch_text(HURDAT_DIR)
        for match in ATLANTIC_FILENAME_RE.finditer(index_html):
            candidates.append(urljoin(HURDAT_DIR, match.group(0)))

    if not candidates:
        raise RuntimeError(
            "Could not discover the Atlantic HURDAT2 download URL from the NHC archive page or hurdat directory."
        )

    # Sort by the date embedded at the end of the filename (MMDDYYYY).
    def sort_key(url: str) -> tuple[int, str]:
        filename = url.rsplit("/", 1)[-1]
        m = re.search(r"(\d{8})\.txt$", filename)
        return (int(m.group(1)) if m else -1, filename)

    return sorted(set(candidates), key=sort_key, reverse=True)[0]


def parse_coord(value: str) -> float | None:
    text = value.strip().upper()
    if not text or text in {"-99", "-999"}:
        return None

    suffix = text[-1]
    number = float(text[:-1]) if suffix in {"N", "S", "E", "W"} else float(text)

    if suffix == "N":
        return number
    if suffix == "S":
        return -number
    if suffix == "W":
        # Positive west longitudes so they can go straight into the HTML page's lonToX() logic.
        return number
    if suffix == "E":
        # East longitude represented as negative west longitude.
        return -number
    return number


def parse_int_or_none(value: str) -> int | None:
    text = value.strip()
    if text in {"", "-99", "-999"}:
        return None
    return int(text)


def iter_storm_blocks(lines: Iterable[str]) -> Iterable[tuple[StormHeader, list[str]]]:
    line_iter = iter(lines)
    for raw in line_iter:
        line = raw.strip("\n\r")
        if not line.strip():
            continue

        header_match = HEADER_RE.match(line)
        if not header_match:
            continue

        storm_id, name, entry_count_text = header_match.groups()
        entry_count = int(entry_count_text)
        block = []
        for _ in range(entry_count):
            try:
                block.append(next(line_iter).strip("\n\r"))
            except StopIteration as exc:
                raise RuntimeError(f"Unexpected end of file while reading storm {storm_id}.") from exc

        yield StormHeader(storm_id=storm_id, name=name.strip(), entry_count=entry_count), block


def parse_data_line(line: str) -> dict[str, Any]:
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 8:
        raise ValueError(f"Unexpected HURDAT2 data line: {line}")

    date_str = parts[0]
    time_str = parts[1]
    record_id = parts[2]
    status = parts[3]
    lat = parse_coord(parts[4])
    lon = parse_coord(parts[5])
    wind_kt = parse_int_or_none(parts[6])
    pressure_mb = parse_int_or_none(parts[7])

    if lat is None or lon is None:
        raise ValueError(f"Missing latitude/longitude in HURDAT2 line: {line}")

    iso_time = (
        f"{date_str[0:4]}-{date_str[4:6]}-{date_str[6:8]}T"
        f"{time_str[0:2]}:{time_str[2:4]}:00Z"
    )

    return {
        "date": date_str,
        "time": time_str,
        "isoTime": iso_time,
        "recordId": record_id,
        "status": status,
        "lat": round(lat, 1),
        "lon": round(lon, 1),
        "windKt": wind_kt,
        "pressureMb": pressure_mb,
    }


def saffir_simpson_category(max_wind_kt: int | None) -> int | None:
    if max_wind_kt is None:
        return None
    if max_wind_kt >= 137:
        return 5
    if max_wind_kt >= 113:
        return 4
    if max_wind_kt >= 96:
        return 3
    if max_wind_kt >= 83:
        return 2
    if max_wind_kt >= 64:
        return 1
    return 0


def build_season_payload(year: int, dataset_url: str, include_all_systems: bool = False) -> dict[str, Any]:
    text = fetch_text(dataset_url, timeout=120)
    storms: list[dict[str, Any]] = []

    for header, data_lines in iter_storm_blocks(text.splitlines()):
        if not header.storm_id.startswith("AL"):
            continue
        if int(header.storm_id[-4:]) != year:
            continue

        points = [parse_data_line(line) for line in data_lines]
        reached_hurricane = any(p["status"] == "HU" for p in points)

        if not include_all_systems and not reached_hurricane:
            continue

        valid_winds = [p["windKt"] for p in points if p["windKt"] is not None]
        valid_pressures = [p["pressureMb"] for p in points if p["pressureMb"] is not None]
        max_wind = max(valid_winds) if valid_winds else None
        min_pressure = min(valid_pressures) if valid_pressures else None

        storms.append(
            {
                "id": header.storm_id,
                "name": header.name,
                "year": year,
                "atcfNumber": int(header.storm_id[2:4]),
                "entryCount": header.entry_count,
                "reachedHurricane": reached_hurricane,
                "maxWindKt": max_wind,
                "minPressureMb": min_pressure,
                "peakCategory": saffir_simpson_category(max_wind),
                "startIsoTime": points[0]["isoTime"] if points else None,
                "endIsoTime": points[-1]["isoTime"] if points else None,
                "trackData": points,
            }
        )

    storms.sort(key=lambda s: (s["startIsoTime"] or "", s["id"]))

    return {
        "basin": "North Atlantic",
        "basinCode": "AL",
        "year": year,
        "generatedAtUtc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": {
            "provider": "NOAA / National Hurricane Center",
            "archivePage": ARCHIVE_PAGE,
            "datasetUrl": dataset_url,
            "datasetType": "HURDAT2",
        },
        "stormCount": len(storms),
        "filter": {
            "includeAllSystems": include_all_systems,
            "description": (
                "All Atlantic tropical/subtropical systems for the selected year"
                if include_all_systems
                else "Only Atlantic storms that reached hurricane status at least once"
            ),
        },
        "storms": storms,
    }


def make_js_wrapper(payload: dict[str, Any], var_name: str) -> str:
    return (
        "// Generated from NOAA/NHC Atlantic HURDAT2\n"
        f"window.{var_name} = "
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        + ";\n"
    )


def positive_year(value: str) -> int:
    year = int(value)
    if year < 1851:
        raise argparse.ArgumentTypeError("Atlantic HURDAT2 begins in 1851.")
    return year


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch Atlantic HURDAT2 data from NOAA/NHC and export one season in a JSON format "
            "that can be read by the hurricane-track HTML page."
        )
    )
    parser.add_argument("year", type=positive_year, help="Season year to export, e.g. 1992")
    parser.add_argument(
        "--out",
        type=Path,
        help="JSON output path. Defaults to atlantic_hurricanes_<year>.json in the current directory.",
    )
    parser.add_argument(
        "--js-out",
        type=Path,
        help="Optional JS wrapper output path for direct <script src=...> loading.",
    )
    parser.add_argument(
        "--js-var",
        default="HURRICANE_SEASON_DATA",
        help="Global variable name used when writing --js-out. Default: HURRICANE_SEASON_DATA",
    )
    parser.add_argument(
        "--include-all-systems",
        action="store_true",
        help="Include all Atlantic tropical/subtropical systems for the year, not just storms that reached HU status.",
    )
    parser.add_argument(
        "--dataset-url",
        help="Override the discovered Atlantic HURDAT2 URL.",
    )

    args = parser.parse_args()
    out_path = args.out or Path(f"atlantic_hurricanes_{args.year}.json")

    try:
        dataset_url = args.dataset_url or discover_latest_atlantic_hurdat2_url()
        payload = build_season_payload(
            year=args.year,
            dataset_url=dataset_url,
            include_all_systems=args.include_all_systems,
        )
    except (HTTPError, URLError) as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.js_out:
        args.js_out.parent.mkdir(parents=True, exist_ok=True)
        args.js_out.write_text(make_js_wrapper(payload, args.js_var), encoding="utf-8")

    print(f"Using dataset: {dataset_url}")
    print(f"Wrote JSON: {out_path}")
    print(f"Storms exported: {payload['stormCount']}")
    if args.js_out:
        print(f"Wrote JS: {args.js_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
