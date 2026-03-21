"""
ERA5 data download script supporting multiple coastal sites.

Usage:
    python load_data.py                    # download all sites
    python load_data.py --site qingdao     # download one site
    python load_data.py --site qingdao --site dubai  # download selected sites

Each site is saved as data/era5_{site_name}.csv.
The default Qingdao file is also symlinked/copied to preserve backward
compatibility with the original data path.
"""

import argparse
import shutil
from pathlib import Path

try:
    import cdsapi
except ImportError:
    cdsapi = None

SITES = {
    "qingdao": {"latitude": 36.25, "longitude": 120.5, "description": "Temperate monsoon, humid summers"},
    "dubai": {"latitude": 25.25, "longitude": 55.25, "description": "Arid subtropical, extreme dry heat"},
    "singapore": {"latitude": 1.25, "longitude": 103.75, "description": "Tropical, year-round high humidity"},
    "miami": {"latitude": 25.75, "longitude": -80.25, "description": "Subtropical, hurricane-prone coast"},
}

VARIABLES = [
    "2m_dewpoint_temperature",
    "10m_wind_gust_since_previous_post_processing",
    "mean_sea_level_pressure",
    "skin_temperature",
    "surface_pressure",
    "surface_solar_radiation_downwards",
    "sea_surface_temperature",
    "surface_thermal_radiation_downwards",
    "2m_temperature",
    "total_precipitation",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "100m_u_component_of_wind",
    "100m_v_component_of_wind",
]

DATASET = "reanalysis-era5-single-levels-timeseries"
DATE_RANGE = "2020-01-01/2026-03-11"


def download_site(site_name: str, output_dir: str = "data") -> Path:
    """Download ERA5 data for a single site and return the output path."""

    site = SITES[site_name]
    output_path = Path(output_dir) / f"era5_{site_name}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        print(f"[{site_name}] File already exists: {output_path}")
        return output_path

    if cdsapi is None:
        raise RuntimeError(
            f"[{site_name}] cdsapi is not installed. Install it with:\n"
            "  pip install cdsapi\n"
            "and configure ~/.cdsapirc with your CDS API key.\n"
            "See https://cds.climate.copernicus.eu/how-to-api"
        )

    print(f"[{site_name}] Downloading ERA5 data ({site['description']})...")
    print(f"  Location: {site['latitude']}N, {site['longitude']}E")

    request = {
        "variable": VARIABLES,
        "location": {"longitude": site["longitude"], "latitude": site["latitude"]},
        "date": [DATE_RANGE],
        "data_format": "csv",
    }

    client = cdsapi.Client()
    client.retrieve(DATASET, request, str(output_path))
    print(f"[{site_name}] Saved to: {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download ERA5 data for coastal heat-stress sites.")
    parser.add_argument(
        "--site",
        action="append",
        choices=list(SITES.keys()),
        help="Site(s) to download. Omit to download all.",
    )
    parser.add_argument("--output-dir", default="data")
    args = parser.parse_args()

    sites_to_download = args.site if args.site else list(SITES.keys())

    for site_name in sites_to_download:
        download_site(site_name, args.output_dir)

    # Backward compatibility: if qingdao was downloaded and the legacy file
    # does not exist, copy it.
    legacy_path = Path(args.output_dir) / "reanalysis-era5-single-levels-timeseries-sfc1esqojiw.csv"
    qingdao_path = Path(args.output_dir) / "era5_qingdao.csv"
    if qingdao_path.exists() and not legacy_path.exists():
        shutil.copy2(qingdao_path, legacy_path)
        print(f"Copied {qingdao_path} -> {legacy_path} for backward compatibility.")

    print("\nAvailable sites:")
    for name, info in SITES.items():
        status = "downloaded" if (Path(args.output_dir) / f"era5_{name}.csv").exists() else "not downloaded"
        print(f"  {name:12s} ({info['latitude']:6.2f}N, {info['longitude']:7.2f}E) - {info['description']} [{status}]")


if __name__ == "__main__":
    main()
