#!/usr/bin/env python3
"""
data_acquisition.py

High-performance, resumable downloader for up to 10,000 high-quality protein
structures from the RCSB PDB.

However you can always edit MAX_ROWS=SAMPLE_COUNT to get the specified number of samples for testing.

Requirements implemented:
- RCSB Search API v2 JSON POST query (no Biopython for query)
- Strict biological filters:
    * Protein only
    * Monomeric proteins only
    * Resolution <= 2.0 Å
    * Sequence length 100-400 amino acids
    * Up to 10,000 rows
- Direct downloads using requests (no Bio.PDB.PDBList)
- Flat output files in data/raw_pdb/{pdb_id}.ent
- Concurrent downloads with ThreadPoolExecutor(max_workers=20)
- Idempotent / resumable:
    * Skip if file already exists
- tqdm progress bar around as_completed()
- Clean terminal output using tqdm.write(...)
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import requests
from tqdm import tqdm

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
DOWNLOAD_URL_TEMPLATE = "https://files.rcsb.org/download/{pdb_id}.pdb"

OUTPUT_DIR = Path("data/raw_pdb")
MAX_ROWS = 1000
MAX_WORKERS = 20
REQUEST_TIMEOUT = 30


def build_search_payload(limit: int = MAX_ROWS) -> dict:
    """
    Construct the RCSB Search API v2 payload using a 'group' query with an
    'and' operator so all biological constraints are enforced.

    Criteria:
    - Entity Type: Protein (only)
    - Oligomeric State: Monomers only (protein entity count == 1)
    - High Resolution: <= 2.0 Å
    - Sequence Length: 100 to 400 aa
    """
    return {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.selected_polymer_entity_types",
                        "operator": "exact_match",
                        "value": "Protein (only)",
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.deposited_polymer_entity_instance_count",
                        "operator": "equals",
                        "value": 1,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "reflns.d_resolution_high",
                        "operator": "less_or_equal",
                        "value": 2.0,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "entity_poly.rcsb_sample_sequence_length",
                        "operator": "range",
                        "value": {
                            "from": 100,
                            "to": 400,
                            "include_lower": True,
                            "include_upper": True,
                        },
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {
                "start": 0,
                "rows": limit,
            }
        },
    }


def fetch_pdb_ids(session: requests.Session, limit: int = MAX_ROWS) -> List[str]:
    payload = build_search_payload(limit=limit)

    try:
        response = session.post(SEARCH_URL, json=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        data = response.json()
        result_set = data.get("result_set", [])

        pdb_ids = []
        for item in result_set:
            identifier = item.get("identifier")
            if identifier:
                pdb_ids.append(identifier.lower())

        return pdb_ids

    except Exception as e:
        print(f"Error while fetching PDB IDs: {e}")
        return []


def download_single_pdb(
    pdb_id: str,
    output_dir: Path,
    timeout: int = REQUEST_TIMEOUT,
) -> Tuple[str, bool]:
    """
    Download a single PDB file as {pdb_id}.ent into output_dir.

    Returns:
        (pdb_id, downloaded_freshly)
        - downloaded_freshly = False if skipped because file already exists
        - downloaded_freshly = True if downloaded now

    Raises:
        requests.HTTPError or other exceptions on failure
    """
    output_path = output_dir / f"{pdb_id}.ent"

    # Requirement: use os.path.exists() before making network request
    if os.path.exists(output_path):
        return pdb_id, False

    url = DOWNLOAD_URL_TEMPLATE.format(pdb_id=pdb_id.upper())

    with requests.get(url, timeout=timeout, stream=True) as response:
        response.raise_for_status()

        # Write to a temporary file first, then atomically replace
        temp_path = output_path.with_suffix(".ent.part")
        with open(temp_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    handle.write(chunk)

        os.replace(temp_path, output_path)

    return pdb_id, True


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    downloaded_count = 0
    counter_lock = threading.Lock()

    with requests.Session() as session:
        session.headers.update(
            {
                "User-Agent": "rcsb-high-performance-downloader/1.0",
                "Accept": "application/json",
            }
        )

        pdb_ids = fetch_pdb_ids(session=session, limit=MAX_ROWS)

    if not pdb_ids:
        tqdm.write("No PDB IDs matched the query.")
        return

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_pdb_id = {
            executor.submit(download_single_pdb, pdb_id, OUTPUT_DIR): pdb_id
            for pdb_id in pdb_ids
        }

        for future in tqdm(
            as_completed(future_to_pdb_id),
            total=len(future_to_pdb_id),
            desc="Processing PDB downloads",
        ):
            pdb_id = future_to_pdb_id[future]

            try:
                completed_pdb_id, downloaded_freshly = future.result()

                if downloaded_freshly:
                    with counter_lock:
                        downloaded_count += 1
                        current_count = downloaded_count

                    tqdm.write(
                        f"{completed_pdb_id}.ent downloaded successfully. count - {current_count}"
                    )

            except Exception as exc:
                tqdm.write(f"{pdb_id}.ent failed: {exc}")


if __name__ == "__main__":
    main()
