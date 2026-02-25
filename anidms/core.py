"""
AniDMS core database interface.

This module provides two core features:
1. Query DMS NetCDF month files for a date range.
2. Annotate tracking data with daily DMS values using LAEA-based IDW interpolation.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import time
import warnings
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests
import xarray as xr
from scipy.spatial import cKDTree


class AniDMS:
    """
    Interface for AniDMS monthly NetCDF data.

    Modes:
    - Local mode: provide `data_path` that contains month files like DMS_YYYYMM_4km_NA.nc.
    - Zenodo mode: omit `data_path`; files are resolved from Zenodo concept DOI and downloaded on demand.
    """

    DEFAULT_ZENODO_DOI = "10.5281/zenodo.18615736"
    ZENODO_RECORDS_API = "https://zenodo.org/api/records"
    ZENODO_ANON_MAX_PAGE_SIZE = 25
    MONTH_FILE_PATTERN = re.compile(r"^DMS_(\d{6})_4km_NA\.nc$", re.IGNORECASE)
    DEFAULT_LAEA_PROJ = (
        "+proj=laea +lat_0=37.5 +lon_0=-32.5 +x_0=0 +y_0=0 "
        "+datum=WGS84 +units=m +no_defs"
    )

    def __init__(
        self,
        data_path: Optional[str] = None,
        zenodo_doi: str = DEFAULT_ZENODO_DOI,
        cache_dir: str = "./dms_cache",
        request_timeout: int = 60,
        request_retries: int = 3,
        retry_backoff: float = 1.5,
    ):
        """
        Initialize AniDMS database.

        Args:
            data_path: Local folder containing monthly DMS files when local mode is used.
            zenodo_doi: Concept DOI used for dynamic Zenodo version discovery.
            cache_dir: Local cache for downloaded monthly files (Zenodo mode).
            request_timeout: HTTP timeout in seconds.
            request_retries: Number of retry attempts for HTTP operations.
            retry_backoff: Exponential backoff base between retries.
        """
        self.data_path = Path(data_path).expanduser().resolve() if data_path else None
        self.local_mode = self.data_path is not None
        self.zenodo_doi = self._normalize_doi(zenodo_doi)
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.request_timeout = int(request_timeout)
        self.request_retries = int(request_retries)
        self.retry_backoff = float(retry_backoff)

        if self.local_mode:
            if not self.data_path.exists() or not self.data_path.is_dir():
                raise ValueError(f"Local data path does not exist or is not a directory: {self.data_path}")
            self._local_month_index = self._scan_local_month_files(self.data_path)
            print(f"AniDMS initialized in local mode with {len(self._local_month_index)} month files")
        else:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._local_month_index = {}
            print(f"AniDMS initialized in Zenodo mode (DOI: {self.zenodo_doi})")

        self._zenodo_records: Optional[List[Dict[str, Any]]] = None
        self._month_index: Optional[Dict[str, Dict[str, Any]]] = None
        self._month_duplicates: Dict[str, List[Dict[str, Any]]] = {}

    @staticmethod
    def _normalize_doi(doi: str) -> str:
        doi_clean = doi.strip()
        prefixes = ("https://doi.org/", "http://doi.org/", "doi:")
        lower = doi_clean.lower()
        for prefix in prefixes:
            if lower.startswith(prefix):
                return doi_clean[len(prefix):]
        return doi_clean

    @staticmethod
    def _parse_date(value: Union[str, datetime, date, pd.Timestamp]) -> date:
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, pd.Timestamp):
            return value.date()
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            parsed = pd.to_datetime(value, errors="raise")
            return parsed.date()
        raise TypeError(f"Unsupported date type: {type(value)!r}")

    @staticmethod
    def _month_start(d: date) -> date:
        return date(d.year, d.month, 1)

    @staticmethod
    def _next_month(d: date) -> date:
        if d.month == 12:
            return date(d.year + 1, 1, 1)
        return date(d.year, d.month + 1, 1)

    @staticmethod
    def _month_key(d: date) -> str:
        return f"{d.year:04d}{d.month:02d}"

    def _months_in_range(self, start_date: date, end_date: date) -> List[str]:
        months: List[str] = []
        cursor = self._month_start(start_date)
        end_month = self._month_start(end_date)
        while cursor <= end_month:
            months.append(self._month_key(cursor))
            cursor = self._next_month(cursor)
        return months

    def _scan_local_month_files(self, directory: Path) -> Dict[str, Path]:
        month_map: Dict[str, Path] = {}
        for fp in sorted(directory.glob("DMS_*_4km_NA.nc")):
            match = self.MONTH_FILE_PATTERN.match(fp.name)
            if not match:
                continue
            month_map[match.group(1)] = fp
        return month_map

    def _request_json_with_retries(self, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        retry_status = {429, 500, 502, 503, 504}
        last_error: Optional[Exception] = None

        for attempt in range(1, self.request_retries + 1):
            try:
                response = requests.get(url, params=params, timeout=self.request_timeout)
                if response.status_code in retry_status:
                    raise requests.HTTPError(
                        f"Transient HTTP status {response.status_code}: {response.text[:200]}",
                        response=response,
                    )
                response.raise_for_status()
                return response.json()
            except requests.HTTPError as exc:
                last_error = exc
                status = exc.response.status_code if exc.response is not None else None
                if status is not None and status not in retry_status:
                    break
                if attempt == self.request_retries:
                    break
                sleep_seconds = self.retry_backoff ** (attempt - 1)
                time.sleep(sleep_seconds)
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt == self.request_retries:
                    break
                sleep_seconds = self.retry_backoff ** (attempt - 1)
                time.sleep(sleep_seconds)

        raise RuntimeError(
            f"Failed to request JSON from {url} with params={params} "
            f"after {self.request_retries} attempts"
        ) from last_error

    def _download_file_with_retries(self, url: str, destination: Path) -> None:
        retry_status = {429, 500, 502, 503, 504}
        last_error: Optional[Exception] = None

        for attempt in range(1, self.request_retries + 1):
            part_file = destination.with_suffix(destination.suffix + ".part")
            try:
                if part_file.exists():
                    part_file.unlink()

                with requests.get(url, stream=True, timeout=self.request_timeout) as response:
                    if response.status_code in retry_status:
                        raise requests.HTTPError(
                            f"Transient HTTP status {response.status_code}: {response.text[:200]}",
                            response=response,
                        )
                    response.raise_for_status()

                    with part_file.open("wb") as fout:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                fout.write(chunk)

                part_file.replace(destination)
                return
            except requests.RequestException as exc:
                last_error = exc
                if part_file.exists():
                    part_file.unlink()
                if attempt == self.request_retries:
                    break
                sleep_seconds = self.retry_backoff ** (attempt - 1)
                time.sleep(sleep_seconds)

        raise RuntimeError(f"Failed to download {url} after {self.request_retries} attempts") from last_error

    @staticmethod
    def _parse_version_tuple(version: str) -> Tuple[int, ...]:
        if not version:
            return (0,)
        numbers = re.findall(r"\d+", str(version))
        if not numbers:
            return (0,)
        return tuple(int(x) for x in numbers)

    @staticmethod
    def _parse_datetime_safe(value: Optional[str]) -> datetime:
        if not value:
            return datetime.min
        parsed = pd.to_datetime(value, errors="coerce", utc=True)
        if pd.isna(parsed):
            return datetime.min
        return parsed.to_pydatetime().replace(tzinfo=None)

    @staticmethod
    def _extract_concept_recid_from_doi(doi: str) -> Optional[str]:
        match = re.search(r"10\.5281/zenodo\.(\d+)", doi, flags=re.IGNORECASE)
        if not match:
            return None
        return match.group(1)

    def _fetch_records_by_search(self, query: str, page_size: int = ZENODO_ANON_MAX_PAGE_SIZE) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        page = 1
        while True:
            params = {"q": query, "page": page, "size": page_size, "all_versions": 1}
            payload = self._request_json_with_retries(self.ZENODO_RECORDS_API, params=params)
            hits = payload.get("hits", {}).get("hits", [])
            if not hits:
                break
            records.extend(hits)
            if len(hits) < page_size:
                break
            page += 1
        return records

    def _fetch_records_from_versions_endpoint(self, recid: str, page_size: int = ZENODO_ANON_MAX_PAGE_SIZE) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        page = 1
        versions_url = f"{self.ZENODO_RECORDS_API}/{recid}/versions"
        while True:
            params = {"page": page, "size": page_size}
            payload = self._request_json_with_retries(versions_url, params=params)
            hits = payload.get("hits", {}).get("hits", [])
            if not hits:
                break
            records.extend(hits)
            if len(hits) < page_size:
                break
            page += 1
        return records

    def _fetch_records_from_versions_url(
        self, versions_url: str, page_size: int = ZENODO_ANON_MAX_PAGE_SIZE
    ) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        page = 1
        while True:
            params = {"page": page, "size": page_size}
            payload = self._request_json_with_retries(versions_url, params=params)
            hits = payload.get("hits", {}).get("hits", [])
            if not hits:
                break
            records.extend(hits)
            if len(hits) < page_size:
                break
            page += 1
        return records

    def _fetch_zenodo_records(self) -> List[Dict[str, Any]]:
        """Fetch all Zenodo records linked to the configured concept DOI."""
        print(f"Fetching Zenodo records for concept DOI: {self.zenodo_doi}")

        all_records: Dict[int, Dict[str, Any]] = {}
        strategy_errors: List[str] = []
        concept_recid = self._extract_concept_recid_from_doi(self.zenodo_doi)

        search_queries: List[Tuple[str, str]] = []
        if concept_recid:
            search_queries.append(("conceptrecid", f"conceptrecid:{concept_recid}"))
        search_queries.append(("conceptdoi", f"conceptdoi:{self.zenodo_doi}"))
        search_queries.append(("doi", f"doi:{self.zenodo_doi}"))

        for strategy_name, query in search_queries:
            try:
                hits = self._fetch_records_by_search(query)
                for record in hits:
                    record_id = int(record.get("id"))
                    all_records[record_id] = record
                if hits:
                    print(f"Zenodo strategy '{strategy_name}' found {len(hits)} records")
            except Exception as exc:  # noqa: BLE001
                strategy_errors.append(f"{strategy_name}: {exc}")

        if concept_recid:
            # Combined query helps when the provided id could behave as conceptrecid or recid.
            try:
                combo_query = f"conceptrecid:{concept_recid} OR recid:{concept_recid}"
                hits = self._fetch_records_by_search(combo_query)
                for record in hits:
                    record_id = int(record.get("id"))
                    all_records[record_id] = record
                if hits:
                    print(f"Zenodo strategy 'conceptrecid_or_recid' found {len(hits)} records")
            except Exception as exc:  # noqa: BLE001
                strategy_errors.append(f"conceptrecid_or_recid: {exc}")

        if concept_recid:
            try:
                single = self._request_json_with_retries(f"{self.ZENODO_RECORDS_API}/{concept_recid}")
                if isinstance(single, dict) and "id" in single:
                    all_records[int(single["id"])] = single
            except Exception as exc:  # noqa: BLE001
                strategy_errors.append(f"record_by_recid: {exc}")

            try:
                hits = self._fetch_records_from_versions_endpoint(concept_recid)
                for record in hits:
                    record_id = int(record.get("id"))
                    all_records[record_id] = record
                if hits:
                    print(f"Zenodo versions endpoint found {len(hits)} records")
            except Exception as exc:  # noqa: BLE001
                strategy_errors.append(f"versions_endpoint: {exc}")

        # Follow versions links discovered in record payloads. This is the most reliable
        # way to fetch all versions when search results only include latest record.
        version_link_errors: List[str] = []
        seed_records = list(all_records.values())
        for record in seed_records:
            links = record.get("links", {}) if isinstance(record, dict) else {}
            versions_link = links.get("versions")
            if not versions_link:
                continue
            try:
                hits = self._fetch_records_from_versions_url(versions_link)
                for ver in hits:
                    record_id = int(ver.get("id"))
                    all_records[record_id] = ver
                if hits:
                    print(
                        f"Zenodo versions-link fetch from record={record.get('id')} "
                        f"found {len(hits)} records"
                    )
            except Exception as exc:  # noqa: BLE001
                version_link_errors.append(f"record={record.get('id')}: {exc}")
        strategy_errors.extend([f"versions_link: {err}" for err in version_link_errors])

        if not all_records:
            details = "; ".join(strategy_errors) if strategy_errors else "no strategies attempted"
            raise RuntimeError(
                f"No Zenodo records found for concept DOI {self.zenodo_doi}. "
                f"Attempt details: {details}"
            )

        records = list(all_records.values())
        print(f"Discovered {len(records)} Zenodo record versions")
        return records

    def _meta_sort_key(self, meta: Dict[str, Any]) -> Tuple[Tuple[int, ...], datetime, int]:
        return (
            tuple(meta.get("version_tuple", (0,))),
            self._parse_datetime_safe(meta.get("publication_date")),
            int(meta.get("record_id", 0)),
        )

    def _build_month_index(self, records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Build month-to-file mapping from Zenodo record metadata.

        Returns:
            Dict mapping YYYYMM -> selected file metadata.
        """
        candidates: Dict[str, List[Dict[str, Any]]] = {}

        for record in records:
            metadata = record.get("metadata", {})
            record_id = int(record.get("id"))
            version = str(metadata.get("version", ""))
            publication_date = metadata.get("publication_date") or record.get("created")

            for fmeta in record.get("files", []):
                filename = fmeta.get("key") or fmeta.get("filename")
                if not filename:
                    continue

                match = self.MONTH_FILE_PATTERN.match(filename)
                if not match:
                    continue

                yyyymm = match.group(1)
                links = fmeta.get("links", {})
                url = links.get("download") or links.get("self")
                if not url:
                    continue

                meta = {
                    "yyyymm": yyyymm,
                    "record_id": record_id,
                    "version": version,
                    "version_tuple": self._parse_version_tuple(version),
                    "publication_date": publication_date,
                    "filename": filename,
                    "url": url,
                    "size": fmeta.get("size"),
                    "checksum": fmeta.get("checksum"),
                }
                candidates.setdefault(yyyymm, []).append(meta)

        if not candidates:
            raise RuntimeError(
                "No monthly files matching DMS_YYYYMM_4km_NA.nc were found in Zenodo records."
            )

        month_index: Dict[str, Dict[str, Any]] = {}
        duplicates: Dict[str, List[Dict[str, Any]]] = {}

        for yyyymm, options in candidates.items():
            if len(options) > 1:
                duplicates[yyyymm] = options
            best = max(options, key=self._meta_sort_key)
            month_index[yyyymm] = best

        self._month_duplicates = duplicates
        for yyyymm, options in duplicates.items():
            best = month_index[yyyymm]
            candidates_str = ", ".join(
                f"record={o['record_id']} version={o.get('version') or 'NA'}" for o in options
            )
            warnings.warn(
                f"Multiple Zenodo candidates found for {yyyymm}; selecting "
                f"record={best['record_id']} version={best.get('version') or 'NA'}. "
                f"Candidates: {candidates_str}",
                RuntimeWarning,
            )

        return month_index

    def _ensure_zenodo_index_loaded(self) -> None:
        if self._month_index is not None:
            return
        self._zenodo_records = self._fetch_zenodo_records()
        self._month_index = self._build_month_index(self._zenodo_records)

    def _resolve_month_file_meta(self, yyyymm: str) -> Dict[str, Any]:
        """Resolve month metadata from Zenodo month index."""
        self._ensure_zenodo_index_loaded()
        assert self._month_index is not None

        if yyyymm not in self._month_index:
            raise KeyError(f"No Zenodo monthly file found for {yyyymm}")
        return self._month_index[yyyymm]

    @staticmethod
    def _file_checksum_ok(path: Path, checksum: Optional[str]) -> bool:
        if not checksum:
            return True

        algo, sep, expected = str(checksum).partition(":")
        if not sep or not expected:
            return True

        algo = algo.lower().strip()
        expected = expected.lower().strip()
        try:
            hasher = hashlib.new(algo)
        except ValueError:
            return True

        with path.open("rb") as fin:
            for chunk in iter(lambda: fin.read(1024 * 1024), b""):
                hasher.update(chunk)

        return hasher.hexdigest().lower() == expected

    @staticmethod
    def _file_size_ok(path: Path, size: Optional[int]) -> bool:
        if size is None:
            return True
        try:
            return path.stat().st_size == int(size)
        except (OSError, ValueError, TypeError):
            return False

    def _download_month_meta_to(self, file_meta: Dict[str, Any], output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = file_meta["filename"]
        url = file_meta["url"]
        target = output_dir / filename

        if target.exists():
            if self._file_size_ok(target, file_meta.get("size")) and self._file_checksum_ok(
                target, file_meta.get("checksum")
            ):
                return target
            target.unlink()

        self._download_file_with_retries(url=url, destination=target)

        if not self._file_size_ok(target, file_meta.get("size")):
            target.unlink(missing_ok=True)
            raise RuntimeError(f"Downloaded file size mismatch: {target}")

        if not self._file_checksum_ok(target, file_meta.get("checksum")):
            target.unlink(missing_ok=True)
            raise RuntimeError(f"Downloaded file checksum mismatch: {target}")

        return target

    def _get_or_download_month_file(self, yyyymm: str, target_dir: Optional[Path] = None) -> Optional[Path]:
        if self.local_mode:
            return self._local_month_index.get(yyyymm)

        out_dir = target_dir or self.cache_dir
        try:
            file_meta = self._resolve_month_file_meta(yyyymm)
        except KeyError:
            return None
        return self._download_month_meta_to(file_meta=file_meta, output_dir=out_dir)

    def query_date_range(
        self,
        start_date: Union[str, datetime, date, pd.Timestamp],
        end_date: Union[str, datetime, date, pd.Timestamp],
        output_dir: str = "./dms_downloads",
    ) -> List[str]:
        """
        Query and download monthly DMS files that overlap a date range.

        Args:
            start_date: Start date (inclusive).
            end_date: End date (inclusive).
            output_dir: Destination folder for monthly files.

        Returns:
            List of downloaded local file paths in ascending month order.
        """
        start = self._parse_date(start_date)
        end = self._parse_date(end_date)
        if start > end:
            raise ValueError(f"start_date must be <= end_date, got {start} > {end}")

        months = self._months_in_range(start, end)
        out_dir = Path(output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        downloaded: List[str] = []
        missing: List[str] = []

        if self.local_mode:
            for yyyymm in months:
                filename = f"DMS_{yyyymm}_4km_NA.nc"
                source = self._local_month_index.get(yyyymm)
                if source is None or not source.exists():
                    missing.append(yyyymm)
                    continue

                destination = out_dir / filename
                if source.resolve() != destination.resolve():
                    shutil.copy2(source, destination)
                downloaded.append(str(destination))

            if missing:
                raise FileNotFoundError(
                    "Missing local monthly DMS files for: " + ", ".join(missing)
                )

            return downloaded

        metas: List[Dict[str, Any]] = []
        for yyyymm in months:
            try:
                metas.append(self._resolve_month_file_meta(yyyymm))
            except KeyError:
                missing.append(yyyymm)

        if missing:
            raise FileNotFoundError(
                "Missing Zenodo monthly DMS files for: " + ", ".join(missing)
            )

        for meta in metas:
            path = self._download_month_meta_to(meta, out_dir)
            downloaded.append(str(path))

        return downloaded

    @staticmethod
    def _pick_latlon_names(ds: xr.Dataset) -> Tuple[Optional[str], Optional[str]]:
        lat_name = None
        lon_name = None

        for cand in ("latitude", "lat", "Latitude", "LAT"):
            if cand in ds.coords or cand in ds.variables:
                lat_name = cand
                break

        for cand in ("longitude", "lon", "Longitude", "LON"):
            if cand in ds.coords or cand in ds.variables:
                lon_name = cand
                break

        return lat_name, lon_name

    @staticmethod
    def _pick_latlon_names_from_keys(keys: Iterable[str]) -> Tuple[Optional[str], Optional[str]]:
        key_set = set(keys)
        lat_name = None
        lon_name = None

        for cand in ("latitude", "lat", "Latitude", "LAT"):
            if cand in key_set:
                lat_name = cand
                break

        for cand in ("longitude", "lon", "Longitude", "LON"):
            if cand in key_set:
                lon_name = cand
                break

        return lat_name, lon_name

    @staticmethod
    def _decode_time_values(values: np.ndarray, units: Optional[Union[str, bytes]]) -> pd.DatetimeIndex:
        if units is None:
            return pd.to_datetime(values, errors="coerce")

        if isinstance(units, (list, tuple, np.ndarray)):
            if len(units) == 0:
                return pd.to_datetime(values, errors="coerce")
            units = units[0]

        if isinstance(units, (bytes, np.bytes_)):
            units_str = bytes(units).decode("utf-8", errors="ignore")
        else:
            units_str = str(units)
        match = re.match(r"^\s*([A-Za-z]+)\s+since\s+(.+?)\s*$", units_str)
        if not match:
            return pd.to_datetime(values, errors="coerce")

        unit_name = match.group(1).lower()
        origin_str = match.group(2)
        origin = pd.to_datetime(origin_str, errors="coerce")
        if pd.isna(origin):
            return pd.to_datetime(values, errors="coerce")

        unit_map = {
            "day": "D",
            "days": "D",
            "hour": "h",
            "hours": "h",
            "minute": "m",
            "minutes": "m",
            "second": "s",
            "seconds": "s",
        }
        td_unit = unit_map.get(unit_name)
        if td_unit is None:
            return pd.to_datetime(values, errors="coerce")

        return origin + pd.to_timedelta(values, unit=td_unit)

    def _load_dms_day_from_month_h5py(
        self, target_date: date, month_file: Union[str, Path]
    ) -> Optional[pd.DataFrame]:
        try:
            import h5py
        except ImportError as exc:
            raise ImportError(
                "Neither xarray NetCDF backends nor h5py fallback is available. "
                "Install netCDF4 or h5netcdf (recommended), or install h5py."
            ) from exc

        month_file = Path(month_file)
        with h5py.File(month_file, "r") as h5:
            keys = list(h5.keys())
            if "DMS" not in keys:
                available = ", ".join(sorted(keys))
                raise ValueError(
                    f"File {month_file.name} does not contain dataset 'DMS'. Available: {available}"
                )
            if "time" not in keys:
                available = ", ".join(sorted(keys))
                raise ValueError(
                    f"File {month_file.name} does not contain dataset 'time'. Available: {available}"
                )

            lat_name, lon_name = self._pick_latlon_names_from_keys(keys)
            if not lat_name or not lon_name:
                available = ", ".join(sorted(keys))
                raise ValueError(
                    f"File {month_file.name} does not contain recognizable latitude/longitude datasets. "
                    f"Available: {available}"
                )

            raw_time = np.asarray(h5["time"][:])
            time_units = h5["time"].attrs.get("units")
            decoded_time = self._decode_time_values(raw_time, time_units)
            if np.all(pd.isna(decoded_time)):
                raise ValueError(f"Unable to decode time values in {month_file.name}")

            time_dates = np.array([t.date() if not pd.isna(t) else None for t in decoded_time])
            match_idx = np.where(time_dates == target_date)[0]
            if len(match_idx) == 0:
                return None

            day_values = np.asarray(h5["DMS"][int(match_idx[0]), ...])
            lat_arr = np.asarray(h5[lat_name][:])
            lon_arr = np.asarray(h5[lon_name][:])

            if lat_arr.ndim == 1 and lon_arr.ndim == 1 and day_values.ndim == 2:
                lat_grid, lon_grid = np.meshgrid(lat_arr, lon_arr, indexing="ij")
            elif lat_arr.ndim == 2 and lon_arr.ndim == 2 and day_values.ndim == 2:
                lat_grid, lon_grid = lat_arr, lon_arr
            else:
                raise ValueError(
                    f"Unsupported coordinate/value shapes in {month_file.name}: "
                    f"DMS={day_values.shape}, {lat_name}={lat_arr.shape}, {lon_name}={lon_arr.shape}"
                )

            day_df = pd.DataFrame(
                {
                    "latitude": lat_grid.reshape(-1),
                    "longitude": lon_grid.reshape(-1),
                    "DMS": day_values.reshape(-1),
                }
            )
            day_df = day_df.dropna(subset=["DMS"]).reset_index(drop=True)
            return day_df

    def _load_dms_day_from_month(self, target_date: date, month_file: Union[str, Path]) -> Optional[pd.DataFrame]:
        """
        Load daily DMS points from a monthly NetCDF file.

        Returns:
            DataFrame with columns: latitude, longitude, DMS (NaNs removed),
            or None if target date is not present in the file.
        """
        month_file = Path(month_file)
        try:
            with xr.open_dataset(month_file, decode_times=True) as ds:
                if "DMS" not in ds.variables:
                    available = ", ".join(sorted(ds.variables.keys()))
                    raise ValueError(
                        f"File {month_file.name} does not contain variable 'DMS'. Available: {available}"
                    )

                if "time" not in ds.coords and "time" not in ds.variables:
                    available = ", ".join(sorted(ds.variables.keys()))
                    raise ValueError(
                        f"File {month_file.name} does not contain coordinate/variable 'time'. Available: {available}"
                    )

                lat_name, lon_name = self._pick_latlon_names(ds)
                if not lat_name or not lon_name:
                    available = ", ".join(sorted(ds.variables.keys()))
                    raise ValueError(
                        f"File {month_file.name} does not contain recognizable latitude/longitude coordinates. "
                        f"Available variables: {available}"
                    )

                times = pd.to_datetime(ds["time"].values, errors="coerce")
                if np.all(pd.isna(times)):
                    raise ValueError(f"Unable to decode 'time' coordinate in {month_file.name}")

                time_dates = np.array([t.date() if not pd.isna(t) else None for t in times])
                match_idx = np.where(time_dates == target_date)[0]
                if len(match_idx) == 0:
                    return None

                day_slice = ds["DMS"].isel(time=int(match_idx[0]))
                day_df = day_slice.to_dataframe(name="DMS").reset_index()

                rename_map = {}
                if lat_name != "latitude":
                    rename_map[lat_name] = "latitude"
                if lon_name != "longitude":
                    rename_map[lon_name] = "longitude"
                if rename_map:
                    day_df = day_df.rename(columns=rename_map)

                missing_cols = [c for c in ("latitude", "longitude", "DMS") if c not in day_df.columns]
                if missing_cols:
                    available = ", ".join(day_df.columns)
                    raise ValueError(
                        f"Failed to build daily dataframe from {month_file.name}. Missing columns: {missing_cols}. "
                        f"Available: {available}"
                    )

                day_df = day_df[["latitude", "longitude", "DMS"]].dropna(subset=["DMS"]).reset_index(drop=True)
                return day_df
        except ValueError as exc:
            if "xarray's IO backends" not in str(exc):
                raise
            return self._load_dms_day_from_month_h5py(target_date=target_date, month_file=month_file)

    def _load_dms_day(self, target_date: date) -> Optional[pd.DataFrame]:
        yyyymm = f"{target_date.year:04d}{target_date.month:02d}"
        month_file = self._get_or_download_month_file(yyyymm, target_dir=self.cache_dir)
        if month_file is None or not month_file.exists():
            return None
        return self._load_dms_day_from_month(target_date=target_date, month_file=month_file)

    @staticmethod
    def _build_transformer_to_laea(proj_string: str):
        try:
            from pyproj import Transformer
        except ImportError as exc:
            raise ImportError(
                "pyproj is required for LAEA-based annotation. Install pyproj>=3.0.0."
            ) from exc

        return Transformer.from_crs("EPSG:4326", proj_string, always_xy=True)

    @staticmethod
    def _interpolate_to_dms(
        ocean_df: pd.DataFrame,
        var_name: str,
        dms_points: pd.DataFrame,
        transformer_to_proj,
        min_valid: int = 4,
        max_k: int = 36,
        power: float = 2.0,
        eps: float = 1e-6,
    ) -> np.ndarray:
        """
        LAEA-projected IDW interpolation with progressive neighbor search.

        This follows the logic in annotation pipeline.py:
        - Build KDTree in LAEA meters.
        - Query neighbors with progressive k list [4, 16, max_k].
        - Weight = 1/(d^power + eps).
        """
        if ocean_df.empty:
            return np.full(len(dms_points), np.nan)

        ocean_lon = ocean_df["longitude"].to_numpy(dtype=float)
        ocean_lat = ocean_df["latitude"].to_numpy(dtype=float)
        ocean_x, ocean_y = transformer_to_proj.transform(ocean_lon, ocean_lat)

        coords = np.c_[ocean_x, ocean_y]
        values = ocean_df[var_name].to_numpy(dtype=float)
        tree = cKDTree(coords)

        k_list = [4, 16, max_k] if max_k > 16 else [4, max_k]
        k_list_unique: List[int] = []
        for k in k_list:
            if k not in k_list_unique:
                k_list_unique.append(k)

        interpolated: List[float] = []

        for _, row in dms_points.iterrows():
            lon = float(row["longitude"]) if pd.notna(row["longitude"]) else np.nan
            lat = float(row["latitude"]) if pd.notna(row["latitude"]) else np.nan

            if np.isnan(lat) or np.isnan(lon):
                interpolated.append(np.nan)
                continue

            tx, ty = transformer_to_proj.transform(lon, lat)
            target = [tx, ty]

            interpolated_val = np.nan
            for k in k_list_unique:
                k = min(k, len(ocean_df))
                dists, idxs = tree.query(target, k=k)

                dists = np.atleast_1d(dists)
                idxs = np.atleast_1d(idxs)

                valid_mask = ~np.isnan(values[idxs])
                valid_dists = dists[valid_mask]
                valid_vals = values[idxs][valid_mask]

                if len(valid_vals) >= min_valid:
                    vd = valid_dists[:min_valid]
                    vv = valid_vals[:min_valid]
                    weights = 1.0 / (np.power(vd, power) + eps)
                    interpolated_val = float(np.sum(weights * vv) / np.sum(weights))
                    break

                elif len(valid_vals) > 0:
                    weights = 1.0 / (np.power(valid_dists, power) + eps)
                    interpolated_val = float(np.sum(weights * valid_vals) / np.sum(weights))
                    break

            interpolated.append(interpolated_val)

        return np.asarray(interpolated, dtype=float)

    def annotate_tracking_data(
        self,
        input_data: Union[str, pd.DataFrame],
        output_path: Optional[str] = None,
        datetime_col: str = "DateTime",
        lat_col: str = "Latitude",
        lon_col: str = "Longitude",
        min_valid: int = 4,
        max_k: int = 36,
        proj_string: str = DEFAULT_LAEA_PROJ,
        power: float = 2.0,
        eps: float = 1e-6,
        temporal_interp: bool = True,
    ) -> pd.DataFrame:
        """
        Annotate tracking data with daily DMS values.

        Args:
            input_data: CSV path or pandas DataFrame.
            output_path: Output CSV path for writing annotated rows.
            datetime_col: Datetime column name.
            lat_col: Latitude column name.
            lon_col: Longitude column name.
            min_valid: Minimum required valid neighbors for strict IDW branch.
            max_k: Maximum number of neighbors queried.
            proj_string: LAEA projection string for distance calculations.
            power: IDW power in 1/(d^power + eps).
            eps: Small positive value to avoid division by zero.
            temporal_interp: If True, perform linear temporal interpolation between
                previous/next 12:00 snapshots before combining final DMS.

        Returns:
            DataFrame with an added DMS column.
        """
        if isinstance(input_data, str):
            print(f"Loading tracking data from CSV: {input_data}")
            result_df = pd.read_csv(input_data)
        elif isinstance(input_data, pd.DataFrame):
            result_df = input_data.copy()
        else:
            raise TypeError("input_data must be a CSV path string or a pandas DataFrame")

        for col in (datetime_col, lat_col, lon_col):
            if col not in result_df.columns:
                raise ValueError(f"Required column '{col}' not found in input data")

        # Normalize all input timestamps to UTC, then drop tz info for UTC-naive output.
        # `format="mixed"` makes pandas robust to mixed timezone-aware and naive strings.
        try:
            result_df[datetime_col] = pd.to_datetime(
                result_df[datetime_col], errors="coerce", utc=True, format="mixed"
            )
        except TypeError:
            result_df[datetime_col] = pd.to_datetime(result_df[datetime_col], errors="coerce", utc=True)
        result_df[datetime_col] = result_df[datetime_col].dt.tz_convert("UTC").dt.tz_localize(None)
        invalid_datetime_count = int(result_df[datetime_col].isna().sum())
        if invalid_datetime_count > 0:
            warnings.warn(
                f"Found {invalid_datetime_count} rows with invalid {datetime_col}; DMS will remain NaN for those rows.",
                RuntimeWarning,
            )

        result_df["_lat"] = pd.to_numeric(result_df[lat_col], errors="coerce")
        result_df["_lon"] = pd.to_numeric(result_df[lon_col], errors="coerce")
        result_df["DMS"] = np.nan

        transformer_to_proj = self._build_transformer_to_laea(proj_string)

        valid_mask = (
            result_df[datetime_col].notna()
            & result_df["_lat"].notna()
            & result_df["_lon"].notna()
        )

        if temporal_interp:
            dt = result_df.loc[valid_mask, datetime_col]
            noon_same_day = dt.dt.normalize() + pd.Timedelta(hours=12)
            one_day = pd.Timedelta(days=1)

            lower_ts = noon_same_day.where(dt >= noon_same_day, noon_same_day - one_day)
            upper_ts = noon_same_day.where(dt <= noon_same_day, noon_same_day + one_day)
            same_ts = lower_ts == upper_ts

            total_seconds = (upper_ts - lower_ts).dt.total_seconds()
            elapsed_seconds = (dt - lower_ts).dt.total_seconds()
            w_upper = (elapsed_seconds / total_seconds).where(~same_ts, 0.0).clip(0.0, 1.0)
            w_lower = 1.0 - w_upper

            result_df["_lower_date"] = pd.NaT
            result_df["_upper_date"] = pd.NaT
            result_df["_w_lower"] = np.nan
            result_df["_w_upper"] = np.nan
            result_df["_dms_lower"] = np.nan
            result_df["_dms_upper"] = np.nan

            result_df.loc[valid_mask, "_lower_date"] = pd.to_datetime(lower_ts.dt.date)
            result_df.loc[valid_mask, "_upper_date"] = pd.to_datetime(upper_ts.dt.date)
            result_df.loc[valid_mask, "_w_lower"] = w_lower.to_numpy(dtype=float)
            result_df.loc[valid_mask, "_w_upper"] = w_upper.to_numpy(dtype=float)

            lower_dates = result_df.loc[valid_mask, "_lower_date"].dt.date.dropna().tolist()
            upper_dates = result_df.loc[valid_mask, "_upper_date"].dt.date.dropna().tolist()
            ref_dates = sorted(set(lower_dates) | set(upper_dates))

            print(
                f"Annotating tracking data for {int(valid_mask.sum())} points "
                f"using {len(ref_dates)} reference DMS days (temporal interpolation)"
            )

            missing_ref_dates: List[date] = []
            for ref_date in ref_dates:
                ref_needed_mask = valid_mask & (
                    (result_df["_lower_date"].dt.date == ref_date)
                    | (result_df["_upper_date"].dt.date == ref_date)
                )
                if not bool(ref_needed_mask.any()):
                    continue

                dms_df = self._load_dms_day(ref_date)
                if dms_df is None or dms_df.empty:
                    missing_ref_dates.append(ref_date)
                    continue

                ref_points = pd.DataFrame(
                    {
                        "latitude": result_df.loc[ref_needed_mask, "_lat"],
                        "longitude": result_df.loc[ref_needed_mask, "_lon"],
                    },
                    index=result_df.index[ref_needed_mask],
                )

                ref_values = self._interpolate_to_dms(
                    ocean_df=dms_df,
                    var_name="DMS",
                    dms_points=ref_points,
                    transformer_to_proj=transformer_to_proj,
                    min_valid=min_valid,
                    max_k=max_k,
                    power=power,
                    eps=eps,
                )
                ref_series = pd.Series(ref_values, index=ref_points.index, dtype=float)

                lower_assign_idx = result_df.index[
                    valid_mask & (result_df["_lower_date"].dt.date == ref_date)
                ]
                if len(lower_assign_idx) > 0:
                    result_df.loc[lower_assign_idx, "_dms_lower"] = ref_series.reindex(lower_assign_idx).to_numpy()

                upper_assign_idx = result_df.index[
                    valid_mask & (result_df["_upper_date"].dt.date == ref_date)
                ]
                if len(upper_assign_idx) > 0:
                    result_df.loc[upper_assign_idx, "_dms_upper"] = ref_series.reindex(upper_assign_idx).to_numpy()

            if missing_ref_dates:
                warnings.warn(
                    "No DMS data available for reference day(s): "
                    + ", ".join(sorted({d.isoformat() for d in missing_ref_dates}))
                    + ". Affected rows keep NaN or fallback to single-side reference.",
                    RuntimeWarning,
                )

            lower_ok = result_df["_dms_lower"].notna()
            upper_ok = result_df["_dms_upper"].notna()
            both_ok = lower_ok & upper_ok

            result_df.loc[both_ok, "DMS"] = (
                result_df.loc[both_ok, "_dms_lower"] * result_df.loc[both_ok, "_w_lower"]
                + result_df.loc[both_ok, "_dms_upper"] * result_df.loc[both_ok, "_w_upper"]
            )

            only_lower = lower_ok & ~upper_ok
            only_upper = upper_ok & ~lower_ok
            result_df.loc[only_lower, "DMS"] = result_df.loc[only_lower, "_dms_lower"]
            result_df.loc[only_upper, "DMS"] = result_df.loc[only_upper, "_dms_upper"]

            result_df = result_df.drop(
                columns=[
                    "_lat",
                    "_lon",
                    "_lower_date",
                    "_upper_date",
                    "_w_lower",
                    "_w_upper",
                    "_dms_lower",
                    "_dms_upper",
                ],
                errors="ignore",
            )
        else:
            valid_dates_series = result_df.loc[valid_mask, datetime_col].dt.date
            unique_dates = sorted(valid_dates_series.unique())

            print(f"Annotating tracking data for {len(unique_dates)} unique dates (spatial only)")

            for current_date in unique_dates:
                day_mask = valid_mask & (result_df[datetime_col].dt.date == current_date)
                day_points = pd.DataFrame(
                    {
                        "latitude": result_df.loc[day_mask, "_lat"],
                        "longitude": result_df.loc[day_mask, "_lon"],
                    },
                    index=result_df.index[day_mask],
                )

                dms_df = self._load_dms_day(current_date)
                if dms_df is None or dms_df.empty:
                    warnings.warn(
                        f"No DMS data available for {current_date}. Rows kept with DMS=NaN.",
                        RuntimeWarning,
                    )
                    continue

                interpolated = self._interpolate_to_dms(
                    ocean_df=dms_df,
                    var_name="DMS",
                    dms_points=day_points,
                    transformer_to_proj=transformer_to_proj,
                    min_valid=min_valid,
                    max_k=max_k,
                    power=power,
                    eps=eps,
                )

                result_df.loc[day_mask, "DMS"] = interpolated

            result_df = result_df.drop(columns=["_lat", "_lon"], errors="ignore")

        if output_path:
            output_fp = Path(output_path).expanduser().resolve()
            output_fp.parent.mkdir(parents=True, exist_ok=True)
            result_df.to_csv(output_fp, index=False)
            print(f"Saved annotated data to: {output_fp}")

        total = len(result_df)
        valid = int(result_df["DMS"].notna().sum())
        print(f"Annotation complete: {valid}/{total} rows have valid DMS values")

        return result_df
