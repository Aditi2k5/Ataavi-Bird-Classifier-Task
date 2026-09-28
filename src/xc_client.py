from __future__ import annotations

import time
from pathlib import Path
from urllib.parse import quote

import requests

API_BASE = "https://xeno-canto.org/api/3/recordings"
UA = {"User-Agent": "birdcall-classifier/1.0 (research take-home)"}


class XenoCanto:
    def __init__(self, api_key: str, per_page: int = 100, sleep: float = 1.0):
        self.key = api_key
        self.per_page = per_page
        self.sleep = sleep

    def search(self, query: str) -> list:
        recs, page, num_pages = [], 1, 1
        while page <= num_pages:
            url = (f"{API_BASE}?query={quote(query)}"
                   f"&per_page={self.per_page}&page={page}&key={self.key}")
            r = requests.get(url, headers=UA, timeout=60)
            if r.status_code == 401:
                raise SystemExit("401 unauthorized - check your XENO_CANTO_API_KEY.")
            r.raise_for_status()
            data = r.json()
            num_pages = int(data.get("numPages", 1))
            recs.extend(data.get("recordings", []))
            page += 1
            time.sleep(self.sleep)
        return recs

    def download(self, rec: dict, dest: Path) -> bool:
        file_url = rec.get("file") or f"https://xeno-canto.org/{rec['id']}/download"
        if "key=" not in file_url:
            file_url += ("&" if "?" in file_url else "?") + f"key={self.key}"
        try:
            with requests.get(file_url, headers=UA, timeout=180, stream=True) as resp:
                resp.raise_for_status()
                ctype = resp.headers.get("Content-Type", "").lower()
                if "audio" not in ctype and "octet-stream" not in ctype:
                    print(f"    ! XC{rec.get('id')}: not audio (Content-Type={ctype!r}) - skipping")
                    return False
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(dest, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 15):
                        fh.write(chunk)
            with open(dest, "rb") as fh:
                head = fh.read(16).lstrip()
            if head[:1] in (b"<", b"{") or head[:5].lower() == b"<!doc":
                print(f"    ! XC{rec.get('id')}: got HTML/text, not audio - discarding")
                dest.unlink(missing_ok=True)
                return False
            return dest.stat().st_size > 2048
        except Exception as e:
            print(f"    ! download failed XC{rec.get('id')}: {e}")
            if dest.exists():
                dest.unlink(missing_ok=True)
            return False


def build_query(sp: dict, q: dict) -> str:
    parts = [f'gen:"{sp["gen"]}"', f'sp:"{sp["sp"]}"', "grp:birds"]
    if q.get("length_seconds"):
        parts.append(f'len:{q["length_seconds"]}')
    if q.get("country"):
        parts.append(f'cnt:"{q["country"]}"')
    if q.get("quality"):
        parts.append(f'q:{q["quality"]}')
    return " ".join(parts)


def is_overlap(rec: dict) -> bool:
    also = rec.get("also") or []
    return isinstance(also, list) and any(str(a).strip() for a in also)
