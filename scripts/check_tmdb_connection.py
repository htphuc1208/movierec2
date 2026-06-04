#!/usr/bin/env python3
"""Check local connectivity to TMDb without printing secrets."""

from __future__ import annotations

import argparse
import socket

import requests

from recommender.config import get_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose TMDb connectivity from this machine")
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--base-url", default=None)
    return parser.parse_args()


def show_dns(host: str) -> None:
    print(f"DNS for {host}:")
    try:
        addresses = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        print(f"  DNS error: {exc}")
        return
    seen: set[str] = set()
    for family, _, _, _, sockaddr in addresses:
        ip = sockaddr[0]
        if ip in seen:
            continue
        seen.add(ip)
        family_name = "IPv6" if family == socket.AF_INET6 else "IPv4"
        print(f"  {family_name} {ip}")


def try_get(label: str, url: str, timeout: float, params: dict | None = None) -> None:
    print(f"\n{label}: {url}")
    try:
        response = requests.get(
            url,
            params=params,
            timeout=timeout,
            headers={"Accept": "application/json", "User-Agent": "movierec3/0.1 connectivity-check"},
        )
        print(f"  status={response.status_code}")
        if response.status_code >= 400:
            print(f"  body={response.text[:200]}")
    except requests.RequestException as exc:
        print(f"  error={type(exc).__name__}: {exc}")


def main() -> None:
    args = parse_args()
    settings = get_settings()
    base_url = (args.base_url or settings.tmdb_base_url).rstrip("/")
    api_key = settings.tmdb_api_key

    show_dns("api.themoviedb.org")
    try_get("TMDb website", "https://www.themoviedb.org", args.timeout)
    try_get("TMDb API no-auth probe", f"{base_url}/configuration", args.timeout)

    if api_key:
        try_get("TMDb API auth probe", f"{base_url}/authentication", args.timeout, params={"api_key": api_key})
    else:
        print("\nTMDb API auth probe skipped: TMDB_API_KEY is missing")


if __name__ == "__main__":
    main()
