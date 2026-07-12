# -*- coding: utf-8 -*-
"""Apoio geografico e demografico da analise estadual.

- Malha municipal da Bahia (GeoJSON) via API de malhas do IBGE, cacheada em
  ``out/malha_ba.json`` (uma unica requisicao; sem dependencia de geopandas).
- Populacao por municipio a partir do registro de entes do proprio
  siconfi-collector (``data/entities.json``), que traz a estimativa do IBGE.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
REPO = HERE.parents[1]                          # .../final-paperz
ENTITIES = REPO / "siconfi-collector" / "data" / "entities.json"

MALHA_URL = ("https://servicodados.ibge.gov.br/api/v3/malhas/estados/29"
             "?formato=application/vnd.geo+json&qualidade=intermediaria"
             "&intrarregiao=municipio")


def load_malha() -> dict:
    """GeoJSON da malha municipal da Bahia (cache em disco)."""
    cache = OUT / "malha_ba.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    import requests
    resp = requests.get(MALHA_URL, timeout=120)
    resp.raise_for_status()
    geo = resp.json()
    OUT.mkdir(exist_ok=True)
    cache.write_text(json.dumps(geo), encoding="utf-8")
    return geo


def load_population() -> pd.DataFrame:
    """(cod_ibge, nome, populacao) para os municipios da Bahia."""
    data = json.loads(ENTITIES.read_text(encoding="utf-8"))
    ba = [e for e in data if e.get("uf") == "BA" and e.get("sphere") == "M"]
    return pd.DataFrame([{"cod_ibge": int(e["cod_ibge"]), "nome": e["name"],
                          "populacao": int(e["population"])} for e in ba])


def feature_polygons(geo: dict) -> dict[int, list]:
    """Mapeia cod_ibge -> lista de aneis exteriores [(x, y), ...] da malha."""
    polys: dict[int, list] = {}
    for feat in geo["features"]:
        cod = int(feat["properties"]["codarea"])
        geom = feat["geometry"]
        rings = []
        if geom["type"] == "Polygon":
            rings.append(geom["coordinates"][0])
        elif geom["type"] == "MultiPolygon":
            rings.extend(part[0] for part in geom["coordinates"])
        polys[cod] = rings
    return polys
