"""Lightweight in-memory vector store / RAG for climate metadata.

The knowledge base contains short documents about the demo datasets, region
definitions, ETCCDI extreme indices, and the CMIP6 model used.  Retrieval uses
TF-IDF + cosine similarity, so it has no extra dependencies beyond scikit-learn.
"""

from __future__ import annotations

import json
from pathlib import Path

import xarray as xr


_BASE_DIR = Path(__file__).resolve().parents[2]
_DATA_DIR = _BASE_DIR / "data" / "demo"


def _xarray_attr_to_text(path: Path, short_id: str) -> str:
    """Summarise a NetCDF file's dimensions and variables as text."""
    try:
        ds = xr.open_dataset(path, decode_times=False)
        dims = ", ".join(f"{k}={v}" for k, v in ds.sizes.items())
        data_vars = ", ".join(ds.data_vars)
        # Try a few attribute keys that often describe the dataset
        attrs = {k: str(v) for k, v in ds.attrs.items() if k.lower() in (
            "title", "source", "institution", "model_id", "experiment", "frequency",
            "variable", "description", "history",
        )}
        ds.close()
        return (
            f"Dataset '{short_id}' ({path.name}): "
            f"data_vars=[{data_vars}], dims={{{dims}}}, "
            f"attrs={json.dumps(attrs)}."
        )
    except Exception as exc:
        return f"Dataset '{short_id}' ({path.name}): could not read attributes ({exc})."


_REGION_DOCS = [
    {
        "id": "region_south_asia",
        "category": "region",
        "text": (
            "Region 'south_asia' covers 60-100°E and 5-35°N. "
            "Demo file: cpc_south_asia_2013.nc (daily, 2013, ~0.5°). "
            "Best for extreme-index maps (RX1day, RX5day, R95p)."
        ),
    },
    {
        "id": "region_europe",
        "category": "region",
        "text": (
            "Region 'europe' covers -15-45°E and 35-75°N. "
            "Demo file: cpc_europe_2013.nc (daily, 2013, ~0.5°). "
            "Best for extreme-index maps (RX1day, RX5day, R95p)."
        ),
    },
    {
        "id": "region_germany",
        "category": "region",
        "text": (
            "Region 'germany' is a 20-year (1995-2014) monthly gridded box "
            "covering 5.5-15.5°E and 47.0-55.5°N. Demo files: "
            "cpc_germany_1995_2014_monthly.nc (CPC observed) and "
            "cmip6_germany_1995_2014_monthly.nc (CMIP6 MPI-ESM1-2-HR model). "
            "Best for 20-year trends, climatology, bias maps and point comparisons. "
            "This is the only region with a multi-year record."
        ),
    },
    {
        "id": "region_global",
        "category": "region",
        "text": (
            "Global domain is available for monthly 2013 fields: "
            "cmip6_global_monthly_2013.nc and cpc_global_monthly_2013_regridded.nc. "
            "Best for global mean precipitation maps and model-vs-reference bias maps. "
            "These are already on a common ~100 km grid for easy comparison."
        ),
    },
]

_ETCCDI_DOCS = [
    {
        "id": "index_rx1day",
        "category": "etccdi",
        "text": (
            "RX1day (ETCCDI index): maximum 1-day precipitation amount (mm) "
            "within the selected period. It captures the single wettest day and "
            "is sensitive to short-duration, high-intensity precipitation events."
        ),
    },
    {
        "id": "index_rx5day",
        "category": "etccdi",
        "text": (
            "RX5day (ETCCDI index): maximum 5-day consecutive precipitation "
            "amount (mm) within the selected period. It captures multi-day "
            "rainfall accumulations such as monsoon bursts or persistent storms."
        ),
    },
    {
        "id": "index_r95p",
        "category": "etccdi",
            "text": (
                "R95p (ETCCDI index): total precipitation (mm) from days when "
                "daily precipitation exceeds the 95th percentile of wet days "
                "(>1 mm). It measures the contribution of heavy-rain days to "
                "total rainfall."
            ),
    },
    {
        "id": "index_prcptot",
        "category": "etccdi",
        "text": (
            "PRCPTOT (ETCCDI index): total precipitation from wet days "
            "(daily precipitation >= 1 mm) in the selected period, in mm. "
            "It is the wet-day contribution to total annual or seasonal rainfall."
        ),
    },
    {
        "id": "index_rx1day_usage",
        "category": "etccdi_usage",
        "text": (
            "RX1day is calculated from the daily precipitation file "
            "(cpc_south_asia_2013.nc, cpc_europe_2013.nc) using a temporal "
            "maximum over the time dimension. It produces a 2D spatial map."
        ),
    },
    {
        "id": "index_r95p_usage",
        "category": "etccdi_usage",
        "text": (
            "R95p requires daily precipitation data and is only meaningful for "
            "regions and years with daily data (currently 2013: South Asia, Europe). "
            "The percentile is computed from all wet days in the record."
        ),
    },
]

_MODES_DOCS = [
    {
        "id": "nao_limitation",
        "category": "teleconnection",
        "text": (
            "The North Atlantic Oscillation (NAO) is a large-scale climate "
            "teleconnection based on sea-level pressure differences between the "
            "Azores High and the Icelandic Low. This demo does NOT include an NAO "
            "index, sea-level pressure, or any other large-scale atmospheric "
            "circulation data. It only contains precipitation. Therefore the NAO "
            "effect on German precipitation cannot be computed from the demo "
            "datasets. Only descriptive or general-knowledge answers are possible."
        ),
    },
    {
        "id": "modes_limitation",
        "category": "teleconnection",
        "text": (
            "Modes of climate variability such as NAO, ENSO, AO (Arctic "
            "Oscillation), PDO and AMO are not part of the demo datasets. Only "
            "precipitation is provided, so correlations, regressions or composites "
            "with these indices cannot be computed unless the user supplies the "
            "index data. Questions about these modes should be answered from the "
            "knowledge base or by explaining the data limitation."
        ),
    },
]

_CMIP6_DOCS = [
    {
        "id": "cmip6_mpi",
        "category": "cmip6",
        "text": (
            "CMIP6 model 'MPI-ESM1-2-HR' (Max Planck Institute for Meteorology). "
            "Resolution ~100 km (1.0° atmosphere), historical experiment used in "
            "this demo. Available as cmip6_global_monthly_2013.nc (monthly, 2013), "
            "cmip6_germany_1995_2014_monthly.nc (monthly, 20-year) and "
            "cmip6_europe_2013.nc / cmip6_south_asia_2013.nc where bundled. "
            "Compared against CPC gauge-based observations for bias analysis."
        ),
    },
    {
        "id": "cmip6_bias",
        "category": "cmip6",
        "text": (
            "CMIP6 bias analysis is done by interpolating the model field onto "
            "the CPC observation grid and subtracting: CMIP6 minus CPC. "
            "Germany and the global 2013 monthly files are already on a common "
            "grid and can be compared directly."
        ),
    },
]


def _build_documents() -> list[dict]:
    """Assemble the full set of metadata documents."""
    docs = []
    docs.extend(_REGION_DOCS)
    docs.extend(_ETCCDI_DOCS)
    docs.extend(_MODES_DOCS)
    docs.extend(_CMIP6_DOCS)

    # Add dataset-specific documents by reading NetCDF attributes
    dataset_files = {
        "cpc_south_asia_2013": _DATA_DIR / "cpc_south_asia_2013.nc",
        "cpc_europe_2013": _DATA_DIR / "cpc_europe_2013.nc",
        "cpc_global_monthly_2013_regridded": _DATA_DIR / "cpc_global_monthly_2013_regridded.nc",
        "cpc_germany_1995_2014_monthly": _DATA_DIR / "cpc_germany_1995_2014_monthly.nc",
        "cmip6_global_monthly_2013": _DATA_DIR / "cmip6_global_monthly_2013.nc",
        "cmip6_germany_1995_2014_monthly": _DATA_DIR / "cmip6_germany_1995_2014_monthly.nc",
    }
    for short_id, path in dataset_files.items():
        if path.exists():
            docs.append({
                "id": f"dataset_{short_id}",
                "category": "dataset",
                "text": _xarray_attr_to_text(path, short_id),
            })
            # Add a human-readable data-availability sentence
            if "daily" in short_id or "south_asia" in short_id or "europe" in short_id:
                docs.append({
                    "id": f"dataset_usage_{short_id}",
                    "category": "dataset_usage",
                    "text": (
                        f"The {short_id} dataset contains DAILY precipitation data "
                        f"for 2013 and is suitable for ETCCDI extreme indices "
                        f"(RX1day, RX5day, R95p)."
                    ),
                })
            elif "monthly" in short_id:
                docs.append({
                    "id": f"dataset_usage_{short_id}",
                    "category": "dataset_usage",
                    "text": (
                        f"The {short_id} dataset contains MONTHLY precipitation data. "
                        f"Germany monthly files cover 1995-2014 and are suitable for "
                        f"trends, climatology and bias maps."
                    ),
                })

    # Add a catch-all temporal/spatial limitation doc
    docs.append({
        "id": "data_limitations",
        "category": "limitations",
        "text": (
            "Data limitations: only 2013 has full daily global/regional coverage "
            "(South Asia, Europe) for extreme indices. Only Germany has a multi-year "
            "(1995-2014) monthly record, so trend and climatology questions are "
            "only reliable for Germany. The global monthly 2013 files are on a "
            "common ~100 km grid for bias maps."
        ),
    })

    return docs


class _VectorStore:
    """Tiny in-memory TF-IDF vector store."""

    def __init__(self, documents: list[dict] | None = None):
        self._docs = documents or _build_documents()
        self._vectorizer = None
        self._matrix = None

    def fit(self) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        self._vectorizer = TfidfVectorizer(stop_words="english")
        texts = [d["text"] for d in self._docs]
        self._matrix = self._vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        if self._matrix is None:
            self.fit()

        from sklearn.metrics.pairwise import cosine_similarity

        qvec = self._vectorizer.transform([query])
        sims = cosine_similarity(qvec, self._matrix).ravel()
        top_idx = sims.argsort()[::-1][:top_k]
        return [
            {
                "id": self._docs[i]["id"],
                "category": self._docs[i].get("category", ""),
                "text": self._docs[i]["text"],
                "score": round(float(sims[i]), 4),
            }
            for i in top_idx
        ]


# Shared, lazily-fitted vector store and a lookup of all documents
_VECTOR_STORE = _VectorStore()
_ALL_DOCUMENTS = _build_documents()


def _get_doc_by_id(doc_id: str) -> dict | None:
    """Return a document by id, or None if not found."""
    for doc in _ALL_DOCUMENTS:
        if doc["id"] == doc_id:
            return doc
    return None


def answer_climate_question(query: str) -> str:
    """Answer a factual climate metadata question using the embedded knowledge base.

    Use this for questions like "what is RX1day?", "which dataset has daily
    data?" or "which region has multi-year data?". The tool retrieves the
    most relevant metadata documents and returns the answer.

    Parameters
    ----------
    query
        The user's metadata or definition question.

    Returns
    -------
    str
        A concise answer backed by retrieved documents.
    """
    hits = _VECTOR_STORE.search(query, top_k=3)

    # Special-case exact ETCCDI index names for an immediate, reliable answer
    q = query.lower()
    exact_matches = {
        "rx1day": "index_rx1day",
        "rx5day": "index_rx5day",
        "r95p": "index_r95p",
        "prcptot": "index_prcptot",
    }
    for key, doc_id in exact_matches.items():
        if key in q:
            doc = _get_doc_by_id(doc_id)
            if doc is not None:
                return doc["text"]

    if not hits:
        return "I could not find any relevant information in the climate metadata store."

    # Fallback: return the top hits as a short answer
    parts = ["Relevant information from the climate metadata store:\n"]
    for h in hits:
        parts.append(f"- [{h['category']}] {h['text']}")

    parts.append(
        "\nUse the returned context to answer the user. If the question asks "
        "about an index, dataset, region or model, quote the relevant line above."
    )
    return "\n".join(parts)


if __name__ == "__main__":
    for q in [
        "what is RX1day?",
        "which dataset has daily data?",
        "which region has multi-year data?",
    ]:
        print(f"Q: {q}")
        print(answer_climate_question(q))
        print()
