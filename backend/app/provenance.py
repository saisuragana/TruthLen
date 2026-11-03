from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _normalise_state(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        value = value.get("state") or value.get("validation_state")
    if value is None:
        return None
    value = getattr(value, "value", None) or getattr(value, "name", None) or str(value)
    state = str(value).rsplit(".", 1)[-1].strip().lower()
    return state if state in {"trusted", "valid", "invalid"} else None


def _no_credentials() -> dict[str, Any]:
    return {
        "status": "no_credentials",
        "label": "No C2PA credentials found",
        "validation_state": None,
        "summary": (
            "No C2PA Content Credentials were found. This is common for ordinary camera files "
            "and does not show whether the media is AI-generated or camera-captured."
        ),
    }


def inspect_provenance(path: Path) -> dict[str, Any]:
    """Read C2PA provenance when the optional SDK is installed.

    C2PA describes a media file's recorded provenance. It cannot establish that
    the depicted scene is true or, by itself, classify the file as AI-generated.
    """
    try:
        from c2pa import Reader
    except ImportError:
        return {
            "status": "sdk_missing",
            "label": "Provenance checker not installed",
            "validation_state": None,
            "summary": (
                "The optional C2PA checker is not installed, so no provenance check was made. "
                "This does not change the detector result."
            ),
        }

    try:
        with Reader(str(path)) as reader:
            store = json.loads(reader.json())
            state = None
            for method_name in ("validation_state", "get_validation_state"):
                accessor = getattr(reader, method_name, None)
                if callable(accessor):
                    try:
                        state = _normalise_state(accessor())
                    except Exception:
                        state = None
                    if state:
                        break
    except Exception as exc:
        # The SDK uses ManifestNotFound when an asset has no embedded C2PA data.
        # Other errors may mean an unsupported format, malformed file, or SDK issue.
        if "manifestnotfound" in str(exc).replace("_", "").lower():
            return _no_credentials()
        return {
            "status": "unavailable",
            "label": "Provenance check unavailable",
            "validation_state": None,
            "summary": (
                "The C2PA checker could not read this file or its format. No provenance conclusion "
                "was made, and this does not mean the file is AI-generated."
            ),
        }

    if not isinstance(store, dict) or not store.get("manifests"):
        return _no_credentials()

    if state is None:
        state = _normalise_state(store.get("validation_state"))
    if state is None:
        active_id = store.get("active_manifest")
        manifests = store.get("manifests", {})
        active = manifests.get(active_id, {}) if isinstance(manifests, dict) else {}
        if isinstance(active, dict):
            state = _normalise_state(active.get("validation_state"))

    summaries = {
        "trusted": (
            "A C2PA credential was found and marked trusted under the SDK's current trust policy. "
            "That supports the credential's provenance claim; it does not prove the depicted scene "
            "is true or settle whether the media is AI-generated."
        ),
        "valid": (
            "A C2PA credential was found and validated, but this result does not say its signer is "
            "trusted. A valid provenance record does not prove the depicted scene is true or settle "
            "whether the media is AI-generated."
        ),
        "invalid": (
            "A C2PA credential was found, but the SDK reported validation problems. Treat its "
            "provenance claim as unreliable; this alone does not classify the media as AI-generated."
        ),
        None: (
            "A C2PA credential was found, but the SDK did not provide a validation state. Its "
            "provenance could not be confirmed, and it does not settle whether the media is AI-generated."
        ),
    }
    labels = {
        "trusted": "C2PA credential trusted",
        "valid": "C2PA credential valid",
        "invalid": "C2PA validation problem",
        None: "C2PA credential not verified",
    }
    return {
        "status": state or "unverified",
        "label": labels[state],
        "validation_state": state,
        "summary": summaries[state],
    }
