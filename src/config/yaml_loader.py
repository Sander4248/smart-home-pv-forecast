from __future__ import annotations

"""
Robust YAML configuration loader for RunCfg.

This module provides:
- Safe YAML loading with schema validation against the RunCfg dataclass tree.
- Deep-merge behavior for incomplete configs with a required base.yaml.
- A *special policy* for the 'features' block:
    • If 'features' is absent: copy features from base.yaml (base is required).
    • If 'features' is present: DO NOT deep-merge with base; treat as full override.
      Missing keys among {weather, solar_pos, timestamp} become empty lists [].
    • If all three lists end up empty: warn and fall back to base.yaml (base required).
      (This covers cases like 'features:', 'features: {}', or keys present but empty.)
    • If a key is written without a list (e.g., 'solar_pos:'), it is treated as [].
    • If a scalar is provided by mistake, it is coerced into a single-item list.

Public API:
- load_run_cfgs(yaml_paths, base_path) -> List[RunCfg]
"""

import logging
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Type, get_origin, get_args
import copy
import yaml

from src.config.schema import (RunCfg)


# ---------- Utils ----------

def _normalize_features_block(d: Dict[str, Any] | None,
                              base_features: Dict[str, Any] | None,
                              source_label: str,
                              base_path: Path,
                              ) -> Dict[str, Any]:
    """
    Normalize and finalize the 'features' block according to special rules.

    The 'features' block must ultimately have the structure:
        {
            "weather": [...],
            "solar_pos": [...],
            "timestamp": [...],
        }

    Policy:
    - If 'd' (the current YAML's 'features') is None:
        -> Copy features from 'base_features'. 'base_features' MUST be available.
    - If 'd' exists:
        -> NO deep-merge with base. Treat as override.
        -> Any missing key among {weather, solar_pos, timestamp} becomes [].
        -> Keys written without a list (YAML 'key:') become [].
        -> Scalar values are coerced to single-item lists.
        -> If all three lists are empty: warn and fall back to 'base_features'.
           'base_features' MUST be available for this fallback.

    :param d: The 'features' mapping from the current YAML (may be None).
    :param base_features: The 'features' mapping from base.yaml (may be None).
    :param source_label: A label or path of the current YAML (used in messages).
    :param base_path: Filesystem path to base.yaml (used in messages).
    :return: A complete, normalized 'features' dict.
    """

    # Helper to normalize values to list
    def _as_list(x: Any) -> List[Any]:
        if x is None:
            return []
        if isinstance(x, list):
            return x
        # Tolerate accidental scalars by wrapping into a single-item list
        return [x]

    # 1) features fully missing -> must use base.features (base is mandatory here)
    if d is None:
        if base_features is None:
            logging.critical(f"'features' is missing in '{source_label}', and base.yaml ('{base_path}') "
                             f"is not available to provide defaults. Aborting.")
            exit(1)

        return {
            "weather": list(base_features.get("weather", [])),
            "solar_pos": list(base_features.get("solar_pos", [])),
            "timestamp": list(base_features.get("timestamp", []))
        }

    # 2) features present -> NO deep merge; explicitly fill missing keys with []
    weather = _as_list(d.get("weather"))
    solar_pos = _as_list(d.get("solar_pos"))
    timestamp = _as_list(d.get("timestamp"))

    all_empty = (len(weather) == 0 and len(solar_pos) == 0 and len(timestamp) == 0)

    # 3) If *all* are empty -> warn and fall back to base.features (base mandatory)
    if all_empty:
        if base_features is None:
            logging.critical(f"In '{source_label}', no features were provided (all empty), "
                             f"and base.yaml ('{base_path}') is not available to fall back to. Aborting.")
            exit(1)

        logging.warning(f"In '{source_label}', no features are defined. Falling back to features from base.yaml.")

        return {
            "weather": list(base_features.get("weather", [])),
            "solar_pos": list(base_features.get("solar_pos", [])),
            "timestamp": list(base_features.get("timestamp", []))
        }

    # 4) Partially present -> return as-is with missing keys normalized to []
    return {
        "weather": weather,
        "solar_pos": solar_pos,
        "timestamp": timestamp,
    }


def _deep_merge(a: Dict[str, Any],
                b: Dict[str, Any]
                ) -> Dict[str, Any]:
    """
    Deep-merge dict 'b' into a deep-copied dict of 'a'.

    Dict values are merged recursively; non-dict values overwrite.

    :param a: The base dictionary.
    :param b: The overlay dictionary.
    :return: A new merged dictionary (does not mutate inputs).
    """
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml(path: Path) -> Dict[str, Any]:
    """
    Load a YAML file as a top-level mapping.

    :param path: Filesystem path to the YAML file.
    :return: A dict (mapping) representing the YAML contents (empty dict if YAML is empty).
    """
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML must be a mapping at top-level: {path}")
    return data


# ---------- Completeness check (recursive for dataclasses) ----------
def _dict_covers_dataclass(d: Dict[str, Any],
                           cls: Type
                           ) -> bool:
    """
    Check recursively whether dict 'd' covers *all* fields required by dataclass 'cls'.

    Rules:
    - For primitive fields: only presence is checked (value may be None).
    - For nested dataclasses: 'd[field]' must be a dict that itself covers the sub-dataclass.
    - For List[SubDataclass]: value must be a list; each element must cover SubDataclass.
    - For other List[...] of primitives: value must be a list.

    :param d: The candidate dict.
    :param cls: The dataclass type to check.
    :return: True if 'd' contains all fields (recursively), else False.
    """
    if not is_dataclass(cls):
        return True
    for f in fields(cls):
        if f.name not in d:
            return False
        v = d[f.name]
        ftype = f.type
        origin = get_origin(ftype)

        # List[...]?
        if origin is list or origin is List:
            (elem_type,) = get_args(ftype) or (Any,)
            if v is None or not isinstance(v, list):
                return False

            # If the list holds dataclasses, check each element
            if is_dataclass(elem_type):
                for item in v:
                    if not isinstance(item, dict):
                        return False
                    if not _dict_covers_dataclass(item, elem_type):
                        return False

        #  Nested dataclass?
        elif is_dataclass(ftype):
            if not isinstance(v, dict):
                return False
            if not _dict_covers_dataclass(v, ftype):
                return False
        else:
            # Primitive: presence is enough (allow None). Tighten here if desired.
            pass
    return True


def _is_full_config(d: Dict[str, Any]) -> bool:
    """
    Determine whether 'd' constitutes a *full* RunCfg configuration.

    :param d: The candidate dict.
    :return: True if 'd' covers all fields of RunCfg (recursively), else False.
    """
    return _dict_covers_dataclass(d, RunCfg)


# ---------- Dict -> Dataclasses (rekursiv) ----------
def _coerce(d: Dict[str, Any],
            cls: Type
            ) -> Any:
    """
    Convert dict 'd' into an instance of dataclass 'cls' (recursively).

    Behavior:
    - For nested dataclasses: recurse into dict values.
    - For List[SubDataclass]: recurse for each list element.
    - For List[primitive]: shallow 'list(val)'.
    - For primitive fields: assign value as-is.

    :param d: The source dict (must contain all required fields for 'cls').
    :param cls: The target dataclass type.
    :return: An instance of 'cls'.
    """
    if not is_dataclass(cls):
        return d
    kwargs = {}
    for f in fields(cls):
        name, ftype = f.name, f.type
        if name not in d:
            raise ValueError(f"Missing required field '{name}' for {cls.__name__}")
        val = d[name]
        origin = get_origin(ftype)

        if origin is list or origin is List:
            (elem_type,) = get_args(ftype) or (Any,)
            if is_dataclass(elem_type):
                kwargs[name] = [_coerce(x, elem_type) for x in val]
            else:
                kwargs[name] = list(val)
        elif is_dataclass(ftype):
            kwargs[name] = _coerce(val, ftype)
        else:
            kwargs[name] = val
    return cls(**kwargs)


def load_run_cfgs(
        yaml_paths: Optional[Iterable[str]] = None,
        base_path: str = "configs/base.yaml",
) -> List[RunCfg]:
    """
    Load one or more RunCfg configurations from YAML files with base.yaml support.

    Modes:
    1) No 'yaml_paths' provided:
       - Load and validate 'base.yaml' (must be a *full* config).
       - Return [RunCfg(base)] as the only configuration.

    2) One or more YAML paths provided:
       For each path:
         a) If the YAML is a *full* config:
            - Apply the special 'features' policy (no deep-merge; see '_normalize_features_block').
            - Convert to RunCfg and append to results.
         b) If the YAML is *incomplete*:
            - Load and validate base.yaml (required).
            - Deep-merge YAML into base (EXCEPT 'features', which follow special policy).
            - Validate merged dict is full.
            - Convert to RunCfg and append.

    Special 'features' handling:
    - When 'features' is absent in a YAML, base.yaml *must* exist and provides the features.
    - When 'features' is present, it replaces base features without deep-merge.
    - If all features lists are empty, we warn and fall back to base features (base required).

    :param yaml_paths: Iterable of filesystem paths (str) to YAML config files. If None or empty,
                       only base.yaml is loaded.
    :param base_path: Filesystem path (str) to base.yaml. Defaults to "configs/base.yaml".
    :return: A list of RunCfg instances, one per resolved configuration.
    """
    base_file = Path(base_path)

    # Helper: load & validate base.yaml as a *full* RunCfg dict
    def _load_base() -> Dict[str, Any]:
        if not base_file.exists():
            logging.critical(f"Expected base.yaml at '{base_file}', but it was not found.")
            exit(1)

        base_dict = _load_yaml(base_file)
        if not _is_full_config(base_dict):
            logging.critical(f"base.yaml ('{base_file}') is incomplete. It must contain all fields of RunCfg.")
            exit(1)

        return base_dict

    # Case 1: No YAML paths -> load base.yaml as-is (features 1:1 from base) ---
    if not yaml_paths:
        base_dict = _load_base()
        return [_coerce(base_dict, RunCfg)]

    # Case 2: YAML paths available
    results: List[RunCfg] = []
    for p in yaml_paths:
        cfg_path = Path(p)
        d = _load_yaml(cfg_path)

        # We may need base.features depending on how 'features' appears in 'd'
        base_dict_for_features: Dict[str, Any] | None = None
        base_features_block: Dict[str, Any] | None = None

        if "features" not in d:
            # 'features' missing -> base.features needed
            base_dict_for_features = _load_base()
            base_features_block = base_dict_for_features.get("features")
        else:
            # 'features' present; if they end up all-empty, '_normalize_features_block'
            # will require base. We load it opportunistically when available
            if not base_file.exists():
                # base existiert evtl. nicht; das ist ok, solange die features NICHT „alle leer“ sind.
                base_dict_for_features = None
                base_features_block = None
            else:
                # Laden wir opportunistisch für den Fallback
                base_dict_for_features = _load_yaml(base_file)
                base_features_block = base_dict_for_features.get("features")

        # 1) Full config?
        if _is_full_config(d):
            # ACHTUNG: Sonderregel Features auch hier anwenden
            d["features"] = _normalize_features_block(
                d.get("features"),
                base_features_block,
                source_label=str(cfg_path),
                base_path=base_file,
            )
            results.append(_coerce(d, RunCfg))
            continue

        # 2) Incomplete config -> base.yaml is mandatory
        base_dict = _load_base()

        # Deep-merge everything EXCEPT 'features' (handled by special policy below)
        merged = _deep_merge(base_dict, d)

        # Finalize 'features' according to the policy (no deep-merge with base)
        merged["features"] = _normalize_features_block(
            d.get("features"),  # only the YAML-provided features (or None)
            base_dict.get("features"),  # base features for fallback
            source_label=str(cfg_path),
            base_path=base_file,
        )

        # Safety: ensure the merged dict is a *full* RunCfg before coercion
        if not _is_full_config(merged):
            logging.critical(f"After merging base.yaml with '{cfg_path}', some fields are still missing. "
                             f"Please verify the keys in '{cfg_path}'.")
            exit(1)

        results.append(_coerce(merged, RunCfg))

    return results
