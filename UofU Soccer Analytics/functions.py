# =========================
# Imports
# =========================
import pandas as pd
import numpy as np
import os
import glob
import re
import gzip
import xml.etree.ElementTree as ET

from pathlib import Path
from collections import defaultdict
from typing import Union, List, Optional, Dict

import matplotlib.pyplot as plt








# =========================
# Compute shotting chain function
# =========================
def compute_shot_chain_scores(
    df,
    k=4,
    weights: list = None,                    # e.g., [0.1, 0.1, 0.2, 0.5] (farthest -> shooter)
    terminal_action='Shot',
    team=None,                               # filter to a specific team; None = all teams
    unique_in_chain=True,                    # don't credit the same player twice within a chain
    eligible_types=None,                     # which actions count before the shot
    id_col='ID',
    team_col='Team',
    player_col='Player Code',
    action_col='Action Type',
    labels_col='Additional Labels',
    goal_shot_weight=0.7,                    # shooter weight to use when the shot is a Goal
    goal_weight_mode='override'              # 'override' (default) or 'bonus'
):
    """
    Aggregate weighted credits to the last `k` players in each shot chain (including the shooter).
    Returns a DataFrame with columns: [Team, Player, TotalScore].

    weights: list of length k, ordered from farthest in the chain to the shooter.
             If a chain has < k players, the *suffix* of weights is used so the shooter
             always aligns with the last weight.
    If the shot has 'Goal' in `labels_col`:
      - goal_weight_mode='override' -> replace shooter weight with `goal_shot_weight`
      - goal_weight_mode='bonus'    -> add `goal_shot_weight` to the shooter weight
    """
    # defaults
    if eligible_types is None:
        eligible_types = {'Touch', 'Pass', 'Duel', 'Free kick', 'Throw in',
       'Interception', 'Clearance', 'Goal kick', 'Corner',
       'Shot against', }
    if weights is None:
        weights = [0.1, 0.1, 0.2, 0.5]
    if len(weights) != k:
        raise ValueError(f"`weights` length ({len(weights)}) must equal k ({k}).")

    df = df.copy().sort_values(id_col).reset_index(drop=True)

    # shots (optionally by team)
    shot_mask = df[action_col].eq(terminal_action)
    if team is not None:
        shot_mask &= df[team_col].eq(team)
    shot_idxs = df.index[shot_mask].tolist()

    totals = defaultdict(float)

    for si in shot_idxs:
        shot_row  = df.loc[si]
        shot_team = shot_row[team_col]
        shooter   = shot_row[player_col]

        # build chain: start with shooter (end of chain)
        chain_players = [shooter]
        seen = {shooter} if unique_in_chain else set()

        # walk backward through same-team, eligible actions to gather up to k-1 more players
        j = si - 1
        while j >= 0 and len(chain_players) < k and df.at[j, team_col] == shot_team:
            if df.at[j, action_col] in eligible_types:
                p = df.at[j, player_col]
                if (not unique_in_chain) or (p not in seen):
                    chain_players.append(p)   # closest-first while walking backward
                    seen.add(p)
            j -= 1

        # farthest -> shooter to match weights
        chain_players = list(reversed(chain_players))
        m = len(chain_players)
        w = weights[-m:]  # keep shooter aligned to the last weight

        # goal-sensitive shooter weight
        labs = str(shot_row.get(labels_col) or '').lower()
        if 'goal' in labs:
            if goal_weight_mode == 'override':
                w[-1] = goal_shot_weight
            elif goal_weight_mode == 'bonus':
                w[-1] = w[-1] + goal_shot_weight

        # accumulate
        for player, weight in zip(chain_players, w):
            totals[(shot_team, player)] += weight

    out = (
        pd.DataFrame([(t, p, s) for (t, p), s in totals.items()],
                     columns=[team_col, player_col, 'TotalScore'])
        .sort_values([team_col, 'TotalScore'], ascending=[True, False])
        .reset_index(drop=True)
    )
    return out






# =========================
# Read player data file(s) function
# =========================
def parse_player_data_xml(
    folder: Union[str, Path],
    files: Union[str, List[str]],
    numeric_cols: Optional[List[str]] = None,
    combine: bool = True
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Parse one or many 'Player-based' soccer XML files into pandas DataFrame(s).
    - Tolerant to gzip, BOM/junk bytes, and encoding quirks
    - Skips bad files but processes the rest
    - Adds 'Source File' when combine=True
    """

    # ---- helpers ----
    ILLEGAL_XML_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]")

    def _safe_cast(val, typ):
        try:
            return typ(val) if val is not None else None
        except (ValueError, TypeError):
            return None

    def _read_xml_text_robust(path: Path) -> str:
        raw = path.read_bytes()
        # gzip magic
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        # decode (handle BOM with utf-8-sig)
        for enc in ("utf-8-sig", "utf-16", "cp1252"):
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            text = raw.decode("latin-1", errors="replace")
        # strip anything before first '<' and illegal control chars
        i = text.find("<")
        if i > 0:
            text = text[i:]
        return ILLEGAL_XML_RE.sub("", text)

    def _parse_one(path: Path) -> pd.DataFrame:
        # Load + parse root
        text = _read_xml_text_robust(path)
        try:
            root = ET.fromstring(text)
        except ET.ParseError as e:
            # HTML or non-XML? (don't crash—just skip)
            head = text[:80].strip().replace("\n", " ")
            raise ValueError(f"{path.name}: not well-formed XML ({e}); head starts: {head!r}")

        # Extract rows from <instance> nodes
        rows = []
        for inst in root.findall(".//instance"):
            row = {
                "ID": _safe_cast(inst.findtext("ID"), int),
                "Player Code": inst.findtext("code"),
                "Start Time": _safe_cast(inst.findtext("start"), float),
                "End Time": _safe_cast(inst.findtext("end"), float),
            }
            labels = [lab.findtext("text") for lab in inst.findall("label") if lab.findtext("text")]
            row["Action Type"] = labels[0] if labels else None
            row["Additional Labels"] = ", ".join(labels[1:]) if len(labels) > 1 else ""
            rows.append(row)

        df = pd.DataFrame(rows)
        if numeric_cols:
            for c in numeric_cols:
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.reset_index(drop=True)

    def _expand_paths(folder: Path, items: List[str]) -> List[Path]:
        out = []
        for it in items:
            if any(ch in it for ch in "*?[]"):            # glob support if you need it
                out.extend(sorted((folder).glob(it)))
            else:
                out.append((folder / it))
        # dedupe but keep order
        seen, uniq = set(), []
        for p in out:
            if p not in seen:
                seen.add(p); uniq.append(p)
        return uniq

    # ---- main ----
    folder = Path(folder)
    if isinstance(files, str):
        files = [files]
    paths = _expand_paths(folder, files)
    if not paths:
        raise FileNotFoundError(f"No files matched {files} in {folder}")

    if combine:
        frames, skipped = [], []
        for p in paths:
            try:
                if not p.exists():
                    skipped.append(f"{p.name}: missing file")
                    continue
                df = _parse_one(p)
                if df.empty:
                    skipped.append(f"{p.name}: parsed but no <instance> rows")
                    continue
                df.insert(0, "Source File", p.name)
                frames.append(df)
            except Exception as e:
                skipped.append(str(e))
        if skipped:
            print("⚠️ Skipped/issue files:\n - " + "\n - ".join(skipped))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
            columns=["Source File","ID","Player Code","Start Time","End Time","Action Type","Additional Labels"]
        )
    else:
        out, skipped = {}, []
        for p in paths:
            try:
                if not p.exists():
                    skipped.append(f"{p.name}: missing file"); continue
                df = _parse_one(p)
                out[p.name] = df
            except Exception as e:
                skipped.append(str(e))
        if skipped:
            print("⚠️ Skipped/issue files:\n - " + "\n - ".join(skipped))
        return out









# =========================
# Function adding appropriate team name to 
# =========================
def add_team_name(df, first_team, first_team_players,
                  second_team=None, second_team_players=None,
                  player_code="Player Code"):
    mapping = {p.strip(): first_team for p in (first_team_players or [])}
    if second_team and second_team_players:
        mapping.update({p.strip(): second_team for p in second_team_players})
    df = df.copy()
    df[player_code] = df[player_code].astype(str).str.strip()
    df["Team"] = df[player_code].map(mapping)
    return df








# =========================
# Team data xml function
# =========================
def parse_team_data_xml(
    folder: Union[str, os.PathLike],
    files: Union[str, List[str]],
    numeric_cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Parse one or multiple 'Team-based' soccer XML files into a pandas DataFrame.

    Parameters
    ----------
    folder : str | PathLike
        Directory containing the XML file(s).
    files : str | List[str]
        - Single filename (e.g., "match.xml")
        - List of filenames (["a.xml", "b.xml"])
        - Or glob pattern(s) (e.g., "*.xml" or ["Utah*.xml", "Cal*.xml"])
    numeric_cols : List[str], optional
        Extra columns to coerce to numeric (errors='coerce'). Common
        team-based coords are auto-included if present.

    Returns
    -------
    pd.DataFrame
        Combined rows from all files with columns:
        ['match_name', 'ID', 'code', 'start', 'end', labels...]
        If a label group appears multiple times, values are joined with ", ".
    """
    # Resolve list of paths from files arg
    if isinstance(files, str):
        files = [files]

    paths: List[str] = []
    for pat in files:
        full_pat = os.path.join(folder, pat)
        matches = glob.glob(full_pat)
        if not matches and os.path.exists(full_pat):
            matches = [full_pat]
        paths.extend(matches)

    # Deduplicate while preserving order
    seen = set()
    uniq_paths = []
    for p in paths:
        if p not in seen:
            uniq_paths.append(p)
            seen.add(p)

    if not uniq_paths:
        raise FileNotFoundError("No XML files matched. Check folder and filenames/globs.")

    rows = []

    for file in uniq_paths:
        try:
            tree = ET.parse(file)
            root = tree.getroot()
        except Exception as e:
            print(f"[WARN] Skipping '{file}' (parse error: {e})")
            continue

        match_name = os.path.splitext(os.path.basename(file))[0]

        all_instances = root.find("ALL_INSTANCES")
        if all_instances is None:
            print(f"[WARN] '{file}' has no <ALL_INSTANCES>; skipping.")
            continue

        for inst in all_instances.findall("instance"):
            row = {
                "match_name": match_name,
                "ID": (inst.findtext("ID") or "").strip(),
                "code": (inst.findtext("code") or "").strip(),
                "start": (inst.findtext("start") or "").strip(),
                "end": (inst.findtext("end") or "").strip(),
            }

            # Collect all labels; if the same group appears multiple times, join texts
            group_values = defaultdict(list)
            for label in inst.findall("label"):
                group = (label.findtext("group") or "").strip()
                text = (label.findtext("text") or "").strip()
                if not group:
                    continue
                if text and text not in group_values[group]:
                    group_values[group].append(text)

            # Flatten label groups into comma-separated strings
            for g, vals in group_values.items():
                row[g] = ", ".join(vals)

            rows.append(row)

    df = pd.DataFrame(rows).reset_index(drop=True)

    # Convert likely numeric columns if they exist
    default_numeric = [
        "11 - Start location X",
        "12 - Start location Y",
        "13 - Duration",
        "15 - End location X",
        "16 - End location Y",
        "start",
        "end",
    ]
    if numeric_cols:
        default_numeric.extend(numeric_cols)

    for col in set(default_numeric):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df








# =========================
# Keeper distribution
# =========================
def plot_keeper_dist(df, col='10 - Players', exclude=None):
    exclude_norm = {x.casefold() for x in (exclude or [])}
    s = (df[col].str.split(',').explode().dropna().str.strip())
    s = s[s.ne('')]
    if exclude_norm:
        s = s[~s.str.casefold().isin(exclude_norm)]
    ax = s.value_counts().plot(kind='bar', figsize=(10,5))
    ax.set_title('Keepers distribution to player frequency')
    plt.tight_layout(); plt.show()
    return ax