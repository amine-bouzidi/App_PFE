"""
emotional_intensity.py — Phase 5.2 : Intensite Emotionnelle
Plateforme d'analyse textuelle multimodale — McDonald's Due Diligence

Deux outils selon la source :
  - VADER    (vaderSentiment) -> Twitter : concu pour le langage social media
  - TextBlob (textblob)       -> Presse  : adapte au langage formel, + subjectivite

Trois signaux bruts universels (sans bibliotheque) :
  - Ratio majuscules        : mots entierement en MAJ / total mots
  - Ratio ponctuation forte : ! et ? / total ponctuation
  - Ratio mots forts        : mots d'indignation/colere/urgence / total mots

Detection automatique d'anomalies :
  - Z-score par metrique sur toute la periode
  - Seuil configurable (defaut : 2.0)
  - Aucune connaissance a priori des evenements
  - Les pics detectes remplacent les dates hardcodees sur les graphiques

Livrable :
  - emotional_intensity_results.json   : resultats doc-level + agregations + anomalies
  - cognitive_indicators_summary.csv   : tableau complet 5.1 + 5.2 par periode x source
  - 4 visualisations matplotlib

Installation :
    pip install vaderSentiment textblob

Usage :
    from emotional_intensity import run_emotional_analysis
    run_emotional_analysis(
        press_path="press_scrapper.json",
        twitter_path="twitter_scraper.json",
        linguistic_json="linguistic_metrics_results.json",
        output_dir="outputs/reports"
    )
"""

import os
import re
import json
import csv
import warnings
from collections import defaultdict
from statistics import mean, stdev
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────

# Lexique d'intensite emotionnelle multilingue EN/FR/DE/ES
INTENSITY_LEXICON = frozenset({
    # Anglais — indignation / colere
    "disgusting", "terrible", "horrible", "awful", "outrageous", "unacceptable",
    "shameful", "disgraceful", "appalling", "atrocious", "abysmal", "pathetic",
    "ridiculous", "absurd", "scandalous", "corrupt", "criminal", "fraud",
    "liar", "liars", "lying", "cheat", "cheating", "scam", "scammer",
    # Anglais — urgence / appel a l'action
    "boycott", "ban", "stop", "quit", "never", "enough", "done",
    "demand", "demands", "protest", "strike", "fire", "fired", "resign",
    "accountability", "justice", "expose", "exposed",
    # Anglais — danger / sante
    "sick", "poison", "poisoned", "contaminated", "contamination", "outbreak",
    "dangerous", "unsafe", "deadly", "death", "died", "dying", "kill",
    "toxic", "hazard", "hazardous", "warning", "alert", "recall",
    # Francais
    "scandaleux", "honteux", "inacceptable", "inadmissible", "catastrophique",
    "boycotter", "arnaque", "escroquerie", "corruption", "honte",
    "intoxication", "dangereux", "alerte", "rappel",
    # Allemand
    "skandal", "schande", "unakzeptabel", "boykott", "betrug", "vergiftung",
    "gefahrlich", "warnung", "ruckruf",
    # Espagnol
    "escandaloso", "vergonzoso", "inaceptable", "boicot", "fraude",
    "contaminacion", "peligroso", "retirada",
})

COLORS = {
    "bg":        "#f8f9fa",
    "text":      "#1a1a1a",
    "primary":   "#0066cc",
    "secondary": "#e63946",
    "tertiary":  "#2dc653",
    "accent":    "#f4a261",
    "anomaly":   "#cc0000",
    "grid":      "#cccccc",
}

# Instance VADER globale — chargee une seule fois au demarrage
_VADER = SentimentIntensityAnalyzer()


# ─────────────────────────────────────────────────────────────────
# 1. SIGNAUX BRUTS UNIVERSELS
# ─────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    """Supprime URLs et mentions, conserve ponctuation et majuscules."""
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[@#]\w+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def ratio_caps(text: str) -> float:
    """
    Proportion de mots entierement en majuscules / total mots.
    Signal d'indignation ou d'urgence sur Twitter.
    Ex : 'McDonald is DISGUSTING and TERRIBLE' -> 2/5 = 0.40
    """
    words = re.findall(r"[A-Za-z]{2,}", text)
    if not words:
        return 0.0
    return round(sum(1 for w in words if w.isupper()) / len(words), 4)


def ratio_strong_punct(text: str) -> float:
    """
    Proportion de ! et ? dans la ponctuation totale.
    Signal d'intensite emotionnelle brute.
    """
    all_punct = re.findall(r"[!?.,;:]", text)
    if not all_punct:
        return 0.0
    return round(sum(1 for p in all_punct if p in "!?") / len(all_punct), 4)


def ratio_intensity_words(text: str) -> float:
    """Proportion de mots du lexique d'intensite / total mots."""
    words = re.findall(r"[a-zA-Z\xc0-\xff]{2,}", text.lower())
    if not words:
        return 0.0
    return round(sum(1 for w in words if w in INTENSITY_LEXICON) / len(words), 4)


def compute_raw_signals(text: str) -> Dict:
    raw = _clean(text)
    return {
        "ratio_caps":            ratio_caps(raw),
        "ratio_strong_punct":    ratio_strong_punct(raw),
        "ratio_intensity_words": ratio_intensity_words(raw),
    }


# ─────────────────────────────────────────────────────────────────
# 2. VADER — TWITTER
# ─────────────────────────────────────────────────────────────────

def analyze_vader(text: str) -> Dict:
    """
    Analyse VADER pour les tweets.
    vader_compound : score synthetique -1 (tres negatif) a +1 (tres positif).
    """
    clean = _clean(text)
    if not clean:
        return {"vader_neg": None, "vader_neu": None,
                "vader_pos": None, "vader_compound": None}
    s = _VADER.polarity_scores(clean)
    return {
        "vader_neg":      round(s["neg"], 4),
        "vader_neu":      round(s["neu"], 4),
        "vader_pos":      round(s["pos"], 4),
        "vader_compound": round(s["compound"], 4),
    }


# ─────────────────────────────────────────────────────────────────
# 3. TEXTBLOB — PRESSE
# ─────────────────────────────────────────────────────────────────

def analyze_textblob(text: str) -> Dict:
    """
    Analyse TextBlob pour la presse.
    tb_polarity    : -1 (negatif) a +1 (positif)
    tb_subjectivity: 0 (factuel) a 1 (tres subjectif)
    """
    clean = _clean(text)
    if not clean:
        return {"tb_polarity": None, "tb_subjectivity": None}
    try:
        blob = TextBlob(clean)
        return {
            "tb_polarity":     round(blob.sentiment.polarity, 4),
            "tb_subjectivity": round(blob.sentiment.subjectivity, 4),
        }
    except Exception:
        return {"tb_polarity": None, "tb_subjectivity": None}


# ─────────────────────────────────────────────────────────────────
# 4. ANALYSE DOCUMENT COMPLET
# ─────────────────────────────────────────────────────────────────

def compute_emotional_metrics(doc: Dict) -> Dict:
    """
    Calcule tous les indicateurs emotionnels pour un document.
    VADER si source='twitter', TextBlob si source='press'.
    Signaux bruts calcules sur toutes les sources.
    """
    text   = doc.get("text", "")
    source = doc.get("source", "")

    meta = {
        "doc_id":   doc.get("id"),
        "date":     doc.get("date", ""),
        "source":   source,
        "platform": doc.get("platform", ""),
    }

    if not text or not text.strip():
        return {**meta,
                "ratio_caps": 0.0, "ratio_strong_punct": 0.0,
                "ratio_intensity_words": 0.0,
                "vader_neg": None, "vader_neu": None,
                "vader_pos": None, "vader_compound": None,
                "tb_polarity": None, "tb_subjectivity": None}

    metrics = compute_raw_signals(text)

    if source == "twitter":
        metrics.update(analyze_vader(text))
        metrics["tb_polarity"]     = None
        metrics["tb_subjectivity"] = None
    elif source == "press":
        metrics["vader_neg"]      = None
        metrics["vader_neu"]      = None
        metrics["vader_pos"]      = None
        metrics["vader_compound"] = None
        metrics.update(analyze_textblob(text))
    else:
        metrics.update(analyze_vader(text))
        metrics.update(analyze_textblob(text))

    return {**meta, **metrics}


# ─────────────────────────────────────────────────────────────────
# 5. AGREGATION PAR PERIODE x SOURCE
# ─────────────────────────────────────────────────────────────────

_EMOTIONAL_KEYS = [
    "ratio_caps", "ratio_strong_punct", "ratio_intensity_words",
    "vader_neg", "vader_neu", "vader_pos", "vader_compound",
    "tb_polarity", "tb_subjectivity",
]


def aggregate_emotional(
    doc_metrics: List[Dict],
    source_filter: Optional[str] = None,
) -> Dict[str, Dict]:
    """Agrege les metriques emotionnelles par mois YYYY-MM."""
    groups = defaultdict(list)
    for m in doc_metrics:
        if source_filter and m.get("source") != source_filter:
            continue
        groups[m.get("date", "unknown")].append(m)

    result = {}
    for period, docs in sorted(groups.items()):
        agg: Dict = {"n_docs": len(docs)}
        for key in _EMOTIONAL_KEYS:
            vals = [
                d[key] for d in docs
                if d.get(key) is not None and isinstance(d.get(key), (int, float))
            ]
            agg[key] = round(mean(vals), 4) if vals else None
        result[period] = agg
    return result


def aggregate_emotional_all_sources(doc_metrics: List[Dict]) -> Dict:
    return {
        "all":     aggregate_emotional(doc_metrics),
        "press":   aggregate_emotional(doc_metrics, source_filter="press"),
        "twitter": aggregate_emotional(doc_metrics, source_filter="twitter"),
    }


# ─────────────────────────────────────────────────────────────────
# 6. DETECTION AUTOMATIQUE D'ANOMALIES (Z-SCORE)
# ─────────────────────────────────────────────────────────────────

def detect_anomalies(
    agg: Dict[str, Dict],
    key: str,
    threshold: float = 2.0,
    direction: str = "both",
) -> Dict[str, Dict]:
    """
    Detecte automatiquement les periodes anormales via z-score.

    Parametres :
      agg       : dict {periode -> {metrique -> valeur}} (agregation mensuelle)
      key       : metrique a analyser (ex: 'vader_compound')
      threshold : seuil z-score pour declarer une anomalie (defaut 2.0)
      direction : 'high'  -> pics positifs seulement (ex: ratio_caps monte)
                  'low'   -> pics negatifs seulement (ex: compound chute)
                  'both'  -> les deux sens

    Retourne :
      dict {periode -> {'zscore': float, 'value': float, 'direction': str}}
      uniquement pour les periodes anormales.

    Principe :
      On ne dit pas "il y a un scandale en 2024-07".
      On dit "le z-score de vader_compound en 2024-07 est -2.8,
      ce qui depasse le seuil de 2.0 sigma".
      C'est la donnee qui parle, pas nous.
    """
    # Extraire les valeurs non nulles
    periods = sorted(agg.keys())
    values  = [(p, agg[p][key]) for p in periods
               if agg[p].get(key) is not None]

    if len(values) < 3:
        return {}

    vals_only = [v for _, v in values]
    mu  = mean(vals_only)
    sig = stdev(vals_only)

    if sig == 0:
        return {}

    anomalies = {}
    for period, val in values:
        z = (val - mu) / sig
        is_high = z >= threshold
        is_low  = z <= -threshold

        if direction == "high" and not is_high:
            continue
        if direction == "low" and not is_low:
            continue
        if direction == "both" and not (is_high or is_low):
            continue

        anomalies[period] = {
            "zscore":    round(z, 2),
            "value":     round(val, 4),
            "mean":      round(mu, 4),
            "std":       round(sig, 4),
            "direction": "high" if z > 0 else "low",
        }

    return anomalies


def detect_all_anomalies(
    emot_agg: Dict[str, Dict[str, Dict]],
    threshold: float = 2.0,
) -> Dict[str, Dict]:
    """
    Lance la detection d'anomalies sur toutes les metriques pertinentes.

    Metriques surveillees et direction :
      vader_compound    (twitter) -> 'low'  : chute = sentiment negatif collectif
      vader_neg         (twitter) -> 'high' : hausse = negativite croissante
      ratio_intensity_words (all) -> 'high' : hausse = vocabulaire de crise
      ratio_strong_punct    (all) -> 'high' : hausse = emotivite expressionnelle
      ratio_caps            (all) -> 'high' : hausse = indignation brute
      tb_polarity          (press)-> 'low'  : chute = presse plus negative
      tb_subjectivity      (press)-> 'high' : hausse = presse moins neutre

    Retourne un dict structure :
      {
        "vader_compound_twitter": {periode -> anomaly_info},
        "ratio_intensity_words_all": {...},
        ...
      }
    """
    checks = [
        ("vader_compound",        "twitter", "low"),
        ("vader_neg",             "twitter", "high"),
        ("ratio_intensity_words", "all",     "high"),
        ("ratio_strong_punct",    "all",     "high"),
        ("ratio_caps",            "all",     "high"),
        ("tb_polarity",           "press",   "low"),
        ("tb_subjectivity",       "press",   "high"),
    ]

    results = {}
    for key, source, direction in checks:
        agg = emot_agg.get(source, {})
        anomalies = detect_anomalies(agg, key, threshold, direction)
        label = f"{key}_{source}"
        results[label] = anomalies

    return results


def build_anomaly_summary(all_anomalies: Dict[str, Dict]) -> Dict[str, Dict]:
    """
    Consolide toutes les anomalies par periode.
    Une periode est "critique" si elle est detectee par plusieurs metriques.

    Retourne :
      {periode -> {"signals": [liste metriques], "max_zscore": float,
                   "n_signals": int, "severity": str}}
    """
    period_signals = defaultdict(list)

    for metric_label, anomalies in all_anomalies.items():
        for period, info in anomalies.items():
            period_signals[period].append({
                "metric":    metric_label,
                "zscore":    info["zscore"],
                "value":     info["value"],
                "direction": info["direction"],
            })

    summary = {}
    for period, signals in sorted(period_signals.items()):
        n = len(signals)
        max_z = max(abs(s["zscore"]) for s in signals)
        severity = "critique" if n >= 3 else "modere" if n == 2 else "faible"
        summary[period] = {
            "signals":    signals,
            "n_signals":  n,
            "max_zscore": round(max_z, 2),
            "severity":   severity,
        }

    return summary


def format_anomaly_labels(anomaly_summary: Dict[str, Dict]) -> Dict[str, str]:
    """
    Convertit le resume d'anomalies en labels lisibles pour les graphiques.
    Format : {periode -> "label_court"}
    Label = severity + z-score max
    """
    labels = {}
    for period, info in anomaly_summary.items():
        z    = info["max_zscore"]
        sev  = info["severity"]
        n    = info["n_signals"]
        labels[period] = f"[{sev}] z={z} ({n} sig.)"
    return labels


# ─────────────────────────────────────────────────────────────────
# 7. TABLEAU COMPLET 5.1 + 5.2
# ─────────────────────────────────────────────────────────────────

def build_summary_table(
    linguistic_agg: Dict[str, Dict[str, Dict]],
    emotional_agg:  Dict[str, Dict[str, Dict]],
    anomaly_summary: Dict[str, Dict],
) -> List[Dict]:
    """
    Fusionne agregations 5.1 + 5.2 + anomalies en tableau plat.
    Une ligne par (periode, source).
    """
    ling_cols = [
        "n_docs", "n_tokens",
        "ttr", "mattr", "mtld",
        "hapax_ratio", "yule_k",
        "avg_sent_length", "clause_density",
        "flesch_reading_ease", "lix",
        "information_density", "lexical_density",
        "topic_concentration",
    ]
    emot_cols = [
        "ratio_caps", "ratio_strong_punct", "ratio_intensity_words",
        "vader_compound", "vader_neg", "vader_pos",
        "tb_polarity", "tb_subjectivity",
    ]

    all_periods = sorted(set(
        list(linguistic_agg.get("all", {}).keys()) +
        list(emotional_agg.get("all", {}).keys())
    ))

    rows = []
    for source in ["all", "press", "twitter"]:
        ling_src = linguistic_agg.get(source, {})
        emot_src = emotional_agg.get(source, {})

        for period in all_periods:
            row = {"periode": period, "source": source}

            for col in ling_cols:
                row[col] = ling_src.get(period, {}).get(col)
            for col in emot_cols:
                row[col] = emot_src.get(period, {}).get(col)

            # Colonnes anomalie (basees sur source "all" et "twitter")
            anom = anomaly_summary.get(period, {})
            row["n_anomaly_signals"] = anom.get("n_signals", 0)
            row["max_zscore"]        = anom.get("max_zscore")
            row["severity"]          = anom.get("severity", "")

            rows.append(row)

    return rows


def display_summary_table(rows: List[Dict], source_filter: str = "all") -> None:
    """
    Affiche le tableau de synthese dans le terminal.
    Les periodes anormales sont marquees d'un indicateur visuel.
    """
    filtered = [r for r in rows if r["source"] == source_filter]
    if not filtered:
        print(f"Aucune donnee pour source='{source_filter}'")
        return

    sep = "-" * 155
    print(f"\n{'=' * 155}")
    print(f"  TABLEAU COMPLET INDICATEURS COGNITIFS — Source : {source_filter.upper()}")
    print(f"  (* = anomalie detectee automatiquement par z-score)")
    print(f"{'=' * 155}")

    header = (
        f"{'Periode':<10} {'Flag':^5} {'N_docs':>6} "
        f"{'MATTR':>6} {'MTLD':>7} {'YuleK':>7} "
        f"{'AvgSL':>6} {'Flesch':>7} {'LIX':>6} "
        f"{'InfoD':>6} "
        f"{'Caps':>6} {'SPunc':>6} {'IntW':>6} "
        f"{'VComp':>7} {'VNeg':>6} "
        f"{'TBPol':>7} {'TBSub':>6} "
        f"{'NAnom':>6} {'MaxZ':>6} {'Sev':>8}"
    )
    print(header)
    print(sep)

    def fmt(val, w=6, d=3):
        if val is None:
            return f"{'—':>{w}}"
        return f"{val:>{w}.{d}f}"

    for r in filtered:
        n_sig = r.get("n_anomaly_signals", 0)
        flag  = " *** " if n_sig >= 3 else " **  " if n_sig == 2 else " *   " if n_sig == 1 else "     "
        sev   = r.get("severity", "")[:8] if r.get("severity") else "—"

        line = (
            f"{r['periode']:<10} {flag} "
            f"{str(r.get('n_docs') or '—'):>6} "
            f"{fmt(r.get('mattr'), 6, 3)} "
            f"{fmt(r.get('mtld'), 7, 1)} "
            f"{fmt(r.get('yule_k'), 7, 1)} "
            f"{fmt(r.get('avg_sent_length'), 6, 1)} "
            f"{fmt(r.get('flesch_reading_ease'), 7, 1)} "
            f"{fmt(r.get('lix'), 6, 1)} "
            f"{fmt(r.get('information_density'), 6, 3)} "
            f"{fmt(r.get('ratio_caps'), 6, 3)} "
            f"{fmt(r.get('ratio_strong_punct'), 6, 3)} "
            f"{fmt(r.get('ratio_intensity_words'), 6, 3)} "
            f"{fmt(r.get('vader_compound'), 7, 3)} "
            f"{fmt(r.get('vader_neg'), 6, 3)} "
            f"{fmt(r.get('tb_polarity'), 7, 3)} "
            f"{fmt(r.get('tb_subjectivity'), 6, 3)} "
            f"{n_sig:>6} "
            f"{fmt(r.get('max_zscore'), 6, 2)} "
            f"{sev:>8}"
        )
        print(line)

    print(f"{'=' * 155}\n")


def export_summary_csv(rows: List[Dict], output_path: str) -> None:
    """Exporte le tableau complet en CSV."""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    size_kb = os.path.getsize(output_path) // 1024
    print(f"  OK {os.path.basename(output_path)} ({len(rows)} lignes, ~{size_kb} KB)")


# ─────────────────────────────────────────────────────────────────
# 8. VISUALISATIONS (4 graphiques)
# ─────────────────────────────────────────────────────────────────

def _to_dt(period: str) -> datetime:
    return datetime.strptime(period + "-01", "%Y-%m-%d")


def _series(agg: Dict, key: str) -> Tuple[List[datetime], List[float]]:
    periods = sorted(agg.keys())
    dts, vals = [], []
    for p in periods:
        v = agg[p].get(key)
        if v is not None:
            dts.append(_to_dt(p))
            vals.append(v)
    return dts, vals


def _add_anomaly_lines(
    ax: plt.Axes,
    anomaly_labels: Dict[str, str],
    anomaly_summary: Dict[str, Dict],
) -> None:
    """
    Ajoute les lignes verticales des anomalies detectees automatiquement.
    Couleur et epaisseur varient selon la severite.
    """
    ylim = ax.get_ylim()
    y_top = ylim[1]

    severity_style = {
        "critique": ("#cc0000", 1.5, 0.8),
        "modere":   ("#ff6600", 1.0, 0.6),
        "faible":   ("#ffaa00", 0.7, 0.4),
    }

    for period, info in anomaly_summary.items():
        sev = info.get("severity", "faible")
        color, lw, alpha = severity_style.get(sev, ("#999999", 0.7, 0.4))
        dt = _to_dt(period)
        z  = info["max_zscore"]
        n  = info["n_signals"]

        ax.axvline(dt, color=color, ls="--", lw=lw, alpha=alpha, zorder=1)
        ax.text(dt, y_top * 0.97,
                f"z={z}\n({n}sig)",
                rotation=90, fontsize=5, color=color,
                va="top", ha="right", zorder=2)


def _style(ax: plt.Axes, title: str, ylabel: str) -> None:
    ax.set_facecolor(COLORS["bg"])
    ax.set_title(title, fontsize=10, fontweight="bold", pad=5)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(axis="y", color=COLORS["grid"], alpha=0.45, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right", fontsize=7)


def _save(fig: plt.Figure, path: Optional[str], show: bool) -> None:
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  OK {os.path.basename(path)}")
    if show:
        plt.show()
    plt.close(fig)


def plot_vader_twitter(
    agg_twitter: Dict,
    anomaly_summary: Dict,
    output_path=None, show=False,
) -> plt.Figure:
    """VIZ 1 — Sentiment VADER Twitter + anomalies detectees."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.patch.set_facecolor(COLORS["bg"])
    fig.suptitle("Sentiment Twitter — VADER (anomalies detectees automatiquement)",
                 fontsize=12, fontweight="bold")

    periods = sorted(agg_twitter.keys())

    ax = axes[0]
    dts, vals = _series(agg_twitter, "vader_compound")
    if dts:
        ax.plot(dts, vals, color=COLORS["text"], lw=2, marker="o",
                markersize=3, label="Compound (-1 a +1)")
        ax.fill_between(dts, vals, 0,
                        where=[v < 0 for v in vals],
                        color=COLORS["secondary"], alpha=0.2, label="Zone negative")
        ax.fill_between(dts, vals, 0,
                        where=[v >= 0 for v in vals],
                        color=COLORS["tertiary"], alpha=0.2, label="Zone positive")
        ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.6)
    _style(ax, "Score compound VADER (Twitter)", "Score (-1 a +1)")
    _add_anomaly_lines(ax, {}, anomaly_summary)
    ax.legend(fontsize=8)

    ax = axes[1]
    for key, label, color in [
        ("vader_neg", "Negatif", COLORS["secondary"]),
        ("vader_pos", "Positif", COLORS["tertiary"]),
        ("vader_neu", "Neutre",  "#aaaaaa"),
    ]:
        dts, vals = _series(agg_twitter, key)
        if dts:
            ax.plot(dts, vals, color=color, lw=1.5, marker="o",
                    markersize=3, label=label, alpha=0.85)
    _style(ax, "Decomposition neg / neu / pos (Twitter)", "Proportion (0-1)")
    _add_anomaly_lines(ax, {}, anomaly_summary)
    ax.legend(fontsize=8)

    _save(fig, output_path, show)
    return fig


def plot_textblob_press(
    agg_press: Dict,
    anomaly_summary: Dict,
    output_path=None, show=False,
) -> plt.Figure:
    """VIZ 2 — Sentiment TextBlob presse + anomalies."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.patch.set_facecolor(COLORS["bg"])
    fig.suptitle("Sentiment Presse — TextBlob (anomalies detectees automatiquement)",
                 fontsize=12, fontweight="bold")

    periods = sorted(agg_press.keys())

    ax = axes[0]
    dts, vals = _series(agg_press, "tb_polarity")
    if dts:
        ax.plot(dts, vals, color=COLORS["primary"], lw=2, marker="o",
                markersize=3, label="Polarite (-1 a +1)")
        ax.fill_between(dts, vals, 0,
                        where=[v < 0 for v in vals],
                        color=COLORS["secondary"], alpha=0.2)
        ax.fill_between(dts, vals, 0,
                        where=[v >= 0 for v in vals],
                        color=COLORS["tertiary"], alpha=0.2)
        ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.6)
    _style(ax, "Polarite TextBlob (Presse)", "Score (-1 a +1)")
    _add_anomaly_lines(ax, {}, anomaly_summary)
    ax.legend(fontsize=8)

    ax = axes[1]
    dts, vals = _series(agg_press, "tb_subjectivity")
    if dts:
        ax.plot(dts, vals, color=COLORS["accent"], lw=2, marker="o",
                markersize=3, label="Subjectivite (0=factuel, 1=subjectif)")
        ax.fill_between(dts, vals, alpha=0.1, color=COLORS["accent"])
    _style(ax, "Subjectivite TextBlob (Presse)", "Score (0-1)")
    ax.set_ylim(0, 1)
    _add_anomaly_lines(ax, {}, anomaly_summary)
    ax.legend(fontsize=8)

    _save(fig, output_path, show)
    return fig


def plot_raw_signals(
    agg_all: Dict, agg_press: Dict, agg_twitter: Dict,
    anomaly_summary: Dict,
    output_path=None, show=False,
) -> plt.Figure:
    """VIZ 3 — Signaux bruts + anomalies detectees."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.patch.set_facecolor(COLORS["bg"])
    fig.suptitle("Signaux d'intensite emotionnelle bruts (anomalies detectees automatiquement)",
                 fontsize=12, fontweight="bold")

    configs = [
        (agg_all,     "Global",  COLORS["text"],      2.0),
        (agg_press,   "Presse",  COLORS["primary"],   1.5),
        (agg_twitter, "Twitter", COLORS["secondary"], 1.5),
    ]

    for key, ax, title, ylabel, note in [
        ("ratio_caps",            axes[0],
         "Ratio majuscules (MOTS EN MAJ / total)",
         "Ratio", "Indignation brute : TERRIBLE, BOYCOTT..."),
        ("ratio_strong_punct",    axes[1],
         "Ratio ponctuation forte (! et ? / total)",
         "Ratio", "Intensite expressionnelle : !!! ???"),
        ("ratio_intensity_words", axes[2],
         "Ratio mots d'intensite (lexique colere/urgence/danger)",
         "Ratio", "Signal lexical : boycott, contamination, poison..."),
    ]:
        for agg, label, color, lw in configs:
            dts, vals = _series(agg, key)
            if dts:
                ax.plot(dts, vals, color=color, lw=lw, marker="o",
                        markersize=3, label=label, alpha=0.85)
        _style(ax, title, ylabel)
        _add_anomaly_lines(ax, {}, anomaly_summary)
        ax.legend(fontsize=8, loc="upper right")
        ax.annotate(note, xy=(0.01, 0.04), xycoords="axes fraction",
                    fontsize=6, color="gray", style="italic")

    _save(fig, output_path, show)
    return fig


def plot_anomaly_overview(
    anomaly_summary: Dict,
    all_anomalies: Dict,
    output_path=None, show=False,
) -> plt.Figure:
    """
    VIZ 4 — Vue d'ensemble des anomalies detectees.
    Heatmap : periodes x metriques surveillees.
    Chaque cellule = z-score absolu (blanc=normal, rouge=anomalie).
    """
    if not anomaly_summary:
        print("  Aucune anomalie detectee — VIZ 4 ignoree")
        return None

    metrics_order = [
        "vader_compound_twitter",
        "vader_neg_twitter",
        "ratio_intensity_words_all",
        "ratio_strong_punct_all",
        "ratio_caps_all",
        "tb_polarity_press",
        "tb_subjectivity_press",
    ]
    metrics_labels = [
        "VADER Compound\n(Twitter)",
        "VADER Negatif\n(Twitter)",
        "Mots Intensite\n(Global)",
        "Ponct. Forte\n(Global)",
        "Majuscules\n(Global)",
        "TextBlob Pol.\n(Presse)",
        "TextBlob Subj.\n(Presse)",
    ]

    # Periodes qui ont au moins une anomalie
    all_periods = sorted(anomaly_summary.keys())
    if not all_periods:
        return None

    # Construire la matrice z-score (periodes x metriques)
    matrix = np.zeros((len(all_periods), len(metrics_order)))
    for j, metric_key in enumerate(metrics_order):
        metric_anomalies = all_anomalies.get(metric_key, {})
        for i, period in enumerate(all_periods):
            if period in metric_anomalies:
                matrix[i, j] = abs(metric_anomalies[period]["zscore"])

    fig, ax = plt.subplots(figsize=(14, max(4, len(all_periods) * 0.4 + 2)))
    fig.patch.set_facecolor(COLORS["bg"])

    im = ax.imshow(matrix, cmap="Reds", aspect="auto",
                   vmin=0, vmax=max(3.5, matrix.max()))

    ax.set_xticks(range(len(metrics_labels)))
    ax.set_xticklabels(metrics_labels, fontsize=8, ha="center")
    ax.set_yticks(range(len(all_periods)))
    ax.set_yticklabels(all_periods, fontsize=7)

    # Annotations z-score dans les cellules
    for i in range(len(all_periods)):
        for j in range(len(metrics_order)):
            val = matrix[i, j]
            if val > 0:
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if val > 2.5 else COLORS["text"])

    plt.colorbar(im, ax=ax, label="|z-score|", shrink=0.8)
    ax.set_title("Heatmap des anomalies detectees — z-score absolu par metrique\n"
                 "(detection automatique sans connaissance a priori des evenements)",
                 fontsize=10, fontweight="bold")
    ax.set_facecolor(COLORS["bg"])

    _save(fig, output_path, show)
    return fig


# ─────────────────────────────────────────────────────────────────
# 9. CHARGEMENT DES JSON
# ─────────────────────────────────────────────────────────────────

def load_corpus(press_path: str, twitter_path: str) -> List[Dict]:
    """Charge press_scrapper.json et twitter_scraper.json."""
    corpus = []
    doc_id = 0

    print(f"  Chargement presse : {press_path}")
    with open(press_path, encoding="utf-8") as f:
        raw = json.load(f)
    articles = raw.get("articles", raw) if isinstance(raw, dict) else raw

    for art in articles:
        text = art.get("text", "").strip()
        if not text:
            continue
        try:
            ym = datetime.strptime(art.get("date", "")[:10], "%Y-%m-%d").strftime("%Y-%m")
        except (ValueError, TypeError):
            continue
        corpus.append({
            "id": doc_id, "source": "press",
            "platform": art.get("source", ""), "date": ym, "text": text,
        })
        doc_id += 1

    press_count = doc_id
    print(f"    -> {press_count:,} articles charges")

    print(f"  Chargement Twitter : {twitter_path}")
    with open(twitter_path, encoding="utf-8") as f:
        raw = json.load(f)
    tweets = raw.get("tweets", raw) if isinstance(raw, dict) else raw

    for tw in tweets:
        text = tw.get("text", "").strip()
        if not text:
            continue
        ym = tw.get("year_month", "")
        if not ym:
            try:
                ym = datetime.strptime(tw.get("date", "")[:10], "%Y-%m-%d").strftime("%Y-%m")
            except (ValueError, TypeError):
                continue
        corpus.append({
            "id": doc_id, "source": "twitter",
            "platform": "Twitter/X", "date": ym, "text": text,
        })
        doc_id += 1

    print(f"    -> {doc_id - press_count:,} tweets charges")
    print(f"  Total : {len(corpus):,} documents")
    corpus.sort(key=lambda x: x["date"])
    return corpus


# ─────────────────────────────────────────────────────────────────
# 10. PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────

def run_emotional_analysis(
    press_path:      str,
    twitter_path:    str,
    linguistic_json: Optional[str] = None,
    output_dir:      str = "outputs/reports",
    show_plots:      bool = False,
    zscore_threshold: float = 2.0,
) -> Dict:
    """
    Pipeline complet Phase 5.2 :
      1. Chargement corpus
      2. Calcul indicateurs emotionnels par document
      3. Agregation par periode x source
      4. Detection automatique d'anomalies (z-score)
      5. Fusion avec 5.1 + affichage tableau
      6. Export JSON + CSV
      7. 4 visualisations

    Pas de dates hardcodees — les anomalies sont detectees par les donnees.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Etape 1 : chargement
    print(f"\n[Phase 5.2] Chargement corpus...")
    corpus = load_corpus(press_path, twitter_path)

    # Etape 2 : calcul par document
    print(f"\n[Phase 5.2] Calcul indicateurs emotionnels — {len(corpus):,} documents...")
    doc_metrics, skipped = [], 0
    for doc in corpus:
        if not doc.get("text", "").strip():
            skipped += 1
            continue
        doc_metrics.append(compute_emotional_metrics(doc))
    print(f"  -> {len(doc_metrics):,} traites | {skipped} ignores")

    # Etape 3 : agregation
    print("\n[Phase 5.2] Agregation par periode x source...")
    emot_agg    = aggregate_emotional_all_sources(doc_metrics)
    agg_all     = emot_agg["all"]
    agg_press   = emot_agg["press"]
    agg_twitter = emot_agg["twitter"]
    print(f"  -> {len(agg_all)} periodes YYYY-MM")

    # Etape 4 : detection automatique d'anomalies
    print(f"\n[Phase 5.2] Detection d'anomalies (z-score threshold={zscore_threshold})...")
    all_anomalies   = detect_all_anomalies(emot_agg, threshold=zscore_threshold)
    anomaly_summary = build_anomaly_summary(all_anomalies)
    anomaly_labels  = format_anomaly_labels(anomaly_summary)

    print(f"  -> {len(anomaly_summary)} periodes anormales detectees :")
    for period, info in sorted(anomaly_summary.items()):
        signals = ", ".join(s["metric"] for s in info["signals"])
        print(f"     {period} [{info['severity']:8s}] z={info['max_zscore']:.2f} "
              f"| {info['n_signals']} signal(s) : {signals}")

    # Etape 5 : fusion avec 5.1
    ling_agg = {"all": {}, "press": {}, "twitter": {}}
    if linguistic_json and os.path.exists(linguistic_json):
        print(f"\n[Phase 5.2] Chargement resultats linguistiques 5.1...")
        with open(linguistic_json, encoding="utf-8") as f:
            ling_data = json.load(f)
        ling_agg = ling_data.get("aggregated", ling_agg)
        print(f"  -> Fusion 5.1 + 5.2 activee")

    summary_rows = build_summary_table(ling_agg, emot_agg, anomaly_summary)

    print("\n" + "=" * 60)
    print("TABLEAU DE SYNTHESE — INDICATEURS COGNITIFS COMPLETS")
    print("=" * 60)
    for source in ["all", "press", "twitter"]:
        display_summary_table(summary_rows, source_filter=source)

    # Etape 6 : exports
    print("[Phase 5.2] Exports...")

    class _Enc(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):  return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, datetime):        return obj.isoformat()
            return super().default(obj)

    json_path = os.path.join(output_dir, "emotional_intensity_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at":    datetime.now().isoformat(),
            "n_documents":     len(doc_metrics),
            "zscore_threshold": zscore_threshold,
            "aggregated":      emot_agg,
            "anomalies":       anomaly_summary,
            "anomalies_by_metric": all_anomalies,
            "documents":       doc_metrics,
        }, f, ensure_ascii=False, indent=2, cls=_Enc)
    print(f"  OK {os.path.basename(json_path)}")

    csv_path = os.path.join(output_dir, "cognitive_indicators_summary.csv")
    export_summary_csv(summary_rows, csv_path)

    # Etape 7 : visualisations
    print("\n[Phase 5.2] Generation des visualisations...")

    plot_vader_twitter(
        agg_twitter, anomaly_summary,
        output_path=os.path.join(output_dir, "em_01_vader_twitter.png"),
        show=show_plots,
    )
    plot_textblob_press(
        agg_press, anomaly_summary,
        output_path=os.path.join(output_dir, "em_02_textblob_press.png"),
        show=show_plots,
    )
    plot_raw_signals(
        agg_all, agg_press, agg_twitter, anomaly_summary,
        output_path=os.path.join(output_dir, "em_03_raw_signals.png"),
        show=show_plots,
    )
    plot_anomaly_overview(
        anomaly_summary, all_anomalies,
        output_path=os.path.join(output_dir, "em_04_anomaly_heatmap.png"),
        show=show_plots,
    )

    print(f"\n[Phase 5.2] Termine — fichiers dans : {output_dir}")
    return emot_agg


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    PRESS_PATH   = r"C:\Users\PC\Documents\PFE\plateforme_analyse_textuelle\src\cognitive\press_scrapper.json"
    TWITTER_PATH = r"C:\Users\PC\Documents\PFE\plateforme_analyse_textuelle\src\cognitive\twitter_scraper.json"
    LING_JSON    = r"C:\Users\PC\Documents\PFE\plateforme_analyse_textuelle\src\cognitive\outputs\reports\linguistic_metrics_results.json"
    OUTPUT_DIR    = r"C:\Users\PC\Documents\PFE\plateforme_analyse_textuelle\src\cognitive\outputs\reports\emotional"

    run_emotional_analysis(
        press_path       = PRESS_PATH,
        twitter_path     = TWITTER_PATH,
        linguistic_json  = LING_JSON,
        output_dir       = OUTPUT_DIR,
        show_plots       = False,
        zscore_threshold = 2.0,
    )
