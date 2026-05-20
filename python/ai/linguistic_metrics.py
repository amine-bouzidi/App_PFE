"""
linguistic_metrics.py — Phase 5.1 : Indicateurs Cognitifs Linguistiques
Plateforme d'analyse textuelle multimodale — McDonald's Due Diligence

Cinq axes d'analyse :
  1. Richesse lexicale        -> TTR, MATTR, MTLD, Hapax ratio, Yule's K
  2. Complexite syntaxique    -> longueur phrases, densite clausale
  3. Lisibilite (textstat)    -> ARI, Coleman-Liau, LIX, Flesch (manuel)
  4. Densite informationnelle -> ratio content/function words, lexical density
  5. Coherence thematique     -> similarite intra-doc (Jaccard), concentration lexicale

Dependances : numpy, matplotlib, textstat, lexicalrichness
Aucun modele a telecharger — 100% offline apres installation pip.

Installation :
    pip install textstat lexicalrichness

Usage :
    from linguistic_metrics import load_corpus, run_linguistic_analysis
    corpus = load_corpus("press_scrapper.json", "twitter_scraper.json")
    run_linguistic_analysis(corpus, output_dir="outputs/reports")
"""

import os
import re
import json
import sys
import warnings
from collections import Counter, defaultdict
from statistics import mean, stdev
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import textstat
from lexicalrichness import LexicalRichness

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ─────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────

SCANDALS_KPMG: Dict[str, str] = {
    "2024-01": "Orpea",
    "2024-04": "Wirecard",
    "2024-10": "Wirecard DE",
    "2025-02": "Immobilier DE",
    "2025-12": "Carillion 20M",
}

SCANDALS_MCDO: Dict[str, str] = {
    "2021-08": "Easterbrook",
    "2022-03": "Pink Slime",
    "2023-10": "FightFor15",
    "2024-07": "E.coli",
    "2025-01": "Palestine Boycott",
}

# Mots-outils multilingues EN / FR / DE / ES (~200 tokens)
FUNCTION_WORDS = frozenset({
    # Anglais
    "a", "an", "the", "and", "or", "but", "nor", "so", "yet", "for",
    "in", "on", "at", "to", "by", "of", "up", "as", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "shall", "should", "may", "might", "must",
    "can", "could", "not", "no", "it", "its", "this", "that", "these",
    "those", "he", "she", "they", "we", "you", "i", "me", "him", "her",
    "us", "them", "my", "your", "his", "our", "their", "who", "which",
    "what", "how", "when", "where", "with", "from", "into", "than",
    "then", "also", "just", "more", "very", "even", "still", "only",
    "both", "each", "some", "any", "all", "such", "if", "out", "about",
    "after", "before", "since", "while", "because", "although", "though",
    # Francais
    "le", "la", "les", "un", "une", "des", "et", "ou", "mais", "donc",
    "ni", "car", "de", "du", "en", "au", "aux", "par", "sur",
    "dans", "avec", "sans", "sous", "pour", "est", "sont", "etait",
    "ont", "ce", "cet", "cette", "ces", "il", "elle", "ils", "elles",
    "je", "tu", "nous", "vous", "me", "te", "se", "lui", "leur", "leurs",
    "mon", "ma", "mes", "ton", "ta", "tes", "son", "sa", "ses", "qui",
    "que", "quoi", "dont", "plus", "tres", "aussi", "meme",
    "tout", "tous", "toute", "toutes", "si", "pas", "ne", "y",
    # Allemand
    "der", "die", "das", "dem", "den", "ein", "eine", "einen",
    "einem", "einer", "und", "oder", "aber", "denn", "weil", "wenn",
    "als", "wie", "an", "auf", "bei", "von", "zu", "mit",
    "ist", "sind", "war", "waren", "hat", "haben", "hatte", "hatten",
    "er", "sie", "es", "wir", "ihr", "ich", "du", "mich", "mir", "ihn",
    "ihm", "uns", "euch", "sich", "auch", "noch", "schon", "nur", "sehr",
    "so", "dass", "ob", "nach", "uber", "unter", "vor", "zwischen", "durch",
    # Espagnol
    "el", "los", "las", "unos", "unas", "del", "al",
    "con", "por", "para", "sin", "sobre", "entre", "ante",
    "esta", "estan", "fue", "eran", "tiene", "tienen",
    "ha", "han", "habia", "lo", "les",
    "como", "cuando", "donde", "cual", "cuyo",
    "sino", "aunque", "tambien", "ya",
    "este", "estos", "estas", "ese", "esa", "aquel", "aquella",
    "su", "sus",
})

# Subordonnants multilingues pour densite clausale
SUBORDINATORS = frozenset({
    "who", "which", "that", "because", "although", "though", "while",
    "since", "unless", "until", "if", "whether", "when", "where",
    "whose", "whom", "after", "before", "as",
    "qui", "que", "parce", "quoique", "tandis", "lorsque",
    "puisque", "avant", "quand", "dont", "lequel",
    "weil", "obwohl", "wahrend", "dass", "wenn", "ob", "bevor",
    "nachdem", "sobald", "damit", "falls",
    "porque", "aunque", "mientras", "cuando", "donde", "como",
    "antes", "despues",
})

COLORS = {
    "bg":        "#f8f9fa",
    "text":      "#1a1a1a",
    "primary":   "#0066cc",
    "secondary": "#e63946",
    "tertiary":  "#2dc653",
    "accent":    "#f4a261",
    "grid":      "#cccccc",
}

# Seuil MTLD standard (McCarthy & Jarvis 2010)
MTLD_THRESHOLD = 0.72


# ─────────────────────────────────────────────────────────────────
# 1. TOKENISATION
# ─────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    """Supprime URLs, mentions, hashtags, normalise les espaces."""
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[@#]\w+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize_words(text: str) -> List[str]:
    """
    Tokens alphabetiques min 2 chars lowercase.
    Couvre EN/FR/DE/ES via Unicode etendu + arabe basique.
    """
    return re.findall(r"[a-zA-Z\xc0-\xff\u0600-\u06FF]{2,}", _clean(text).lower())


def tokenize_sentences(text: str) -> List[str]:
    """
    Decoupe en phrases sur .!? avec protection des abreviations courantes.
    """
    text = _clean(text)
    text = re.sub(r"\b(Mr|Mrs|Dr|Prof|vs|etc|M|Mme|No)\.", r"\1<DOT>", text)
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [re.sub(r"<DOT>", ".", s).strip() for s in parts if s.strip()]


def content_words(tokens: List[str]) -> List[str]:
    """Filtre les mots-outils, retourne les mots porteurs de sens."""
    return [w for w in tokens if w not in FUNCTION_WORDS and len(w) > 2]


# ─────────────────────────────────────────────────────────────────
# 2. AXE 1 — RICHESSE LEXICALE (TTR + MATTR + MTLD)
# ─────────────────────────────────────────────────────────────────

def compute_ttr(tokens: List[str]) -> float:
    """
    Type-Token Ratio : types / tokens.
    Metrique de reference. Sensible a la longueur du texte :
    diminue mecaniquement quand le texte s'allonge.
    Ne pas comparer directement entre tweets et articles.
    """
    if not tokens:
        return 0.0
    return round(len(set(tokens)) / len(tokens), 4)


def compute_mattr(tokens: List[str], window: int = 50) -> float:
    """
    Moving Average Type-Token Ratio (Covington & McFall 2010).
    Calcule le TTR sur des fenetres glissantes de `window` tokens puis moyenne.
    Insensible a la longueur du texte -> valide pour comparer
    tweets (~15 tokens) et articles (~300 tokens).
    Si le texte est plus court que la fenetre : TTR simple.
    """
    n = len(tokens)
    if n == 0:
        return 0.0
    if n <= window:
        return compute_ttr(tokens)
    scores = [
        len(set(tokens[i: i + window])) / window
        for i in range(n - window + 1)
    ]
    return round(mean(scores), 4)


def compute_mtld(text: str) -> Optional[float]:
    """
    Measure of Textual Lexical Diversity (McCarthy & Jarvis 2010).
    Parcourt le texte sequentiellement et mesure la longueur des segments
    ou le TTR reste au-dessus du seuil MTLD_THRESHOLD (0.72).
    Plus les segments sont longs avant que le TTR chute, plus le score est eleve.

    Proprietes :
      - Insensible a la longueur du texte
      - Tres sensible aux repetitions en periode de crise
      - Score typique : 50-200 (texte normal), < 20 (crise / slogan)

    Retourne None si le texte est trop court (< 10 tokens)
    pour que LexicalRichness puisse calculer.
    Utilise la bibliotheque lexicalrichness.
    """
    clean = _clean(text)
    tokens = tokenize_words(text)
    if len(tokens) < 10:
        return None
    try:
        # LexicalRichness attend le texte brut, pas les tokens
        lex = LexicalRichness(clean)
        if lex.words < 10:
            return None
        return round(lex.mtld(threshold=MTLD_THRESHOLD), 2)
    except Exception:
        return None


def compute_hapax_ratio(tokens: List[str]) -> float:
    """
    Ratio hapax legomena : mots n'apparaissant qu'une seule fois / total types.
    Eleve -> vocabulaire tres varie.
    Bas   -> discours concentre, repetitif (signal de pensee de groupe).
    """
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    hapax = sum(1 for c in counts.values() if c == 1)
    return round(hapax / len(counts), 4)


def compute_yule_k(tokens: List[str]) -> float:
    """
    Yule's K (1944) — diversite du vocabulaire independante de la longueur.
    Formule : K = 10^4 x (Sigma V(m,N)xm^2 - N) / N^2
    ou V(m,N) = nombre de types apparaissant exactement m fois.
    K eleve -> vocabulaire varie | K bas -> vocabulaire repetitif.
    """
    n = len(tokens)
    if n < 2:
        return 0.0
    counts = Counter(tokens)
    freq_of_freq = Counter(counts.values())
    sum_sq = sum(freq * (m ** 2) for m, freq in freq_of_freq.items())
    denom = n * n
    if denom == 0:
        return 0.0
    return round(max(0.0, 10_000 * (sum_sq - n) / denom), 2)


def axe_lexical_richness(text: str) -> Dict:
    """
    Axe 1 complet : richesse lexicale.
    Trois metriques de diversite complementaires :
      TTR   -> reference brute (sensible a la longueur)
      MATTR -> fenetre glissante (insensible a la longueur)
      MTLD  -> endurance de la diversite (le plus sensible aux crises)
    Plus Hapax ratio et Yule's K comme indicateurs supplementaires.
    `is_short` signale les textes < 30 tokens.
    """
    tokens = tokenize_words(text)
    n = len(tokens)
    return {
        "n_tokens":    n,
        "n_types":     len(set(tokens)),
        "ttr":         compute_ttr(tokens),
        "mattr":       compute_mattr(tokens),
        "mtld":        compute_mtld(text),   # None si texte trop court
        "hapax_ratio": compute_hapax_ratio(tokens),
        "yule_k":      compute_yule_k(tokens),
        "is_short":    n < 30,
    }


# ─────────────────────────────────────────────────────────────────
# 3. AXE 2 — COMPLEXITE SYNTAXIQUE
# ─────────────────────────────────────────────────────────────────

def compute_avg_sentence_length(text: str) -> float:
    """Longueur moyenne des phrases en tokens."""
    sentences = tokenize_sentences(text)
    if not sentences:
        return 0.0
    lengths = [len(tokenize_words(s)) for s in sentences]
    valid = [l for l in lengths if l > 0]
    return round(mean(valid), 2) if valid else 0.0


def compute_sentence_length_std(text: str) -> float:
    """Ecart-type longueur des phrases — variabilite structurelle."""
    sentences = tokenize_sentences(text)
    lengths = [len(tokenize_words(s)) for s in sentences if tokenize_words(s)]
    return round(stdev(lengths), 2) if len(lengths) >= 2 else 0.0


def compute_clause_density(text: str) -> float:
    """
    Clauses par phrase (approximation sans POS-tagger).
    Detecte les subordonnants multilingues -> +1 clause par subordonnant.
    Valeur typique : 1.0 (phrase simple) a 3.0+ (phrase complexe imbriquee).
    """
    sentences = tokenize_sentences(text)
    if not sentences:
        return 0.0
    total = sum(
        sum(1 for w in set(tokenize_words(s)) if w in SUBORDINATORS) + 1
        for s in sentences
    )
    return round(total / len(sentences), 2)


def axe_syntactic_complexity(text: str) -> Dict:
    """Axe 2 complet : complexite syntaxique."""
    return {
        "n_sentences":     len(tokenize_sentences(text)),
        "avg_sent_length": compute_avg_sentence_length(text),
        "sent_length_std": compute_sentence_length_std(text),
        "clause_density":  compute_clause_density(text),
    }


# ─────────────────────────────────────────────────────────────────
# 4. AXE 3 — LISIBILITE (textstat)
# ─────────────────────────────────────────────────────────────────

def _flesch_manual(text: str) -> float:
    """
    Flesch Reading Ease — implementation manuelle.
    Formule : 206.835 - 1.015*(mots/phrases) - 84.6*(syllabes/mots)
    Syllabe approximee = groupes de voyelles consecutives dans un mot.
    Implementation manuelle car textstat 0.7.x requiert NLTK cmudict
    non disponible en mode offline.
    Score : 100 (tres facile) -> 0 (tres difficile).
    """
    sentences = tokenize_sentences(text)
    words = tokenize_words(text)
    if not sentences or not words:
        return 0.0

    vowels = set("aeiouy\xe0\xe2\xe4\xe9\xe8\xea\xeb\xee\xef\xf4\xf9\xfb\xfc\u0153\xe6")

    def count_syllables(word: str) -> int:
        count = sum(
            1 for i, c in enumerate(word)
            if c in vowels and (i == 0 or word[i - 1] not in vowels)
        )
        return max(1, count)

    total_syllables = sum(count_syllables(w) for w in words)
    score = (206.835
             - 1.015 * (len(words) / len(sentences))
             - 84.6 * (total_syllables / len(words)))
    return round(max(0.0, min(100.0, score)), 2)


def axe_readability(text: str) -> Dict:
    """
    Axe 3 : lisibilite via textstat + Flesch manuel.
    Metriques textstat utilisees (sans NLTK cmudict) :
      ARI          : base sur chars/mots et mots/phrases. Score = niveau scolaire US.
      Coleman-Liau : base sur chars/100 mots. Robuste sur textes non-anglais.
      LIX          : ratio mots longs (>6 chars). Multilingue par construction.
    """
    clean = _clean(text)
    if not clean:
        return {
            "flesch_reading_ease": None,
            "ari": None,
            "coleman_liau": None,
            "lix": None,
            "avg_chars_per_word": None,
        }

    results = {}
    for key, func in [
        ("ari",                textstat.automated_readability_index),
        ("coleman_liau",       textstat.coleman_liau_index),
        ("lix",                textstat.lix),
        ("avg_chars_per_word", textstat.avg_character_per_word),
    ]:
        try:
            results[key] = round(func(clean), 2)
        except Exception:
            results[key] = None

    results["flesch_reading_ease"] = _flesch_manual(text)
    return results


# ─────────────────────────────────────────────────────────────────
# 5. AXE 4 — DENSITE INFORMATIONNELLE
# ─────────────────────────────────────────────────────────────────

def compute_information_density(tokens: List[str]) -> float:
    """
    Ratio mots-a-contenu / total tokens.
    Eleve -> texte factuellement dense (presse, rapports).
    Bas   -> texte discursif ou emotionnel (tweets de crise).
    """
    if not tokens:
        return 0.0
    return round(len(content_words(tokens)) / len(tokens), 4)


def compute_lexical_density(tokens: List[str]) -> float:
    """
    Types-a-contenu / types totaux.
    Proportion du vocabulaire utilise qui est semantiquement porteur.
    """
    if not tokens:
        return 0.0
    types_total = set(tokens)
    types_content = set(content_words(tokens))
    if not types_total:
        return 0.0
    return round(len(types_content) / len(types_total), 4)


def axe_information_density(text: str) -> Dict:
    """Axe 4 complet : densite informationnelle."""
    tokens = tokenize_words(text)
    return {
        "information_density": compute_information_density(tokens),
        "lexical_density":     compute_lexical_density(tokens),
    }


# ─────────────────────────────────────────────────────────────────
# 6. AXE 5 — COHERENCE THEMATIQUE
# ─────────────────────────────────────────────────────────────────

def compute_intra_doc_coherence(text: str, window: int = 40) -> Optional[float]:
    """
    Similarite de Jaccard entre fenetres adjacentes de mots-a-contenu.
    Retourne None si texte trop court (< 2xwindow tokens-contenu).
    """
    tokens = content_words(tokenize_words(text))
    if len(tokens) < window * 2:
        return None
    scores = []
    for i in range(len(tokens) - window):
        w1 = set(tokens[i: i + window])
        w2 = set(tokens[i + 1: i + 1 + window])
        union = len(w1 | w2)
        if union > 0:
            scores.append(len(w1 & w2) / union)
    return round(mean(scores), 4) if scores else None


def compute_topic_concentration(text: str, top_n: int = 10) -> float:
    """
    Part des top_n mots-a-contenu les plus frequents dans le total.
    Eleve -> discours focalise (slogan repete, crise concentree).
    Bas   -> discours thematiquement diversifie.
    """
    tokens = content_words(tokenize_words(text))
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    top_count = sum(c for _, c in counts.most_common(top_n))
    return round(top_count / len(tokens), 4)


def axe_coherence(text: str) -> Dict:
    """Axe 5 complet : coherence thematique."""
    return {
        "intra_doc_coherence": compute_intra_doc_coherence(text),
        "topic_concentration": compute_topic_concentration(text),
    }


# ─────────────────────────────────────────────────────────────────
# 7. METRIQUES DOCUMENT COMPLET
# ─────────────────────────────────────────────────────────────────

def compute_document_metrics(doc: Dict) -> Dict:
    """
    Calcule l'ensemble des indicateurs cognitifs pour un document.
    Entree : dict avec 'text', 'id', 'date', 'source', 'language', etc.
    Sortie : dict metriques (5 axes, 20 indicateurs) + metadonnees.
    """
    text = doc.get("text", "")

    meta = {
        "doc_id":   doc.get("id"),
        "date":     doc.get("date", ""),
        "source":   doc.get("source", ""),
        "language": doc.get("language", ""),
        "category": doc.get("category", ""),
        "platform": doc.get("platform", ""),
    }

    if not text or not text.strip():
        return {**meta, **{
            "n_tokens": 0, "n_types": 0,
            "ttr": 0.0, "mattr": 0.0, "mtld": None,
            "hapax_ratio": 0.0, "yule_k": 0.0, "is_short": True,
            "n_sentences": 0, "avg_sent_length": 0.0,
            "sent_length_std": 0.0, "clause_density": 0.0,
            "flesch_reading_ease": None, "ari": None,
            "coleman_liau": None, "lix": None, "avg_chars_per_word": None,
            "information_density": 0.0, "lexical_density": 0.0,
            "intra_doc_coherence": None, "topic_concentration": 0.0,
        }}

    metrics = {}
    metrics.update(axe_lexical_richness(text))
    metrics.update(axe_syntactic_complexity(text))
    metrics.update(axe_readability(text))
    metrics.update(axe_information_density(text))
    metrics.update(axe_coherence(text))

    return {**meta, **metrics}


# ─────────────────────────────────────────────────────────────────
# 8. AGREGATION PAR PERIODE x SOURCE
# ─────────────────────────────────────────────────────────────────

_NUMERIC_KEYS = [
    "n_tokens", "n_types",
    "ttr", "mattr", "mtld", "hapax_ratio", "yule_k",
    "n_sentences", "avg_sent_length", "sent_length_std", "clause_density",
    "flesch_reading_ease", "ari", "coleman_liau", "lix", "avg_chars_per_word",
    "information_density", "lexical_density",
    "intra_doc_coherence", "topic_concentration",
]


def aggregate_by_period(
    doc_metrics: List[Dict],
    source_filter: Optional[str] = None,
) -> Dict[str, Dict]:
    """
    Agrege les metriques par mois (YYYY-MM).
    Les None (MTLD trop court, coherence trop courte) sont exclus
    des moyennes sans penaliser la periode.
    """
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for m in doc_metrics:
        if source_filter and m.get("source") != source_filter:
            continue
        try:
            period = _normalize_period(m.get("date", ""))
        except ValueError:
            continue
        groups[period].append(m)

    result = {}
    for period, docs in sorted(groups.items()):
        agg: Dict = {"n_docs": len(docs)}
        for key in _NUMERIC_KEYS:
            vals = [
                d[key] for d in docs
                if d.get(key) is not None and isinstance(d.get(key), (int, float))
            ]
            agg[key] = round(mean(vals), 4) if vals else None
        result[period] = agg
    return result


def aggregate_all_sources(doc_metrics: List[Dict]) -> Dict[str, Dict[str, Dict]]:
    """Retourne trois vues agregees : all / press / twitter."""
    return {
        "all":     aggregate_by_period(doc_metrics),
        "press":   aggregate_by_period(doc_metrics, source_filter="press"),
        "twitter": aggregate_by_period(doc_metrics, source_filter="twitter"),
    }


# ─────────────────────────────────────────────────────────────────
# 9. HELPERS VISUALISATION
# ─────────────────────────────────────────────────────────────────

def _normalize_period(period: str) -> str:
    """Retourne un period format YYYY-MM à partir d'une date ou d'un mois."""
    if not period or not isinstance(period, str):
        raise ValueError(f"Invalid period: {period!r}")
    period = period.strip()
    if re.match(r"^\d{4}-\d{2}$", period):
        return period

    # Accepte des dates ISO complètes comme 2026-05-19T00:03:36.000Z
    if re.match(r"^\d{4}-\d{2}-\d{2}", period):
        return period[:7]

    raise ValueError(f"Unsupported period format: {period!r}")


def _to_dt(period: str) -> datetime:
    normalized = _normalize_period(period)
    return datetime.strptime(normalized + "-01", "%Y-%m-%d")


def _series(agg: Dict[str, Dict], key: str) -> Tuple[List[datetime], List[float]]:
    """Extrait la serie temporelle d'une metrique, ignore les None."""
    periods = sorted(agg.keys())
    dts, vals = [], []
    for p in periods:
        v = agg[p].get(key)
        if v is not None:
            dts.append(_to_dt(p))
            vals.append(v)
    return dts, vals


def _add_scandals(ax: plt.Axes, periods: List[str], scandals: Dict[str, str]) -> None:
    """Lignes verticales rouges pour chaque scandale."""
    ylim = ax.get_ylim()
    for month, label in scandals.items():
        if month in periods:
            ax.axvline(_to_dt(month), color="#cc0000", ls="--",
                       lw=0.9, alpha=0.65, zorder=1)
            ax.text(_to_dt(month), ylim[1] * 0.97, label,
                    rotation=90, fontsize=6, color="#cc0000",
                    va="top", ha="right", zorder=2)


def _style(ax: plt.Axes, title: str, ylabel: str) -> None:
    """Style commun a tous les axes."""
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


# ─────────────────────────────────────────────────────────────────
# 10. VISUALISATIONS (8 graphiques)
# ─────────────────────────────────────────────────────────────────

def plot_lexical_richness(
    agg_all, agg_press, agg_twitter,
    scandals=SCANDALS_MCDO, output_path=None, show=False,
) -> plt.Figure:
    """
    VIZ 1 — Richesse lexicale : TTR, MATTR, MTLD.
    Layout 3x1 pour voir les trois metriques ensemble et comparer
    ce que chacune capture differemment sur les crises.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.patch.set_facecolor(COLORS["bg"])
    fig.suptitle("Richesse lexicale — TTR / MATTR / MTLD", fontsize=12, fontweight="bold")

    periods = sorted(agg_all.keys())
    configs = [
        (agg_all,     "Global",  COLORS["text"],      2.0),
        (agg_press,   "Presse",  COLORS["primary"],   1.5),
        (agg_twitter, "Twitter", COLORS["secondary"], 1.5),
    ]

    for key, ax, title, ylabel, note in [
        ("ttr",   axes[0], "TTR — Type-Token Ratio (sensible a la longueur)",
         "Score TTR", "Sensible a la longueur : ne pas comparer presse vs twitter directement"),
        ("mattr", axes[1], "MATTR — Moving Average TTR (insensible a la longueur)",
         "Score MATTR", "Fenetre glissante 50 tokens — valide pour comparer toutes sources"),
        ("mtld",  axes[2], "MTLD — Measure of Textual Lexical Diversity (endurance diversite)",
         "Score MTLD", "Le plus sensible aux crises : chute = convergence lexicale collective"),
    ]:
        for agg, label, color, lw in configs:
            dts, vals = _series(agg, key)
            if dts:
                ax.plot(dts, vals, color=color, lw=lw, marker="o",
                        markersize=3, label=label, alpha=0.85)
        _style(ax, title, ylabel)
        _add_scandals(ax, periods, scandals)
        ax.legend(fontsize=8, loc="upper right")
        ax.annotate(note, xy=(0.01, 0.04), xycoords="axes fraction",
                    fontsize=6, color="gray", style="italic")

    _save(fig, output_path, show)
    return fig


def plot_lexical_richness_extra(
    agg_all, agg_press, agg_twitter,
    scandals=SCANDALS_MCDO, output_path=None, show=False,
) -> plt.Figure:
    """VIZ 2 — Yule's K + Hapax ratio."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.patch.set_facecolor(COLORS["bg"])
    fig.suptitle("Richesse lexicale — Yule's K & Hapax ratio", fontsize=12, fontweight="bold")

    periods = sorted(agg_all.keys())
    configs = [
        (agg_all,     "Global",  COLORS["text"],      2.0),
        (agg_press,   "Presse",  COLORS["primary"],   1.5),
        (agg_twitter, "Twitter", COLORS["secondary"], 1.5),
    ]

    for key, ax, title, ylabel in [
        ("yule_k",      axes[0], "Yule's K — diversite vocabulaire (insensible longueur)", "Yule's K"),
        ("hapax_ratio", axes[1], "Hapax ratio — part des mots n'apparaissant qu'une fois", "Ratio"),
    ]:
        for agg, label, color, lw in configs:
            dts, vals = _series(agg, key)
            if dts:
                ax.plot(dts, vals, color=color, lw=lw, marker="o",
                        markersize=3, label=label, alpha=0.85)
        _style(ax, title, ylabel)
        _add_scandals(ax, periods, scandals)
        ax.legend(fontsize=8, loc="upper right")

    _save(fig, output_path, show)
    return fig


def plot_syntactic_complexity(
    agg_all, agg_press, agg_twitter,
    scandals=SCANDALS_MCDO, output_path=None, show=False,
) -> plt.Figure:
    """VIZ 3 — Longueur phrases + densite clausale."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.patch.set_facecolor(COLORS["bg"])
    fig.suptitle("Complexite syntaxique", fontsize=12, fontweight="bold")

    periods = sorted(agg_all.keys())
    configs = [
        (agg_all,     "Global",  COLORS["text"],      2.0),
        (agg_press,   "Presse",  COLORS["primary"],   1.5),
        (agg_twitter, "Twitter", COLORS["secondary"], 1.5),
    ]

    for key, ax, title, ylabel in [
        ("avg_sent_length", axes[0], "Longueur moyenne des phrases", "Tokens / phrase"),
        ("clause_density",  axes[1], "Densite clausale",             "Clauses / phrase"),
    ]:
        for agg, label, color, lw in configs:
            dts, vals = _series(agg, key)
            if dts:
                ax.plot(dts, vals, color=color, lw=lw, marker="o",
                        markersize=3, label=label, alpha=0.85)
        _style(ax, title, ylabel)
        _add_scandals(ax, periods, scandals)
        ax.legend(fontsize=8, loc="upper right")

    _save(fig, output_path, show)
    return fig


def plot_readability(
    agg_all, agg_press, agg_twitter,
    scandals=SCANDALS_MCDO, output_path=None, show=False,
) -> plt.Figure:
    """VIZ 4 — Lisibilite textstat : Flesch, ARI, Coleman-Liau, LIX."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor(COLORS["bg"])
    fig.suptitle("Lisibilite — Metriques textstat", fontsize=12, fontweight="bold")

    periods = sorted(agg_all.keys())
    configs = [
        (agg_all,     "Global",  COLORS["text"],      2.0),
        (agg_press,   "Presse",  COLORS["primary"],   1.5),
        (agg_twitter, "Twitter", COLORS["secondary"], 1.5),
    ]

    indicators = [
        ("flesch_reading_ease", axes[0, 0], "Flesch Reading Ease (100=facile, 0=difficile)", "Score Flesch"),
        ("ari",                 axes[0, 1], "ARI — Automated Readability Index (niveau scolaire US)", "Niveau"),
        ("coleman_liau",        axes[1, 0], "Coleman-Liau Index (robuste multilingue)", "Niveau"),
        ("lix",                 axes[1, 1], "LIX — Laesbarhetsindex (<30 facile | >60 difficile)", "Score LIX"),
    ]

    for key, ax, title, ylabel in indicators:
        for agg, label, color, lw in configs:
            dts, vals = _series(agg, key)
            if dts:
                ax.plot(dts, vals, color=color, lw=lw, marker="o",
                        markersize=3, label=label, alpha=0.85)
        _style(ax, title, ylabel)
        _add_scandals(ax, periods, scandals)
        ax.legend(fontsize=7, loc="upper right")

    _save(fig, output_path, show)
    return fig


def plot_information_density(
    agg_all, agg_press, agg_twitter,
    scandals=SCANDALS_MCDO, output_path=None, show=False,
) -> plt.Figure:
    """VIZ 5 — Densite informationnelle + densite lexicale."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.patch.set_facecolor(COLORS["bg"])
    fig.suptitle("Densite informationnelle", fontsize=12, fontweight="bold")

    periods = sorted(agg_all.keys())
    configs = [
        (agg_all,     "Global",  COLORS["text"],      2.0),
        (agg_press,   "Presse",  COLORS["primary"],   1.5),
        (agg_twitter, "Twitter", COLORS["secondary"], 1.5),
    ]

    for key, ax, title, ylabel in [
        ("information_density", axes[0],
         "Densite informationnelle (mots-a-contenu / total)", "Score (0-1)"),
        ("lexical_density",     axes[1],
         "Densite lexicale (types-a-contenu / types totaux)", "Score (0-1)"),
    ]:
        for agg, label, color, lw in configs:
            dts, vals = _series(agg, key)
            if dts:
                ax.plot(dts, vals, color=color, lw=lw, marker="o",
                        markersize=3, label=label, alpha=0.85)
                ax.fill_between(dts, vals, alpha=0.07, color=color)
        _style(ax, title, ylabel)
        ax.set_ylim(0, 1)
        _add_scandals(ax, periods, scandals)
        ax.legend(fontsize=8, loc="upper right")

    _save(fig, output_path, show)
    return fig


def plot_coherence(
    agg_all, agg_press,
    scandals=SCANDALS_MCDO, output_path=None, show=False,
) -> plt.Figure:
    """VIZ 6 — Coherence thematique."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.patch.set_facecolor(COLORS["bg"])
    fig.suptitle("Coherence thematique", fontsize=12, fontweight="bold")

    periods = sorted(agg_all.keys())

    ax = axes[0]
    dts, vals = _series(agg_press, "intra_doc_coherence")
    if dts:
        ax.plot(dts, vals, color=COLORS["primary"], lw=2, marker="o",
                markersize=3, label="Presse")
    _style(ax, "Coherence intra-document — Jaccard fenetres adjacentes", "Score Jaccard")
    ax.annotate("Non calcule sur Twitter (textes < 80 tokens-contenu)",
                xy=(0.01, 0.05), xycoords="axes fraction",
                fontsize=7, color="gray", style="italic")
    _add_scandals(ax, periods, scandals)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=8)

    ax = axes[1]
    for agg, label, color in [
        (agg_all,   "Global", COLORS["text"]),
        (agg_press, "Presse", COLORS["primary"]),
    ]:
        dts, vals = _series(agg, "topic_concentration")
        if dts:
            ax.plot(dts, vals, color=color, lw=1.8, marker="o",
                    markersize=3, label=label, alpha=0.85)
        _style(ax, "Concentration thematique — part top-10 mots-a-contenu", "Score (0-1)")
        _add_scandals(ax, periods, scandals)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=8)

    _save(fig, output_path, show)
    return fig


def plot_dashboard(
    agg_all, agg_press, agg_twitter,
    scandals=SCANDALS_MCDO, output_path=None, show=False,
) -> plt.Figure:
    """VIZ 7 — Dashboard 3x3 : les 9 metriques cles en un coup d'oeil."""
    periods = sorted(agg_all.keys())
    indicators = [
        ("ttr",                 "TTR",                 "Score"),
        ("mattr",               "MATTR",               "Score"),
        ("mtld",                "MTLD",                "Score"),
        ("avg_sent_length",     "Long. phrases",       "Tokens"),
        ("clause_density",      "Densite clausale",    "Clauses/phrase"),
        ("flesch_reading_ease", "Flesch",              "Score"),
        ("lix",                 "LIX",                 "Score"),
        ("information_density", "Densite info",        "Score"),
        ("topic_concentration", "Concentration them.", "Score"),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(20, 14))
    fig.patch.set_facecolor(COLORS["bg"])
    fig.suptitle("Dashboard — Indicateurs Cognitifs Linguistiques",
                 fontsize=13, fontweight="bold", y=1.01)

    configs = [
        (agg_all,     "Global",  COLORS["text"],      1.8),
        (agg_press,   "Presse",  COLORS["primary"],   1.3),
        (agg_twitter, "Twitter", COLORS["secondary"], 1.3),
    ]

    for ax, (key, title, ylabel) in zip(axes.flat, indicators):
        for agg, label, color, lw in configs:
            dts, vals = _series(agg, key)
            if dts:
                ax.plot(dts, vals, color=color, lw=lw, marker="o",
                        markersize=2.5, label=label, alpha=0.85)
        _style(ax, title, ylabel)
        _add_scandals(ax, periods, scandals)
        ax.legend(fontsize=6, loc="upper right")

    _save(fig, output_path, show)
    return fig


def plot_press_vs_twitter_bars(
    agg_press, agg_twitter,
    output_path=None, show=False,
) -> plt.Figure:
    """VIZ 8 — Profil cognitif moyen : barres groupees Presse vs Twitter."""
    metrics_to_compare = [
        ("ttr",                 "TTR"),
        ("mattr",               "MATTR"),
        ("mtld",                "MTLD"),
        ("hapax_ratio",         "Hapax ratio"),
        ("avg_sent_length",     "Long. phrase"),
        ("clause_density",      "Densite clausale"),
        ("flesch_reading_ease", "Flesch"),
        ("lix",                 "LIX"),
        ("information_density", "Densite info"),
        ("topic_concentration", "Conc. them."),
    ]

    def corpus_mean(agg: Dict[str, Dict], key: str) -> float:
        vals = [
            v[key] for v in agg.values()
            if v.get(key) is not None and isinstance(v.get(key), (int, float))
        ]
        return round(mean(vals), 3) if vals else 0.0

    labels       = [lbl for _, lbl in metrics_to_compare]
    press_vals   = [corpus_mean(agg_press,   k) for k, _ in metrics_to_compare]
    twitter_vals = [corpus_mean(agg_twitter, k) for k, _ in metrics_to_compare]

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(16, 5))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    bars_p = ax.bar(x - width / 2, press_vals,   width,
                    label="Presse",  color=COLORS["primary"],   alpha=0.85)
    bars_t = ax.bar(x + width / 2, twitter_vals, width,
                    label="Twitter", color=COLORS["secondary"], alpha=0.85)

    for bar in list(bars_p) + list(bars_t):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                f"{h:.1f}", ha="center", va="bottom", fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=15, ha="right")
    ax.set_ylabel("Valeur moyenne (corpus entier)", fontsize=9)
    ax.set_title("Profil cognitif moyen — Presse vs Twitter", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=COLORS["grid"], alpha=0.4, linewidth=0.6)

    _save(fig, output_path, show)
    return fig


# ─────────────────────────────────────────────────────────────────
# 11. EXPORT JSON
# ─────────────────────────────────────────────────────────────────

def export_json(
    doc_metrics: List[Dict],
    agg: Dict[str, Dict[str, Dict]],
    output_path: str,
) -> None:
    """Exporte doc-level + agregations en JSON structure."""
    class _Enc(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):  return int(obj)
            if isinstance(obj, (np.floating,)): return float(obj)
            if isinstance(obj, datetime):        return obj.isoformat()
            return super().default(obj)

    payload = {
        "generated_at": datetime.now().isoformat(),
        "n_documents":  len(doc_metrics),
        "aggregated":   agg,
        "documents":    doc_metrics,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, cls=_Enc)

    size_kb = os.path.getsize(output_path) // 1024
    print(f"  OK {os.path.basename(output_path)} ({len(doc_metrics):,} docs, ~{size_kb} KB)")


# ─────────────────────────────────────────────────────────────────
# 12. PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────

def run_linguistic_analysis(
    corpus: List[Dict],
    output_dir: str = "outputs/reports",
    show_plots: bool = False,
    scandals: Dict[str, str] = SCANDALS_MCDO,
) -> Dict:
    """
    Pipeline complet Phase 5.1 :
      1. Metriques document par document (5 axes, 20 indicateurs)
      2. Agregation par periode x source (all / press / twitter)
      3. Export JSON
      4. 8 visualisations matplotlib
    Retourne le dict des agregations pour usage programmatique.
    """
    output_dir = os.getenv("LINGUISTIC_OUTPUT_DIR", output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Etape 1 : calcul par document
    print(f"\n[Phase 5.1] Calcul des metriques — {len(corpus):,} documents...")
    print("  (MTLD peut prendre quelques minutes sur 100k docs)")
    doc_metrics, skipped = [], 0
    for doc in corpus:
        if not doc.get("text", "").strip():
            skipped += 1
            continue
        doc_metrics.append(compute_document_metrics(doc))
    print(f"  -> {len(doc_metrics):,} traites | {skipped} ignores (vides)")

    # Etape 2 : agregation
    print("\n[Phase 5.1] Agregation par periode x source...")
    agg         = aggregate_all_sources(doc_metrics)
    agg_all     = agg["all"]
    agg_press   = agg["press"]
    agg_twitter = agg["twitter"]
    print(f"  -> {len(agg_all)} periodes YYYY-MM")

    # Apercu terminal
    if agg_all:
        print("\n  Apercu metriques globales (moy. corpus) :")
        for key in ["ttr", "mattr", "mtld", "avg_sent_length",
                    "flesch_reading_ease", "lix", "information_density"]:
            vals = [v[key] for v in agg_all.values() if v.get(key) is not None]
            if vals:
                print(f"    {key:25s}: {round(mean(vals), 3)}")

    # Etape 3 : export JSON
    json_path = os.path.join(output_dir, "linguistic_metrics_results.json")
    print("\n[Phase 5.1] Export JSON...")
    export_json(doc_metrics, agg, json_path)

    # Etape 4 : visualisations
    print("\n[Phase 5.1] Generation des visualisations...")

    plot_lexical_richness(
        agg_all, agg_press, agg_twitter, scandals,
        output_path=os.path.join(output_dir, "lm_01_ttr_mattr_mtld.png"),
        show=show_plots,
    )
    plot_lexical_richness_extra(
        agg_all, agg_press, agg_twitter, scandals,
        output_path=os.path.join(output_dir, "lm_02_yulek_hapax.png"),
        show=show_plots,
    )
    plot_syntactic_complexity(
        agg_all, agg_press, agg_twitter, scandals,
        output_path=os.path.join(output_dir, "lm_03_syntactic.png"),
        show=show_plots,
    )
    plot_readability(
        agg_all, agg_press, agg_twitter, scandals,
        output_path=os.path.join(output_dir, "lm_04_readability.png"),
        show=show_plots,
    )
    plot_information_density(
        agg_all, agg_press, agg_twitter, scandals,
        output_path=os.path.join(output_dir, "lm_05_information_density.png"),
        show=show_plots,
    )
    plot_coherence(
        agg_all, agg_press, scandals,
        output_path=os.path.join(output_dir, "lm_06_coherence.png"),
        show=show_plots,
    )
    plot_dashboard(
        agg_all, agg_press, agg_twitter, scandals,
        output_path=os.path.join(output_dir, "lm_07_dashboard.png"),
        show=show_plots,
    )
    plot_press_vs_twitter_bars(
        agg_press, agg_twitter,
        output_path=os.path.join(output_dir, "lm_08_press_vs_twitter.png"),
        show=show_plots,
    )

    print(f"\n[Phase 5.1] Termine — fichiers dans : {output_dir}")
    return agg


# ─────────────────────────────────────────────────────────────────
# 13. CHARGEMENT DES JSON
# ─────────────────────────────────────────────────────────────────

def load_corpus(press_path: str, twitter_path: str) -> List[Dict]:
    """
    Charge le corpus multimodal depuis les deux fichiers JSON.
    press_scrapper.json  -> {"meta": {...}, "articles": [...]}
    twitter_scraper.json -> {"meta": {...}, "tweets": [...]}
    """
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
            year_month = datetime.strptime(art.get("date", "")[:10], "%Y-%m-%d").strftime("%Y-%m")
        except (ValueError, TypeError):
            continue
        corpus.append({
            "id": doc_id, "source": "press",
            "platform": art.get("source", ""), "language": "en",
            "category": "", "date": year_month,
            "title": art.get("title", ""), "text": text,
        })
        doc_id += 1

    print(f"    -> {doc_id:,} articles charges")
    press_count = doc_id

    print(f"  Chargement Twitter : {twitter_path}")
    with open(twitter_path, encoding="utf-8") as f:
        raw = json.load(f)
    tweets = raw.get("tweets", raw) if isinstance(raw, dict) else raw

    for tw in tweets:
        text = tw.get("text", "").strip()
        if not text:
            continue
        year_month = tw.get("year_month", "")
        if not year_month:
            try:
                year_month = datetime.strptime(tw.get("date", "")[:10], "%Y-%m-%d").strftime("%Y-%m")
            except (ValueError, TypeError):
                continue
        corpus.append({
            "id": doc_id, "source": "twitter",
            "platform": "Twitter/X", "language": "en",
            "category": "", "date": year_month,
            "handle": tw.get("handle", ""), "text": text,
        })
        doc_id += 1

    print(f"    -> {doc_id - press_count:,} tweets charges")
    print(f"  Total corpus : {len(corpus):,} documents")
    corpus.sort(key=lambda x: x["date"])
    return corpus


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    PRESS_PATH   = r"C:\Users\GAB Informatique\Documents\Master 2\PFE\plateforme_analyse_textuelle\datasets\press_scrapper.json"
    TWITTER_PATH = r"C:\Users\GAB Informatique\Documents\Master 2\PFE\plateforme_analyse_textuelle\datasets\twitter_scraper (1).json"
    OUTPUT_DIR   = r"C:\Users\PC\Documents\PFE\plateforme_analyse_textuelle\src\cognitive\outputs\reports"

    corpus = load_corpus(PRESS_PATH, TWITTER_PATH)

    run_linguistic_analysis(
        corpus,
        output_dir=OUTPUT_DIR,
        show_plots=False,
        scandals=SCANDALS_MCDO,
    )
