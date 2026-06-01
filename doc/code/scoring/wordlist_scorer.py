# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.0
# ---

# %% [markdown]
# # WordList Scorer
#
# The `WordListScorer` flags responses that contain any term from a curated wordlist. It compiles all terms into a
# single word-boundary regex alternation, so one search covers the whole list and partial-word false positives like
# `ass` inside `class` are avoided. No LLM call is involved.
#
# PyRIT does not redistribute the lexicons themselves. `load_predefined_wordlist` fetches each list from its upstream
# maintainer (Shutterstock's LDNOOBW for profanity, NVIDIA's garak for the Ofcom potentially-offensive list, the
# reclaimed-slurs seed, and the Surge AI profanity mirror) on first use and caches the file under
# `DB_DATA_PATH / "lexicons"`. Upstream commit SHAs are pinned, so a given PyRIT release always fetches the same data.
#
# **Caveats**
# - Matching is context-blind. A "slur" hit can also be a quote, an in-group reclamation, an academic reference, or a
#   reported insult. Treat hits as a signal to review, not as ground truth.
# - The Ofcom "general" bucket at the default `ofcom_min_strength=2` includes mild words like *Bullshit* that are about
#   tone, not safety. Pass a higher `ofcom_min_strength` if you want only strong language.
# - The Surge AI profanity list was not published with an explicit license by Surge AI. PyRIT fetches it from
#   NVIDIA/garak's mirror at a pinned commit rather than redistributing it. Downstream use is the caller's
#   responsibility.

# %%
from pyrit.score import (
    PredefinedWordList,
    WordListMatchMode,
    WordListScorer,
    load_predefined_wordlist,
)
from pyrit.setup import IN_MEMORY, initialize_pyrit_async

await initialize_pyrit_async(memory_db_type=IN_MEMORY)  # type: ignore

# %% [markdown]
# ## Available predefined lists
#
# Use `PredefinedWordList` to pick one of the curated lists. The Ofcom entries correspond to the six category buckets
# that Ofcom's audience research splits offensive language into.

# %%
for entry in PredefinedWordList:
    print(entry.name)

# %% [markdown]
# ## Loading a list and constructing a scorer
#
# `load_predefined_wordlist` returns the terms; pass them into `WordListScorer` with the category label you want
# attached to any hit.

# %%
ldnoobw_terms = load_predefined_wordlist(wordlist=PredefinedWordList.LDNOOBW_EN)
print(f"LDNOOBW EN term count: {len(ldnoobw_terms)}")

profanity_scorer = WordListScorer(terms=ldnoobw_terms, category="profanity")

# A response that quotes a profane reply
flagged_response = "The model replied: 'you bitch'"
clean_response = "The model declined politely."

flagged_score = (await profanity_scorer.score_text_async(text=flagged_response))[0]  # type: ignore
clean_score = (await profanity_scorer.score_text_async(text=clean_response))[0]  # type: ignore

print(f"Flagged response: detected={flagged_score.get_value()}, category={flagged_score.score_category}")
print(f"Clean response:   detected={clean_score.get_value()}")

# %% [markdown]
# ## Word boundaries vs. substring matches
#
# By default, terms are anchored on word boundaries (`\b`), so `ass` does not fire inside `class`. Pass
# `match_mode=WordListMatchMode.SUBSTRING` if you specifically want embedded-match behaviour.

# %%
word_scorer = WordListScorer(terms=["ass"], category="profanity")
substring_scorer = WordListScorer(
    terms=["ass"],
    category="profanity",
    match_mode=WordListMatchMode.SUBSTRING,
)

text = "I have a class today"
print(f"WORD mode      on {text!r}: {(await word_scorer.score_text_async(text=text))[0].get_value()}")  # type: ignore
print(f"SUBSTRING mode on {text!r}: {(await substring_scorer.score_text_async(text=text))[0].get_value()}")  # type: ignore

# %% [markdown]
# ## Ofcom categories with a strength filter
#
# The Ofcom list ships with a 1–4 strength rating per term (1=mild, 4=strongest). The default `ofcom_min_strength=2`
# mirrors garak's runtime filter. Raise it to drop tone-only terms like *Bullshit*.

# %%
ofcom_terms = load_predefined_wordlist(
    wordlist=PredefinedWordList.OFCOM_RACEETHNIC,
    ofcom_min_strength=3,
)
print(f"Ofcom race/ethnic terms at strength>=3: {len(ofcom_terms)}")

# %% [markdown]
# ## Reclaimed slurs — read the caveats
#
# The reclaimed-slurs list is a seed of recognised slur tokens curated by NVIDIA/garak. A hit means a token is present,
# nothing more — the same word can appear in academic discussion, reporting, in-group reclamation, or as a slur. Use
# the scorer as a triage signal and always combine with human review for this category.

# %%
slur_terms = load_predefined_wordlist(wordlist=PredefinedWordList.RECLAIMED_SLURS_EN)
print(f"Reclaimed-slurs seed term count: {len(slur_terms)}")

# %% [markdown]
# ## Surge AI profanity — categorical buckets with severity
#
# The Surge AI obscenity list groups 1,600+ English profanities into 11 categories (sexual anatomy / sexual acts,
# bodily fluids / excrement, sexual orientation / gender, racial / ethnic slurs, mental disability, physical
# disability, physical attributes, animal references, religious offense, political, and a catch-all
# *other / general insult* bucket). A single term can belong to up to three categories at once; PyRIT includes a
# term in a category if any of its three category slots match. Each term also carries a `severity_rating` (mean of
# five 1–3 human ratings); the default `surge_min_severity=1.0` includes everything, matching garak's loader.
#
# Surge AI did not publish this file with an explicit license. PyRIT fetches it from NVIDIA/garak's mirror at a pinned
# commit rather than redistributing it — downstream use is your responsibility.

# %%
surge_terms = load_predefined_wordlist(
    wordlist=PredefinedWordList.SURGE_RACIAL_ETHNIC_SLURS_EN,
    surge_min_severity=2.0,
)
print(f"Surge racial/ethnic slurs at severity>=2.0: {len(surge_terms)}")
