# Reviewer-pool coverage report

**Question:** which paper topics does the *Human Factors* editorial board cover
weakly, and do the 87 `Not_EB` candidate authors close those gaps?

**Method.** Cosine similarity (L2-normalized MiniLM embeddings, the same the app
uses) between each of the 976 background papers and each reviewer profile. A
paper is "weakly covered" if its **maximum similarity to any editorial-board
member (the 75 AE/EB/PR/R reviewers) ≤ 0.40**. Corpus is post-cleanup (editorial
boilerplate removed).

## Headline

The board covers the corpus well: **56 of 976 papers (5.7%)** are weakly covered
(≤ 0.40), and only **1** falls below 0.25. Adding the 87 `Not_EB` candidates
closes most of the gap:

| Weak papers (board ≤ 0.40) | a Not_EB lifts above… |
|---|---|
| 56 total | **0.25:** 55/56 |
| | **0.30:** 53/56 |
| | **0.40:** 29/56 |

Mean best-similarity on the weak set rises **0.356 → 0.431** when the candidates
are included. The three domain gaps below account for most of it.

## Gap 1 — Clinical auditory alarms & patient-monitoring sonification

The board handles *automotive/aviation* auditory warnings well (Sanderson, Vu,
Helton, Feng, Boyle), but the **medical-device alarm** sub-area — spearcons,
earcons, IEC 60601-1-8 alarm standards, neonatal/ICU monitoring — tops out around
0.35–0.43 (Sanderson is the only real anchor). This is the **largest closable
gap**, and the candidates close it decisively.

| Paper (truncated) | Board best | Not_EB best |
|---|---|---|
| Spearcon Sequences for Monitoring Multiple Patients | Sanderson 0.37 | **Mohamed 0.72** |
| Spearcons for Patient Monitoring | Sanderson 0.40 | **Mohamed 0.73** |
| Smooth or Stepped? Enhanced Sonifications | Sanderson 0.42 | **Loeb 0.69** |
| Head-Worn Displays & Strategic Alarm Management | Helton 0.55 | **Brecknell 0.73** |
| Masking Between Reserved Alarm Sounds (IEC 60601-1-8) | Sanderson 0.35 | **Brecknell 0.49** |

**Fillers:** Ismail Mohamed, Birgit Brecknell, Robert G. Loeb (all from the
clinical-monitoring research line). The weak-set auditory cluster goes **6/6
above 0.40, +0.19 mean** — the cleanest "gap closed" result.

## Gap 2 — Occupational anthropometry (body-dimension surveys)

Body-measurement studies — law-enforcement officer anthropometry, high-BMI
populations, spaceflight anthropometry — are weakly covered (Acosta-Sojo /
Armstrong ≈ 0.35–0.39). (Note: *anthropomorphism* papers are a different topic
and are well covered by J. D. Lee.)

| Paper (truncated) | Board best | Not_EB best |
|---|---|---|
| Encumbered & Traditional Anthropometry of Law Enforcement | Armstrong 0.38 | **Hsiao 0.65** |
| Anthropometric Dimensions of High-BMI Individuals | Acosta-Sojo 0.39 | **Hsiao 0.61** |
| National Anthropometry Study of Law Enforcement (Needs/Procedures) | Armstrong 0.35 | **Hsiao 0.55** |
| Anthropometric Changes in Spaceflight | Acosta-Sojo 0.35 | **Hsiao 0.43** |

**Filler:** Hongwei Hsiao (NIOSH anthropometry) single-handedly closes this gap
(jumps to 0.43–0.65). Steven Fischer and Robert Seibt add occupational-ergonomics
depth.

## Gap 3 — EV charging-infrastructure reliability

EV *human factors* (eco-driving, range/energy displays, EV auditory experience)
are well covered (Horrey ≈ 0.51–0.54, Krems 0.58–0.59). The **only** genuine gap
is a single paper on charging-*infrastructure* reliability:

| Paper | Board best | Not_EB best |
|---|---|---|
| Reliability of Open Public EV DC Fast Chargers | Alsaid **0.22** | Krems **0.28** |

This is the one weak paper the candidates do **not** close (still < 0.40):
public-charger reliability/uptime engineering sits outside both the board's and
the candidates' expertise. If this topic matters editorially, it needs a
dedicated infrastructure-reliability reviewer.

## Bottom line

Adding the `Not_EB` candidates **closes the auditory-alarm and anthropometry gaps
outright** (Mohamed/Brecknell/Loeb and Hsiao respectively) and substantially
lifts the long tail (53/56 weak papers reach ≥ 0.30). The lone unresolved gap is
EV charging-infrastructure reliability — a true white space in the pool.

_Generated from `data/members_enriched.csv` + `Human_Factors_*abstracts*.json`.
Similarity is LLM-embedding-based and indicative, not a substitute for editorial
judgment._
