# Reviewer-profile research task (Not_EB authors)

You research prolific *Human Factors* journal authors and produce a structured
reviewer profile for each, matching the existing reviewer dataset's format.

## Input
Read `/Users/jdlee/_Code/HF-Reviewer-Finder/data/not_eb_seed.json` — a list of
authors. You are assigned a contiguous range of **indices** (given in your prompt).
Process ONLY your assigned indices. Each seed entry has:
- `name`: author name as it appears in bylines (e.g. "Patricia R. DeLucia")
- `hf_papers`: that author's REAL papers in the journal *Human Factors* (title,
  date, doi, citation_count). These DOIs are already verified-real — use them as
  (a) ground truth to identify/disambiguate the correct researcher, and (b)
  candidate recent/seminal entries.

## What to produce per author
Disambiguate the correct researcher using the hf_papers as an anchor (these are
genuinely their papers). Confirm identity/affiliation via web search (Google
Scholar profile, university faculty page, ORCID, ResearchGate).

Then research and write these fields (mirror the STYLE of the example below):

- `name`: canonical **"Last, First M."** format (handle compound surnames
  correctly: "van Paassen, M. M.", "de Winter, Joost", "Ng Boyle, Linda").
- `rank`: exactly `"Not_EB"`
- `found`: `"yes"` if you confidently identified the person, else `"no"`
- `recent_1`, `recent_2`, `recent_3`: their three most RECENT papers (prefer
  2023–2026), any venue. Full citation: Authors (Year). Title. Venue, vol(iss),
  pages. DOI: 10.xxxx/...  May include hf_papers.
- `recent_1_synopsis` … `recent_3_synopsis`: 2–3 sentences — what the paper did
  and why it matters.
- `seminal_1`, `seminal_2`: their two most-cited / most influential papers (any
  era). Same citation format.
- `seminal_1_synopsis`, `seminal_2_synopsis`: 2–3 sentences; INCLUDE the
  approximate citation count and why the work is foundational.
- `expertise_overview`: 4–6 sentences describing their **research interests** —
  topics, methods, application domains. This is NOT a biography: no career
  history, no titles/positions, no "Professor at X". Describe what they study.
- `notes`: identification confidence; which sources confirmed identity; the
  source and date of citation counts; and explicitly flag any citation you could
  NOT verify.

## DOI VERIFICATION (required)
For EVERY DOI you cite (recent + seminal), confirm it resolves before including it:
fetch `https://api.crossref.org/works/<DOI>` — a 200 with matching title means
verified. If a DOI does not resolve or you cannot find one, either pick a
different verifiable paper or include it WITHOUT a fabricated DOI and flag it in
`notes`. NEVER invent a DOI. The hf_papers DOIs are pre-verified; reuse freely.

## Output
Write ONE JSON file per author (a single JSON object with exactly the fields
above) to:
`/Users/jdlee/_Code/HF-Reviewer-Finder/data/not_eb_parts/<index>_<lastnameslug>.json`
where `<index>` is the seed index (zero-padded 2 digits) and `<lastnameslug>` is
the lowercase surname (ascii, no spaces/punct). Example: `02_delucia.json`.

Use `ensure_ascii=false`-equivalent UTF-8 (write accented characters directly).

## Example (existing reviewer — match this depth and tone)
{
  "name": "Acosta-Sojo, Yadrianna",
  "rank": "Not_EB",
  "found": "yes",
  "recent_1": "Outlaw, L., Acosta-Sojo, Y., Schall, M. C., Purdy, G. T., & Sesek, R. F. (2025). Comparing the effects of user preference and experience when using passive low back exoskeletons on physical and cognitive load while performing simulated manufacturing tasks. Proceedings of the Human Factors and Ergonomics Society Annual Meeting. DOI: 10.1177/10711813251369813",
  "recent_1_synopsis": "This study evaluated how user preference and training experience interact with passive low-back exoskeleton use during simulated manufacturing work, measuring both physical (biomechanical/EMG) and cognitive load outcomes. It provides empirical guidance on fitting and adoption strategies for occupational exoskeletons.",
  "seminal_1": "Peng, X., Acosta-Sojo, Y., Wu, M. I., & Stirling, L. (2022). Actuation timing perception of a powered ankle exoskeleton and its associated ankle angle changes during walking. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 30, 869–877. DOI: 10.1109/TNSRE.2022.3162213",
  "seminal_1_synopsis": "This paper quantified how accurately users perceive actuation timing in a powered ankle exoskeleton and how those perceptions correlate with ankle kinematics. It is her most-cited work (~45 citations) and establishes perceptual benchmarks for intuitive human-exoskeleton control.",
  "expertise_overview": "Occupational human factors and human-exoskeleton interaction—how users perceive, adapt to, and are physically and cognitively burdened by powered and passive exoskeletons in manufacturing and rehabilitation. Wearable-sensor exposure assessment (surface EMG, inertial measurement) within systems-level frameworks for gauging physical work demands and musculoskeletal disorder risk. Perceptual accuracy of exoskeleton actuation timing and inter-individual variability in muscle activation, motivating personalized human-in-the-loop assistance.",
  "notes": "Identified via Google Scholar (user=Mg3K5a4AAAAJ) and Auburn University faculty page. Citation counts from Google Scholar as of mid-2025. The 2025 HFES paper DOI confirmed via Crossref."
}

## Return
After writing your files, return a brief plain-text summary: which indices you
completed, any author you marked found="no", and any unverifiable citations.
