# Thread R — Confusion Buffer & Anki Pack (Research Output, Networking & Applications)
### Companion to the Threads section (runs from the Phase-1 self-test, ~2 h/week in the Evening). ~110 hrs — this is the thread that turns study into *evidence*: reproduced papers, write-ups, a preprint/workshop paper, open-source PRs, relationships, and a live application pipeline.

**How to use:** a weekly rhythm — monthly reproduce-a-paper + write-up, 1–2 short posts/month, aim one capstone at a venue, and run the application ladder. Cards here are process/strategy (like the interview pack). **Triangulation targets:** *the three-pass paper-reading method*, *what a workshop paper actually needs*, and *why a merged PR beats a blog post*.

## Ranked focus
1. **Reproduce-a-paper** — the referral hook + level-proof + best way to learn a subfield.
2. **Publishing** — one clean, reproducible workshop result > many half-finished ideas.
3. **Open-source contribution** — a merged PR to a core library is the strongest signal.
4. **Applications pipeline** — the tier ladder, tailored resumes, calibration.
5. **Networking** — reproduce → note → engage (genuine, not spammy).

## Anki deck (`Q → A`)

### Deck A · Reading & reproducing papers
- **Q:** The three-pass method (Keshav)? → **A:** Pass 1 (~5 min): title/abstract/intro/headings/conclusion — get the gist + decide relevance. Pass 2 (~1 h): figures, method, results — grasp the content, mark unknowns. Pass 3 (hours): re-implement/derive to fully understand + critique.
- **Q:** How do you reproduce a paper efficiently? → **A:** Start from the **official code** (get their numbers first), then re-implement the core yourself; only then ablate/extend — don't re-derive everything from zero if a reference exists.
- **Q:** What do you log in a reproduction write-up? → **A:** Exact setup/seeds, where your numbers matched or diverged, the gotchas/undocumented details, and one figure reproducing a key result — the "deltas" are the value.
- **Q:** Tools for paper discovery + context? → **A:** Papers with Code (SOTA + code), Connected Papers / Semantic Scholar (citation graph), and following a target lab's authors.
- **Q:** How should you read for *this* program? → **A:** With a question in mind (what does this let me build/claim?), mapping each paper to a phase capstone or a Thread-M primitive.

### Deck B · Writing & communication
- **Q:** Structure of a workshop paper? → **A:** Abstract → intro (problem + contribution) → related work → method → experiments (+ ablations) → limitations → conclusion; 4–8 pages typical.
- **Q:** What makes a method section clear? → **A:** State the problem formally, define notation once, give the key equation/algorithm, and a figure of the architecture/pipeline — reader should be able to reimplement it.
- **Q:** What ablations actually convince reviewers? → **A:** Remove each contribution one at a time to show it matters; compare against a strong baseline (not a strawman); report variance across seeds.
- **Q:** How to present results honestly? → **A:** Error bars / multiple seeds, matched compute, the *right* metric (Metrics Reloaded), and a candid limitations section — honesty reads as competence.
- **Q:** The "why me / why digital biology" narrative? → **A:** One coherent thread linking imaging + molecular + physics (+ quantum), each backed by a public artifact, plus genuine curiosity about the biology/chemistry.
- **Q:** Value ordering: blog post vs preprint vs merged PR? → **A:** A merged PR to a core library > a reproduced paper + write-up > a preprint > a blog post — impact and verifiability rise in that order.
- **Q:** What makes a good repo README? → **A:** One-line what/why, install, a runnable example, results table, and how to reproduce — so a stranger (or recruiter) can run it in minutes.

### Deck C · Publishing & venues
- **Q:** Which venues for your two tracks? → **A:** Imaging → **MIDL / MICCAI** workshops; molecular → **MLSB / LoG** (and AI4Science). Preprints → arXiv / bioRxiv.
- **Q:** Workshop vs main-track — the trade-off? → **A:** Workshops = lower bar, faster, great for a clean single result / first paper; main-track = higher bar/effort/prestige. Start with a workshop.
- **Q:** What does a workshop paper minimally need? → **A:** One clean, reproducible result with a proper baseline + ablation + honest limitations — novelty can be modest if the execution is solid.
- **Q:** How does OpenReview work? → **A:** Submissions + reviews (often public/semi-anonymous) + author rebuttal; read prior reviews of a venue to calibrate expectations.
- **Q:** Preprint norms? → **A:** Post to arXiv/bioRxiv when the result is solid and you want a citable timestamp; check any employer external-publication review first.

### Deck D · Open-source & networking
- **Q:** Why is a merged PR the strongest signal? → **A:** It's externally verified competence in real code that a lab actually uses (e.g., e3nn / Chemprop / MONAI / DeepChem / PennyLane) — beats any self-published artifact.
- **Q:** How to find a first contribution? → **A:** Look for "good first issue"/"help wanted" labels, fix a doc/bug/test, or add a small feature you needed while reproducing a paper.
- **Q:** How to engage with a target lab authentically? → **A:** Reproduce their work → write it up → share a specific, useful note/finding (not "please hire me") → a genuine technical conversation follows.
- **Q:** Building advisor relationships (for PhD-later)? → **A:** Over the program, cultivate 1–2 relationships via reproductions/PRs/thoughtful questions — the basis of a future application.
- **Q:** Conference/community presence? → **A:** Attend (even virtually), post reproductions/threads, ask good questions — visibility compounds and creates referral surface.

### Deck E · Applications pipeline
- **Q:** The tier ladder (0→3)? → **A:** Tier 0 = India imaging AI (from Wk 3); Tier 1 = broader medical/phenomics imaging + video; Tier 2 = drug-discovery labs (DESRES/Isomorphic/DeepMind-science) after the molecular track; Tier 3 = full top-lab loops after the crescendo.
- **Q:** How do you tailor per tier? → **A:** Lead with the tier-relevant capstones/skills (imaging-forward vs molecular-forward resume variants); keep everything public and defensible (COI).
- **Q:** Why treat early loops as calibration? → **A:** They're paid practice — collect the format/feedback; every rejection is data on the bar and your gaps.
- **Q:** Application cadence? → **A:** ~3–5 *quality* applications/week (fit + tailored), tracked in a sheet, with the reproduce-a-paper as the referral hook.
- **Q:** The COI rule for everything public? → **A:** Public data + generic methods only; no proprietary codenames/targets/teammate work in any artifact — that's what keeps the portfolio defensible.

### Deck F · Cadence & discipline (the thread)
- **Q:** The weekly/monthly rhythm? → **A:** ~2 h/week: monthly a reproduce-a-paper + write-up; 1–2 short posts/month; continuously run the ladder; aim ≥1 capstone at a venue over the program.
- **Q:** The single highest-leverage habit here? → **A:** Reproduce-a-paper monthly — it deepens a subfield, produces a public artifact, and is the natural referral/interview hook.

## Common misconceptions & traps
- **"You need a PhD to publish."** Workshops (MIDL/MLSB/LoG) accept strong industry/independent work with a clean result.
- **"A polished blog post impresses top labs."** Reproductions and merged PRs impress far more — verifiable competence over marketing.
- **"Networking is spammy self-promotion."** Reproduce-then-engage with a specific finding is genuine and welcome; "please hire me" cold messages aren't.
- **"More applications = more offers."** Fit + tailoring + a referral beat volume; 5 quality > 50 generic.
- **"Wait until the work is perfect."** Ship a clean, honest, reproducible result — a modest, solid contribution beats a grand unfinished one.

## Glossary starter
three-pass reading · reproduce-a-paper / deltas · Papers with Code / Connected Papers · ablation · error bars/seeds · limitations section · workshop vs main-track · MIDL/MICCAI/MLSB/LoG · arXiv/bioRxiv · OpenReview / rebuttal · merged PR (core library) · good-first-issue · reproduce→note→engage · advisor relationship · tier ladder (0–3) · resume tailoring · loop calibration · application cadence · COI (public/defensible) · "why digital biology" narrative.

## Drills
**Monthly:** reproduce one target-lab paper end-to-end + a public write-up (setup, deltas, one reproduced figure).
**Ongoing:** open one PR to a core library; draft the two job-talk decks; maintain the application tracker; write one paper-critique.
**Milestone:** submit ≥1 phase capstone to a workshop (MIDL/MLSB/LoG) over the program.

## Leaving bar (checkpoint, not a one-time gate)
A monthly cadence of reproduced-paper + write-up; ≥1 merged PR to a core library; a submitted (or submission-ready) workshop paper from a capstone; a coherent public narrative + tailored resume variants; and an active, tracked application pipeline.
