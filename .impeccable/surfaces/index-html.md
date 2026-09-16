---
version: 1
slug: "index-html"
primary_target: "index.html"
related_targets: []
---

## Scope & mode

Single scrolling page (`index.html`), Read mode. Static HTML/CSS/JS, no build step, deploys to GitHub Pages.

## Audience, job, action, proof, constraints

Speech/audio-ML researchers and reviewers skimming the AIP-Speech benchmark paper. Job: understand the contribution and headline findings fast, then reach the PDF, the GitHub code, or the BibTeX. Proof: real tables/figures from the paper (Tables 1-5, Figures 1-7) rendered as real charts, not invented numbers. Constraint: no live demo, no audio playback.

## Direction contract

**THESIS:** The paper as a calibrated instrument report, not a marketing page — every claim reads like a measurement with a traceable source, refusing the generic centered-hero-plus-card-grid academic-page default.

**OWN-WORLD:** Warm cream ground (#F6F3EC) with a slightly lighter panel tone (#FAF8F2), near-black ink (#1B1812, never pure black), one brick-red accent (#B5432B) used only for links, active states, and figure callouts. A serif built for dense text and equations for body/headings (Source Serif 4 or Newsherman-class face), paired with one monospace (IBM Plex Mono) for labels, captions, table headers, and data. Numbered sections ("01 Benchmark", "02 Descriptors", "03 Results", "04 Fairness", "05 Limitations"). Hairline 1px rules (#C9C7C4-ish) divide regions instead of shadows or cards. A boxed abstract at the top styled like a paper's own abstract block. Every chart/table gets a mono-uppercase caption line and small registration tick marks at its corners, so figures read as calibrated instrument output.

**STORY:** Visitor lands, reads the one-line finding in under 10 seconds, sees the benchmark pipeline as a diagram, explores the descriptor-correlation results interactively (a model/descriptor selector, matching the reference site's interactive-table pattern), scans fairness and prompt-steering results, hits limitations, and leaves via PDF/GitHub/BibTeX.

**FIRST VIEWPORT:** Corner stamp "TR · AIP-SPEECH" top-left. Title set large in the serif, authors and affiliation below in smaller mono caps, a boxed one-paragraph abstract beneath, three inline action links (Paper PDF · GitHub · BibTeX) styled as underlined text not buttons, and a single bolded one-line key finding ("Simple spectral descriptors — ZCR, spectral centroid, spectral rolloff — consistently predict how much a background recording degrades Speech LLM performance.") sitting directly under the abstract box.

**FORM:** Assigned direction 5 of 7 grounded candidates (concept-seed key `feb22bd7`, mode `read`), raised with one borrowed discipline from a competing catalog challenger (design-annual plate-section system): mono-uppercase caption labels and registration-mark precision on every figure/table.

**FINISH:** unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance.

## Unresolved decisions

- Exact serif/mono font pairing to load from Google Fonts (will finalize during build).
- Venue/arXiv ID/publication date remain TBD placeholders in the BibTeX block until the user supplies them.
