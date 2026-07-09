---
name: annotated-review-html
description: >-
  Create a bespoke self-contained HTML review companion with annotations and
  copyable markdown feedback for plans, specs, deep dives, app features, code
  areas, diffs, audits, and handoff material. Use when a user wants an
  ingestible walkthrough, annotated review surface, feedback collector, signoff
  companion, or structured review page.
---

# Annotated Review HTML

Create a static HTML review companion that helps a human understand source
material, annotate meaningful review units, and export feedback markdown back
into chat.

This is an authoring skill, not a markdown converter. The source material is
evidence, not the page outline.

## Mode Selection

If the user does not specify a mode, ask one concise chat question before
creating the page:

```text
Which mode should I use?

- review: general section-by-section review.
- digest: easier comprehension pass for a dense document.
- signoff: approval-focused pass for decisions, risks, and blockers.
- exec: implementation-focused pass for phases, tasks, and validation.

I recommend review unless you want a more focused pass.
```

Do not use terminal prompts.

## Workflow

1. Read the source material first.
   - For a document: read it end to end.
   - For a feature/app area: inspect the relevant docs, code, routes, services,
     data models, screenshots, or browser state needed to understand it.
   - For a diff/branch: inspect the actual diff and high-signal changed files.
2. Decide the review units yourself.
   - Do not mechanically mirror markdown headings.
   - Group content by what the user needs to understand and review.
   - Prefer decision/risk/workflow units over source-file structure.
3. Author a tailored HTML walkthrough.
   - Use `references/design.md` as the visual and interaction contract.
   - Use `assets/review-template.html` only as a pattern library, not as a
     required fill-in template.
   - Keep it self-contained: embedded CSS and JavaScript, no external assets.
4. Add annotations to each meaningful review unit.
   - Status dropdown.
   - Freeform note textarea.
   - Local save state.
   - Clear-this-note action.
5. Add export controls.
   - Copy feedback markdown.
   - Preview markdown fallback.
   - Download markdown fallback.
   - Clear all notes.

## Review Unit Guidance

Good review units look like:

- "Customer/Vendor Profile Strategy"
- "Legacy Mining Pipeline"
- "Auto-Fill Safety Gates"
- "Data Model Commitments"
- "Rule Candidate Review Path"
- "Rollout Sequence"
- "Open Business Decisions"

Poor review units look like:

- "Section 3.2"
- "File A"
- "Paragraph 5"
- A direct copy of every markdown heading without synthesis.

## Required Behavior

- Preserve the source material as the authority.
- Explain enough context for the user to ingest the material without reading the
  whole source first.
- Highlight decisions, assumptions, risks, non-goals, review prompts, and likely
  change points.
- Link to source files when local paths are known.
- Save annotations in browser `localStorage`.
- Export only sections with a status or note.
- Do not require Python, a dev server, or external CDNs.
- Do not modify the source material unless the user explicitly asks.

## Output Location

For a source document, place the HTML beside it:

```text
Documentation/Plans/example.md
Documentation/Plans/example-walkthrough.html
```

For a feature or code area without one source document, choose a clear location
under `Documentation/Reviews/`, `Documentation/BranchExplainers/`, or the
nearest existing documentation folder.

## Validation

After writing the HTML, validate the embedded JavaScript when Node is available:

```powershell
node -e "const fs=require('fs'); const html=fs.readFileSync('<target.html>','utf8'); const s=html.indexOf('<script>')+8; const e=html.indexOf('</script>', s); if(s<8||e<0) throw new Error('script missing'); new Function(html.slice(s,e)); console.log('script syntax ok')"
```

Also scan for unreplaced placeholders:

```powershell
rg -n "\\{\\{|TODO|TBD|PLACEHOLDER" <target.html>
```

If Node is unavailable, still do the placeholder scan and inspect the browser
page directly when possible.

## Output Summary

Report:

- HTML file path.
- Mode used.
- Source material reviewed.
- Whether validation passed.
- How to use the feedback export.
