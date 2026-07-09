# Annotated Review HTML Design

Use this design for every generated review companion unless the user explicitly
asks for a different style.

## Purpose

The page is a reading and feedback surface, not a landing page and not a
markdown rendering. It should make dense material easier to ingest and make
feedback easy to return to the agent.

The agent authors the walkthrough after understanding the source material. The
source is evidence; it is not the page outline.

## Layout

- Static, single-file HTML with embedded CSS and JavaScript.
- Left sticky sidebar on desktop with title, section nav, and global actions.
- Single-column top sidebar on narrow screens.
- Main content uses compact white sections on a light gray page background.
- Each synthesized review unit gets one review section with:
  - heading and one-line purpose,
  - generated summary/details,
  - optional review prompts,
  - annotation panel.
- Review units should be chosen for reader comprehension and feedback quality,
  not because a matching markdown heading exists.

## Visual System

- Use a restrained operational-tool style, not a marketing page.
- Palette:
  - page background `#f6f7f9`
  - panels `#ffffff`
  - sidebar `#18232b`
  - text `#1d2730`
  - muted text `#63717d`
  - lines `#cfd8df`
  - primary blue `#1e6b96`
  - success green `#35754b`
  - warning gold `#8a651d`
  - risk red `#a13f3f`
- Border radius stays at `8px` or less.
- Avoid decorative gradients, blobs, bokeh, oversized heroes, and nested cards.
- Use stable dimensions and responsive grids so notes and buttons do not shift
  awkwardly.

## Interaction

- Notes save automatically to `localStorage`.
- Storage key should include a slug derived from the source path.
- Each annotation has:
  - status dropdown,
  - freeform note textarea,
  - clear-this-note button,
  - saved timestamp.
- Global actions:
  - copy feedback markdown,
  - preview markdown,
  - download markdown,
  - clear all notes,
  - open source markdown.
- Clipboard failures must fall back to selecting the generated markdown textarea.

## Feedback Export

Export markdown should be directly pasteable into chat.

Required structure:

```markdown
# Feedback on <document title>

- Source: `<source path>`
- Walkthrough: `<html path>`
- Mode: `<mode>`
- Generated: <local timestamp>

## Section Feedback

### <review unit title>

- Status: <status label>

<note text>
```

Only include sections with a status or note.

## Review Prompts

Generate short prompts from the synthesized review unit. Prefer practical
questions:

- "Is this permission gate correct?"
- "Is this non-goal still true?"
- "Does this phase have a usable Done-when?"
- "Would users trust this automation?"
- "What evidence would make this safe?"

Do not generate more than four prompts for one section.

## Source Handling

For plans and specs, read the full document and then build the review structure
around decisions, workflows, data model commitments, risks, rollout, and open
questions.

For app features or code areas, inspect enough real source material to explain
the feature accurately. Good source material may include routes, UI components,
services, models, tests, screenshots, browser state, and existing docs.

For diffs or branches, inspect the actual diff and representative changed files.
Build the review around behavioral surfaces and decisions, not file order.

## Accessibility

- Use semantic headings.
- Buttons must be real `button` elements.
- Add `aria-live` to export status messages.
- Labels must point at form controls.
- Keep color contrast readable.

## What Not To Do

- Do not require a dev server.
- Do not require Python or any generator script.
- Do not mechanically split markdown headings into sections.
- Do not call external CDNs.
- Do not include external fonts.
- Do not create a web-app route unless explicitly requested.
- Do not overwrite hand-authored walkthrough pages without explicit approval.
