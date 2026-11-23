---
name: EuroSAT land-cover classifier
description: A local Random Forest classifier for Sentinel-2 tiles, read like a survey map sheet.
colors:
  ground: "#0e1513"
  sheet: "#141d1a"
  raise: "#1a2522"
  rule: "#2b3a35"
  rule-faint: "#22302b"
  ink: "#dbe4de"
  collar-ink: "#93a69d"
  revision-purple: "#b994f5"
  revision-purple-hover: "#cbb0f8"
  purple-ink: "#120d1c"
  water-blue: "#5cb8ec"
  alert: "#ff8a7a"
  class-annual-crop: "#e2ac35"
  class-forest: "#2f9e4f"
  class-herbaceous: "#97c45a"
  class-highway: "#e5564d"
  class-industrial: "#9aa4ae"
  class-pasture: "#cfdc7a"
  class-permanent-crop: "#ec8a4b"
  class-residential: "#e98bb5"
  class-river: "#3fb8c8"
  class-sea-lake: "#4d8de6"
typography:
  display:
    fontFamily: "Archivo Narrow, Archivo, sans-serif"
    fontSize: "clamp(40px, 4.6vw, 60px)"
    fontWeight: 700
    lineHeight: 0.95
  headline:
    fontFamily: "Archivo Narrow, Archivo, sans-serif"
    fontSize: "26px"
    fontWeight: 700
    lineHeight: 1.15
  title:
    fontFamily: "Archivo, system-ui, sans-serif"
    fontSize: "15px"
    fontWeight: 600
    lineHeight: 1.3
  body:
    fontFamily: "Archivo, system-ui, sans-serif"
    fontSize: "15px"
    fontWeight: 400
    lineHeight: 1.5
    fontFeature: "tnum"
  small:
    fontFamily: "Archivo, system-ui, sans-serif"
    fontSize: "13px"
    fontWeight: 400
    lineHeight: 1.5
    fontFeature: "tnum"
  label:
    fontFamily: "Archivo, sans-serif"
    fontSize: "13px"
    fontWeight: 600
    lineHeight: 1.3
    letterSpacing: "0.09em"
  wordmark:
    fontFamily: "Archivo Narrow, Archivo, sans-serif"
    fontSize: "15px"
    fontWeight: 700
    lineHeight: 1
    letterSpacing: "0.06em"
rounded:
  hair: "1px"
  chip: "2px"
  tag: "3px"
  control: "4px"
spacing:
  gutter-mobile: "16px"
  block: "24px"
  sheet-top: "36px"
  gutter: "40px"
  column-gap: "48px"
components:
  button-primary:
    backgroundColor: "{colors.revision-purple}"
    textColor: "{colors.purple-ink}"
    typography: "{typography.title}"
    rounded: "{rounded.control}"
    padding: "12px 20px"
  button-primary-hover:
    backgroundColor: "{colors.revision-purple-hover}"
  button-ghost:
    backgroundColor: "transparent"
    textColor: "{colors.ink}"
    rounded: "{rounded.control}"
    padding: "12px 20px"
  button-ghost-hover:
    backgroundColor: "{colors.raise}"
  button-small:
    padding: "6px 12px"
    typography: "{typography.small}"
  input:
    backgroundColor: "{colors.sheet}"
    textColor: "{colors.ink}"
    rounded: "{rounded.control}"
    padding: "7px 10px"
  option-chip:
    backgroundColor: "transparent"
    textColor: "{colors.ink}"
    rounded: "{rounded.control}"
    padding: "9px 14px"
  state-chip-active:
    backgroundColor: "{colors.revision-purple}"
    textColor: "{colors.purple-ink}"
    typography: "{typography.label}"
    rounded: "{rounded.chip}"
    padding: "3px 8px"
  block-tooltip:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.ground}"
    typography: "{typography.small}"
    rounded: "{rounded.tag}"
    padding: "6px 9px"
---

# Design System: EuroSAT land-cover classifier

## Overview

**Creative North Star: "The Survey Sheet Collar"**

Every classification is read like a printed survey map sheet. The tile sits inside a collar of true marginalia: corner coordinates taken from the GeoTIFF, graticule ticks every 8 px, a north arrow, an alternating scale bar and an imprint line naming the model. The model's answer is overprinted in revision purple, the colour a map office uses for revisions laid over a printed sheet. The window class map lies on the tile as graticule cells in land-cover tints.

The ground is deep map-ink green-black, dark only. Surfaces are separated by hairline rules rather than cards or shadows; the page is dense, columnar and tabular, closer to a map legend or a timetable than an app dashboard. Hierarchy comes from two working text sizes ranked by weight, capitals and reversal, plus one condensed display size reserved for the verdict.

Map conventions are binding where they exist: roads red, built-up pink, industry grey, water blue, and water names lettered in italic blue as hydrography is on a printed map.

**Key Characteristics:**
- Dark map-ink ground only; there is no light theme.
- Revision purple speaks for the model and for the user's next action; nothing else is purple.
- Hairline-ruled columns, no cards, no drop-shadowed panels.
- Two working sizes (13 px, 15 px), a 26 px sheet title and a condensed verdict display.
- Tabular figures everywhere; every number is a real model output.
- Real marginalia (coordinates, ticks, scale bar) frame the tile; none of it is decorative.

## Colors

A near-black green ground with pale green-grey ink, one revision-purple voice, map-convention class tints, and water blue for hydrography. The shipped values are lighter than the direction contract's draft: sheet, ink, collar ink, purple and water were raised so body and collar text keep at least 4.5:1 against the dark ground (measured: ink 14.2:1, collar ink 7.2:1, purple 7.6:1, water blue 8.4:1 on #0e1513; all above 6.7:1 on the sheet). The ground is unchanged.

### Primary
- **Revision Purple** (#b994f5): every model answer (the verdict, the probability bars, the imprint's "Overprinted by" line), the primary button, the active tab underline, checked options, running/done job states, the progress meter, focus rings, caret and text selection.
- **Revision Purple Light** (#cbb0f8): primary button hover only.
- **Purple Ink** (#120d1c): text on purple fills (buttons, active state chips, selection).

### Secondary
- **Hydrography Blue** (#5cb8ec): text colour for water classes (River, SeaLake) wherever their names appear, always with italic lettering, including a water verdict.

### Tertiary
- **Land-cover tints**: the ten class colours, used only as map cells, legend swatches and class-map overlays, never as UI chrome. Annual crop amber (#e2ac35), forest green (#2f9e4f), herbaceous green (#97c45a), highway red (#e5564d), industrial grey (#9aa4ae), pasture yellow-green (#cfdc7a), permanent crop orange (#ec8a4b), residential pink (#e98bb5), river cyan (#3fb8c8), sea/lake blue (#4d8de6).
- **Alert Coral** (#ff8a7a): errors and the failed job state only.

### Neutral
- **Map-Ink Ground** (#0e1513): page background, also the inner neatline ring around the tile and the sticky action bar.
- **Sheet** (#141d1a): field backgrounds, log panel, empty tile.
- **Raise** (#1a2522): hover fill for ghost buttons and highlighted table rows, meter track, inline code.
- **Rule** (#2b3a35): structural hairlines between columns, sections, fieldsets; control borders.
- **Faint Rule** (#22302b): row dividers inside tables and legend, the empty tile's 8 × 8 graticule.
- **Ink** (#dbe4de): primary text, the tile's outline, scale bar fills.
- **Collar Ink** (#93a69d): marginalia, labels, hints, secondary text, ticks, the outer neatline.

### Named Rules
**The Overprint Rule.** Revision purple marks only what the model says and what the user does next (answers, primary actions, active and focused state). If it is not the model's voice or the next step, it is not purple.

**The Hydrography Rule.** Water class names are always lettered in italic Hydrography Blue, in prose, tables, legends and the verdict alike.

**The Map Convention Rule.** Class tints follow printed-map convention (roads red, built-up pink, industry grey, water blue) and stay on the map and legend; they never colour buttons, headings or states.

## Typography

**Display Font:** Archivo Narrow (with Archivo, sans-serif), self-hosted variable 400-700, upright and italic
**Body Font:** Archivo (with system-ui, sans-serif), self-hosted variable 100-900, upright and italic
**Label/Mono Font:** ui-monospace / Consolas for code and the job log only

**Character:** One grotesque family in two widths: Narrow for marks (wordmark, sheet title, verdict), normal width for reading. Tabular figures site-wide make columns of numbers line up like a timetable.

### Hierarchy
- **Display** (700, clamp(40px, 4.6vw, 60px), 0.95): the verdict only, one per answer column, in revision purple or italic water blue.
- **Headline** (700, 26px, 1.15, Narrow): the sheet title (file name or page name) and the empty-state intro heading.
- **Title** (600, 15px, 1.3): approach headings and button text.
- **Body** (400, 15px, 1.5): reading text and table cells; prose capped at 62-75ch.
- **Small** (400, 13px): marginalia, hints, confidence lines, imprint, tooltips.
- **Label** (600, 13px, 0.09em, uppercase, collar ink): section headings, table heads, fieldset legends, model-select labels, job state.

### Named Rules
**The Two Sizes Rule.** Working text uses 13 px and 15 px only. Rank comes from weight, capitals and reversal (ink or purple fill), not from new sizes. The 26 px title and the verdict display are the only exceptions.

**The Verdict Rule.** The condensed display size is reserved for the model's answer. Nothing else is set that large.

## Layout

Hairline-ruled columns across the full width. Classify is a two-column grid: the sheet column (up to 690 px) and the answers column holding two equal answer columns 48 px apart, followed by a three-column lower band of model parameters, method and a full-width legend. Train & validate pairs a form column with a sticky status column (up to 520 px). The page gutter is 40 px, section tops 36 px, column gaps 48 px, block gaps 24 px.

The tile is 448 px square (shrinking with its container) with ticks at every 8 px graticule step and coordinates set in the collar. The top bar is 60 px: wordmark, page tabs, model selects pushed right.

Responsive: at 1180 px (Classify) and 1100 px (Train) the columns stack and the status column stops being sticky. At 760 px the gutter drops to 16 px, the model selects become a two-column label/field grid, the answers and lower band go to one column, right-side coordinates and the north arrow hide, and before a result the upload action moves above the empty tile. In the result state the imprint moves below the class-map toggle and the upload button.

## Elevation & Depth

Flat. Depth comes from tone (ground, sheet, raise) and hairline rules, not shadows. The tile carries a double map neatline built from rings (1 px ink outline, a 4 px ground gap, a 1 px collar-ink outer ring). Two functional shadows exist: the block tooltip floats over the tile, and the sticky Train action bar fades content beneath it.

### Shadow Vocabulary
- **Neatline** (`outline: 1px solid ink; box-shadow: 0 0 0 4px ground, 0 0 0 5px collar-ink`): the tile frame only.
- **Tooltip lift** (`box-shadow: 0 6px 18px rgba(0,0,0,.45)`): the block tooltip on the class map.
- **Sticky fade** (`box-shadow: 0 -12px 24px rgba(14,21,19,.85)`): the sticky Start bar on Train.

### Named Rules
**The Ruled Sheet Rule.** Sections are divided by 1 px rules, never boxed in shadowed cards.

## Shapes

Nearly square. Controls and option chips have 4 px corners, inline code and tooltips 3 px, swatches and state chips 2 px, bars and meters 1 px. The tile, cells and graticule are sharp. Rules are always 1 px. The empty tile shows an 8 × 8 faint graticule; class-map cells carry a half-pixel dark inset so adjacent blocks stay legible.

## Components

### Buttons
Plain and firm; the primary is a purple slab, the ghost a ruled outline.
- **Shape:** gently squared (4px)
- **Primary:** revision purple fill and border, purple-ink text, 600 15 px, 12px 20px padding.
- **Hover / Focus:** lighter purple on hover, 1 px press on active, 2 px purple focus ring offset 2 px.
- **Ghost:** transparent with a rule border and ink text; hover shows collar-ink border and raise fill.
- **Small:** 6px 12px padding, 13 px text.
- **Disabled:** 45% opacity, no press.

### Chips
- **Option chips (radio):** rule-bordered 4 px boxes, 9px 14px padding; checked takes a purple border and a 13% purple tint.
- **Job state labels:** label type, no glyph. Idle is a plain collar-ink caps label with no box. Queued is purple text in a 1 px dashed purple outline. Running and done reverse to a purple plate with purple-ink text. Failed is an alert-coral plate with ground-coloured text. Plates have 2 px corners and 3px 8px padding.

### Inputs / Fields
- **Style:** sheet background, rule border, 4 px corners, 7px 10px padding; selects use a two-triangle collar-ink chevron.
- **Focus:** 2 px purple ring; hover lifts the border to collar ink.
- **Layout:** 230 px label column, field, hint below in small collar ink.

### Navigation
Wordmark in Narrow caps with a purple slash; tabs in collar ink 500 weight that turn ink on hover, with the current page marked by ink text and a 2 px purple underline.

### Tables
Full-width, faint row rules, label-type heads, numbers right-aligned in tabular figures. Probability rows carry a 4 px purple bar; legend rows carry a 12 px class swatch.

### Disclosure
Collapsible extras (the block grid as a table) use a small-text collar-ink summary with a drawn CSS triangle (no glyph character) that turns 90° when open; hover turns the text to ink.

### Survey Sheet (signature)
The tile in its double neatline, collar coordinates (CRS on the top-left corner, then E/N), graticule ticks and a north arrow. Below the tile, a scale bar draws four round-number segments to true scale (for example 4 × 150 m = 600 m at 600/639 of the tile width), with alternating ink cells and the stops labelled beside it ("0 · 150 · 300 · 450 · 600 m"). Under the collar, the imprint gives the source (Sentinel-2, 13 bands, pixel size and resolution, without repeating the CRS) and a purple "Overprinted by <model>" line with the training timestamp. The class map lies on the tile at 58% opacity; hovering or focusing a block raises it, dims the rest, highlights its legend row and shows an ink tooltip with class, confidence and block position. Hovering a legend row highlights its blocks. Cells settle in with a staggered opacity ease (0.9 s, cubic-bezier(.16,1,.3,1)); reduced motion removes all transitions and animation.

## Do's and Don'ts

### Do:
- **Do** keep the ground dark map-ink (#0e1513); the product is dark theme only.
- **Do** set every model answer in revision purple, and water answers in italic Hydrography Blue.
- **Do** letter River and SeaLake in italic blue wherever their names appear.
- **Do** colour classes by map convention and keep tints on the map, swatches and legend.
- **Do** divide space with 1 px rules (#2b3a35 between sections, #22302b between rows).
- **Do** rank text by weight, capitals and reversal within the 13 px and 15 px sizes.
- **Do** use tabular figures and right-align numeric columns.
- **Do** frame tiles with real marginalia derived from the data (coordinates, scale, ticks).

### Don't:
- **Don't** add a light theme or light surfaces.
- **Don't** use purple for decoration, headings or non-model emphasis.
- **Don't** put sections in shadowed or filled cards; the Neatline and the tooltip are the only framed objects.
- **Don't** introduce working text sizes beyond 13 px and 15 px, or use the display size for anything but the verdict.
- **Don't** show decorative marginalia or numbers that are not real outputs of the data or model.
