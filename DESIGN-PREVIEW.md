# Personal homepage design preview

Local branch: `design/personal-homepage`. The homepage uses the existing Jekyll repository and original assets, publication metadata, and product destinations.

Run from the repository root:

```sh
python3 tools/preview.py --port 4173
```

Open http://localhost:4173. If running on a remote machine, forward port 4173 to your local computer first. The preview server binds to loopback by default.

The preview server renders the homepage without Ruby. It serves the existing standalone product pages and assets unchanged; it is not a full Jekyll replacement. For GitHub Pages, `index.md` uses `_layouts/personal.html` normally.

Changed files:

- `index.md`: homepage content, research, products, experience, milestones and contact.
- `_layouts/personal.html`: independent homepage document shell.
- `assets/css/personal.css`: responsive visual design; optional Google Fonts with local system fallbacks.
- `assets/js/personal.js`: WeChat copy button with a visible fallback.
- `tools/preview.py`: local preview server.

The product illustrations are decorative depictions, not live metrics or actual interface screenshots. Publication titles, authors, venues and links come from the original `publications.md`. No personal contribution claims were added.

## Revision 2

White background, blue accents, system sans-serif type. Research is the default homepage. Investment Perspective is a separate page at `/investment/`, with its own active navigation tab, experience and areas of interest. The investment page uses the original internship information; the areas of interest are editorial drafts based on the existing biography, not investment case studies or published articles.

The first design is saved under `design-archive/v1/` for reference. The local preview now renders both pages.

## Revision 3

Rebuilt after inspecting the rendered purshow.github.io and kolakivy.github.io references: readable 17–18 px body copy, 64 px medium-weight name, 34 px section titles, 1008 px content width, white and light grey section rhythm, quiet blue links, and plain publication links. Research and Investment Perspective remain independent pages. Figtree is served locally with its OFL license. Reference source code and personal assets were not copied.

## Revision 4

Monochrome interface: uniform white background, neutral black/grey typography, no gradient or coloured section backgrounds. Social links are plain icon/text links, venue labels have no filled badge, and portrait shadows are removed. Inline links are underlined. Original portrait and publication imagery retain their colours.

## Research typography adjustment

Research-only styles increase introductory text from 17 to 19 px, publication titles from 19 to 21 px, and authors from 15 to 16 px. Adjacent section padding decreases from 113 px combined to 58 px. Mobile reading sizes increase as well. Investment page rendering is unchanged.

## Three sections and classified writing

Navigation is now Research / Investing / Fragments. `_data/writing.yml` records all 54 previously indexed articles and notes: 6 Research, 12 Investing, 36 Fragments. The duplicated product essay is listed once using its descriptive title, while both original URLs remain accessible.

- Research: technical notes, in addition to the existing research page.
- Investing: internship experience and selected writing, with more writing in a disclosure.
- Fragments: SciAgent, TimeTodo, HIVE GROUP, IdeaMesh, WeShare, nine competition awards, scholarship information and personal essays.
- `/blogs/` remains a complete categorized writing archive.
- Article bodies and permalinks are preserved. Categorized articles use `_layouts/note.html` and the shared navigation, with a link back to their section.

The local preview also renders Markdown article pages. Its dependencies are listed in `tools/requirements-preview.txt`; install with `python3 -m pip install -r tools/requirements-preview.txt` if unavailable. The current host already has these installed. GitHub Pages continues using Jekyll and does not require these Python packages.
