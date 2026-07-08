# dfnaiff.github.io

Personal site — a plain static page. No Jekyll, no CMS, no build step.

```
index.html      home / landing page
writing.html    list of all technical notes
note.html       renders a single note (markdown + math + code)
style.css       all styling (one file, two themes, EN/PT)
app.js          language + theme toggles
notes/          the technical notebook
  index.json      metadata for every note (the index)
  *.md            note bodies, in markdown
.nojekyll       tells GitHub Pages to serve files as-is
```

## Run it locally

Any static server works (needed because the notes are loaded with `fetch`):

```bash
cd site
python3 -m http.server 8000
# open http://localhost:8000
```

## Add a technical note

1. Create `notes/my-slug.md` with the post body (plain markdown).
   - Math: `$inline$` and `$$display$$` (KaTeX).
   - Code: fenced ```` ```lang ```` blocks.
2. Add one entry to the **top** of the `notes` array in `notes/index.json`:

   ```json
   { "slug": "my-slug", "title": "My title", "date": "2026", "summary": "One line." }
   ```

3. Optionally add it to the curated list on the home page (`index.html`, the
   `#notes-list` block). Otherwise it already appears on `writing.html`.

The post is live at `note.html?slug=my-slug`. No rebuild.

> Or just hand the markdown to Claude Code and say "publish this as a note" — steps 1–3
> are exactly the kind of thing it can do for you.

## Deploy to dfnaiff.github.io

The repo `dfnaiff/dfnaiff.github.io` serves its root (or `/docs`) as the site.

- Copy the **contents** of this `site/` folder to the repo root (or to `docs/` and set
  Pages → Source → `/docs`).
- Push. GitHub Pages publishes at `https://dfnaiff.github.io`.

### Custom domain (later, dfnaiff.com)

Add a file named `CNAME` containing `dfnaiff.com`, point the domain's DNS at GitHub Pages,
and enable the domain under repo Settings → Pages.

## Dependencies

Only `note.html` loads anything external, from the jsDelivr CDN, and only when you open a
note: [marked](https://github.com/markedjs/marked) (markdown) and
[KaTeX](https://katex.org) (math). The home page has zero dependencies. To go fully
self-contained, vendor those files into a `vendor/` folder and update the `<script>`/`<link>`
tags in `note.html`.
