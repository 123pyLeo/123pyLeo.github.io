# Homepage deployment and rollback

The existing GitHub Pages site and `lintao.online` custom domain are retained. The redesigned pages replace the current presentation. Blog permalinks, original article bodies, products and assets remain in the repository.

## Original version

The complete original source is preserved on GitHub in `backup/pre-redesign-2026-10-22`, at commit `995a677572d75ceb4d444808bcf43c97ac85d3b8`. No force push or history rewrite is needed to publish the redesign.

## Local build

```sh
bundle install
bundle exec jekyll build
```

Python preview instructions are in `DESIGN-PREVIEW.md`; the Python preview is not the production build. Draft designs and development tools are excluded from the generated site.

## Rollback

Restore the original tree as a new commit on the publishing branch, or revert the redesign merge commit. Keep the backup branch. This returns the original site without rewriting the branch history. Verify the GitHub Pages deployment and open `https://lintao.online/` after rolling back.

## Analytics

Umami website ID and public dashboard link are in `assets/js/analytics-config.js`. Tracking runs only on `lintao.online` and `www.lintao.online`; localhost previews are excluded. The configuration contains no password or private API key.
