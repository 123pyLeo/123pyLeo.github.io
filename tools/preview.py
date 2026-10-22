#!/usr/bin/env python3
"""Local preview of the personal pages and Markdown posts without Ruby.

Install preview dependencies with: python3 -m pip install -r tools/requirements-preview.txt
GitHub Pages continues to build the actual site with Jekyll.
"""
import argparse
import html
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

import yaml
from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
MARKDOWN = MarkdownIt('commonmark', {'html': True}).enable('table')


def source_page(path):
    text = path.read_text()
    if text.startswith('---'):
        _, front, body = text.split('---', 2)
        return yaml.safe_load(front) or {}, body.lstrip()
    return {}, text


def routes():
    result = {'/': ROOT / 'index.md', '/index.html': ROOT / 'index.md'}
    for source in list(ROOT.glob('*.md')) + list((ROOT / 'blogs').rglob('*.md')):
        meta, _ = source_page(source)
        permalink = meta.get('permalink')
        if permalink:
            result[permalink] = source
            if permalink.endswith('/index.html'):
                result[permalink[:-10]] = source
                result[permalink[:-11]] = source
    return result


def navigation(section):
    items = [('research', 'Research', '/'), ('investing', 'Business Reflections', '/investment/'),
             ('fragments', 'Fragments', '/fragments/')]
    links = ''.join(f'<a href="{url}"' + (' aria-current="page"' if key == section else '')
                    + f'>{label}</a>' for key, label, url in items)
    return ('<header class="topbar"><div class="wrap topbar-inner">'
            '<a class="brand" href="/">Tao Lin <span>林涛</span></a>'
            f'<nav class="main-nav" aria-label="Main navigation">{links}</nav></div></header>')


def render_page(source):
    meta, body = source_page(source)
    title = str(meta.get('title', 'Research'))
    if meta.get('layout') == 'note':
        section = meta.get('section', 'fragments')
        label, url = {'research': ('Research', '/#research'),
                      'investing': ('Business Reflections', '/investment/#writing'),
                      'fragments': ('Fragments', '/fragments/#blog')}[section]
        content = MARKDOWN.render(body)
        note_layout = (ROOT / '_layouts/note.html').read_text().split('---', 2)[2].lstrip()
        note_layout = note_layout.replace('{% include personal-nav.html %}', navigation(section))
        start = note_layout.index('{% if page.section')
        end = note_layout.index('{% endif %}', start) + len('{% endif %}')
        note_layout = note_layout[:start] + f'<a href="{url}">← {label}</a>' + note_layout[end:]
        body = note_layout.replace('{{ content }}', content)
    elif meta.get('layout') != 'personal':
        body = (navigation('fragments') + '<main id="main" class="note-page wrap">'
                f'<h1>{html.escape(title)}</h1><div class="note-body">'
                + MARKDOWN.render(body) + '</div></main>')
    body = body.replace('{{ site.url }}', '').replace('{{ page.title }}', html.escape(title))
    layout = (ROOT / '_layouts/personal.html').read_text()
    return layout.replace('{{ content }}', body).replace('{{ page.title }}', html.escape(title)).encode()


class Preview(SimpleHTTPRequestHandler):
    def do_GET(self):
        path = unquote(urlsplit(self.path).path)
        if any(part.startswith('.') for part in Path(path).parts):
            self.send_error(404)
            return
        source = routes().get(path)
        if source:
            payload = render_page(source)
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(payload)
            return
        super().do_GET()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=4173)
    parser.add_argument('--host', default='127.0.0.1')
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), partial(Preview, directory=str(ROOT)))
    print(f'Homepage preview: http://{args.host}:{args.port}', flush=True)
    server.serve_forever()
