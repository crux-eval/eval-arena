from pathlib import Path

def write_sections_index(out_dir: Path):
    """
    Generate ${OUTPATH}/sections/index.html listing all section files across benchmarks.
    """
    sections_index_dir = out_dir / "sections"
    sections_index_dir.mkdir(parents=True, exist_ok=True)

    html = ['<!DOCTYPE html><html><head>',
            '<meta charset="utf-8">',
            '<title>Sections Index</title>',
            '<link rel="stylesheet" href="../static/css/bulma.min.css">',
            '<link rel="stylesheet" href="../static/css/custom.css">',
            '</head><body>',
            '<section class="section"><div class="container">',
            '<h1 class="title">All Sections</h1>']

    # Find all benchmark directories with sections
    for benchmark_dir in sorted(out_dir.iterdir()):
        sections_path = benchmark_dir / "sections"
        if sections_path.is_dir():
            html.append(f'<h2 class="subtitle" style="margin-top: 1.5rem;">{benchmark_dir.name}</h2><ul>')
            for f in sorted(sections_path.iterdir()):
                if f.is_file() and f.suffix == '.html':
                    html.append(f'<li><a href="../{benchmark_dir.name}/sections/{f.name}">{f.stem}</a></li>')
            html.append('</ul>')

    html.append('</div></section></body></html>')

    index_path = sections_index_dir / "index.html"
    with open(index_path, "w", encoding="utf-8") as f:
        f.write('\n'.join(html))
