# Architecture illustration

`heatadapter_architecture.tex` extracts the architecture panel from the
manuscript's TikZ method figure. Its panel letter is removed and the canvas is
tightly bounded around the architecture. Scientific labels, equations, colors
and connections are preserved, including the manuscript's original spectral
cutoff notation. The README explains that the released code defaults to full
spectra.

The READMEs embed `heatadapter_architecture.svg`; a high-resolution PNG is also
included. To regenerate the graphics, compile the standalone TeX source with
Tectonic or pdfLaTeX, then export the resulting PDF with PyMuPDF:

```python
import fitz

with fitz.open("heatadapter_architecture.pdf") as document:
    page = document[0]
    svg = page.get_svg_image(text_as_path=True)
    svg = svg.replace(
        ">", '>\n<rect width="100%" height="100%" fill="#ffffff"/>', 1
    )
    with open("heatadapter_architecture.svg", "w") as handle:
        handle.write(svg)
    page.get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False).save(
        "heatadapter_architecture.png"
    )
```

These graphics tools are optional and are not needed to install or train the model.
