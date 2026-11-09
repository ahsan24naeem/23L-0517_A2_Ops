"""Customise pandoc's reference.docx: fonts, heading colours, shaded screenshot boxes, table look."""
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor, Cm

d = Document("ref_default.docx")
ACC = RGBColor(0x2B, 0x4C, 0x7E)
INK = RGBColor(0x1D, 0x1F, 0x24)
st = {s.name: s for s in d.styles}

def font(s, name="Calibri", size=None, bold=None, color=None, italic=None):
    s.font.name = name
    rpr = s.element.get_or_add_rPr()
    rf = rpr.find(qn("w:rFonts"))
    if rf is None:
        rf = OxmlElement("w:rFonts"); rpr.append(rf)
    for a in ["w:ascii", "w:hAnsi", "w:cs", "w:eastAsia"]:
        rf.set(qn(a), name)
    for t in ["w:asciiTheme", "w:hAnsiTheme", "w:cstheme", "w:eastAsiaTheme"]:
        if rf.get(qn(t)) is not None:
            del rf.attrib[qn(t)]
    if size: s.font.size = Pt(size)
    if bold is not None: s.font.bold = bold
    if italic is not None: s.font.italic = italic
    if color is not None: s.font.color.rgb = color

def shade(s, fill, border=None):
    ppr = s.element.get_or_add_pPr()
    shd = OxmlElement("w:shd"); shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), fill)
    ppr.append(shd)
    if border:
        bdr = OxmlElement("w:pBdr")
        for side in ["top", "left", "bottom", "right"]:
            e = OxmlElement(f"w:{side}"); e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "6")
            e.set(qn("w:space"), "4"); e.set(qn("w:color"), border); bdr.append(e)
        ppr.append(bdr)

for name in ["Normal", "Body Text", "First Paragraph", "Compact"]:
    if name in st:
        font(st[name], size=10.5, color=INK)
st["Body Text"].paragraph_format.space_after = Pt(6)
font(st["Title"], size=24, bold=True, color=ACC)
font(st["Subtitle"], size=13, color=RGBColor(0x64, 0x6A, 0x73), italic=False)
for lvl, size in [(1, 17), (2, 13), (3, 11.5)]:
    s = st[f"Heading {lvl}"]
    font(s, size=size, bold=True, color=ACC if lvl < 3 else INK)
    s.paragraph_format.space_before = Pt(16 if lvl == 1 else 10)
    s.paragraph_format.space_after = Pt(6)
    s.paragraph_format.keep_with_next = True
st["Heading 1"].paragraph_format.page_break_before = True
# screenshot placeholders are blockquotes -> "Block Text"
bt = st["Block Text"]
font(bt, size=10, color=RGBColor(0x33, 0x3A, 0x45), italic=False)
shade(bt, "EEF1F6", border="9AA7BD")
bt.paragraph_format.left_indent = Cm(0.2); bt.paragraph_format.right_indent = Cm(0.2)
bt.paragraph_format.space_before = Pt(6); bt.paragraph_format.space_after = Pt(8)
# code
for name in ["Source Code", "Verbatim Char"]:
    if name in st:
        font(st[name], name="Menlo", size=8.5)
from docx.enum.style import WD_STYLE_TYPE
if "Source Code" not in st:
    sc = d.styles.add_style("Source Code", WD_STYLE_TYPE.PARAGRAPH)
    st["Source Code"] = sc
font(st["Source Code"], name="Menlo", size=8.5)
shade(st["Source Code"], "F4F4F2", border="E2E0D8")
st["Source Code"].paragraph_format.space_after = Pt(6)
font(st["Image Caption"], size=9.5, italic=True, color=RGBColor(0x46, 0x4C, 0x55))
st["Image Caption"].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
if "Captioned Figure" in st:
    st["Captioned Figure"].paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
font(st["Table Caption"], size=9.5, italic=True)
for sec in d.sections:
    sec.page_width, sec.page_height = Cm(21), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(2.0)
    sec.top_margin = sec.bottom_margin = Cm(2.0)
d.save("reference.docx")
print("reference.docx written")
