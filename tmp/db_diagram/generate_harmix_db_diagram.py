from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output" / "pdf"
OUTPUT.mkdir(parents=True, exist_ok=True)
PDF_PATH = OUTPUT / "harmix_diagrama_banco_de_dados.pdf"

pdfmetrics.registerFont(TTFont("Arial", "C:/Windows/Fonts/arial.ttf"))
pdfmetrics.registerFont(TTFont("Arial-Bold", "C:/Windows/Fonts/arialbd.ttf"))

PAGE_W, PAGE_H = landscape(A4)


def field_row(pdf, x, y, label, detail, kind=None):
    if kind == "pk":
        color = colors.HexColor("#0f766e")
        tag = "PK"
    elif kind == "fk":
        color = colors.HexColor("#a16207")
        tag = "FK"
    else:
        color = colors.HexColor("#64748b")
        tag = ""
    if tag:
        pdf.setFillColor(color)
        pdf.roundRect(x + 8, y - 9, 18, 11, 2, fill=1, stroke=0)
        pdf.setFillColor(colors.white)
        pdf.setFont("Arial-Bold", 6.5)
        pdf.drawCentredString(x + 17, y - 5.8, tag)
        label_x = x + 31
    else:
        label_x = x + 10
    pdf.setFillColor(colors.HexColor("#172033"))
    pdf.setFont("Arial-Bold", 7.5)
    pdf.drawString(label_x, y - 6, label)
    pdf.setFont("Arial", 7.2)
    pdf.setFillColor(colors.HexColor("#526173"))
    pdf.drawRightString(x + 202, y - 6, detail)


def entity(pdf, x, top, name, fields, note=None):
    width = 212
    header_h = 25
    line_h = 17
    note_h = 20 if note else 0
    height = header_h + len(fields) * line_h + note_h + 8
    bottom = top - height
    pdf.setStrokeColor(colors.HexColor("#b9c3d0"))
    pdf.setLineWidth(0.9)
    pdf.setFillColor(colors.white)
    pdf.roundRect(x, bottom, width, height, 5, fill=1, stroke=1)
    pdf.setFillColor(colors.HexColor("#1f3b5b"))
    pdf.roundRect(x, top - header_h, width, header_h, 5, fill=1, stroke=0)
    pdf.rect(x, top - header_h, width, 6, fill=1, stroke=0)
    pdf.setFillColor(colors.white)
    pdf.setFont("Arial-Bold", 10)
    pdf.drawString(x + 10, top - 16, name)
    y = top - header_h - 8
    for index, (label, detail, kind) in enumerate(fields):
        if index:
            pdf.setStrokeColor(colors.HexColor("#e5eaf0"))
            pdf.line(x + 7, y + 3, x + width - 7, y + 3)
        field_row(pdf, x, y, label, detail, kind)
        y -= line_h
    if note:
        pdf.setFillColor(colors.HexColor("#f4f7fa"))
        pdf.rect(x + 1, bottom + 1, width - 2, note_h + 3, fill=1, stroke=0)
        pdf.setFillColor(colors.HexColor("#4b5563"))
        pdf.setFont("Arial", 6.8)
        pdf.drawString(x + 9, bottom + 11, note)
    return {"x": x, "top": top, "bottom": bottom, "width": width, "height": height}


def relationship(pdf, start, end, label, cardinality_left, cardinality_right):
    x1, y1 = start
    x2, y2 = end
    pdf.setStrokeColor(colors.HexColor("#485b70"))
    pdf.setLineWidth(1.2)
    pdf.line(x1, y1, x2, y2)
    pdf.setFillColor(colors.HexColor("#1f3b5b"))
    pdf.setFont("Arial-Bold", 8)
    pdf.drawCentredString(x1 + 9, y1 + 4, cardinality_left)
    pdf.drawCentredString(x2 - 9, y2 + 4, cardinality_right)
    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
    pdf.setFillColor(colors.white)
    pdf.roundRect(mid_x - 44, mid_y - 6, 88, 13, 3, fill=1, stroke=0)
    pdf.setFillColor(colors.HexColor("#34465a"))
    pdf.setFont("Arial", 7)
    pdf.drawCentredString(mid_x, mid_y - 1.5, label)


pdf = canvas.Canvas(str(PDF_PATH), pagesize=landscape(A4))
pdf.setTitle("Diagrama do Banco de Dados da Harmix")

pdf.setFillColor(colors.HexColor("#172033"))
pdf.setFont("Arial-Bold", 18)
pdf.drawString(42, 552, "Diagrama do Banco de Dados - Harmix")
pdf.setFillColor(colors.HexColor("#526173"))
pdf.setFont("Arial", 9)
pdf.drawString(42, 535, "Modelos persistidos da aplicação: accounts.User e modelos do app music.")
pdf.drawString(42, 522, "As tabelas internas do Django (auth, sessions, contenttypes e migrations) não estão detalhadas.")

user = entity(pdf, 42, 480, "accounts_user", [
    ("id", "BigAutoField", "pk"),
    ("email", "EmailField UNIQUE", None),
    ("username", "CharField(150)", None),
    ("first_name / last_name", "CharField(150)", None),
    ("age", "PositiveSmallInteger NULL", None),
    ("gender", "CharField(20)", None),
], "Também herda password, is_active, is_staff, date_joined e outros campos do AbstractUser.")

project = entity(pdf, 325, 480, "music_musicproject", [
    ("id", "BigAutoField", "pk"),
    ("user_id", "→ accounts_user.id", "fk"),
    ("title / artist", "CharField(255)", None),
    ("youtube_url", "URLField", None),
    ("audio_file", "FileField", None),
    ("key / bpm", "CharField / PositiveInteger", None),
    ("status / processing_step", "CharField", None),
    ("task_id / error_message", "CharField / TextField", None),
    ("created_at / updated_at", "DateTimeField", None),
    ("completed_at", "DateTimeField NULL", None),
])

track = entity(pdf, 607, 480, "music_track", [
    ("id", "BigAutoField", "pk"),
    ("music_project_id", "→ music_musicproject.id", "fk"),
    ("instrument_name", "CharField(100)", None),
    ("audio_file", "FileField", None),
])

chord = entity(pdf, 607, 248, "music_chordsegment", [
    ("id", "BigAutoField", "pk"),
    ("track_id", "→ music_track.id", "fk"),
    ("start_time", "FloatField", None),
    ("end_time", "FloatField", None),
    ("chord_name", "CharField(20)", None),
], "Ordenação padrão: start_time.")

score = entity(pdf, 325, 224, "music_score", [
    ("id", "BigAutoField", "pk"),
    ("music_project_id", "→ music_musicproject.id UNIQUE", "fk"),
    ("data", "JSONField", None),
    ("manual_chord_edits", "JSONField", None),
    ("musicxml_file", "FileField", None),
    ("pdf_file", "FileField", None),
], "music_project_id é OneToOneField: uma cifra por projeto.")

relationship(pdf, (user["x"] + user["width"], 365), (project["x"], 365), "possui projetos", "1", "N")
relationship(pdf, (project["x"] + project["width"], 350), (track["x"], 350), "gera faixas", "1", "N")
relationship(pdf, (track["x"] + track["width"] / 2, track["bottom"]), (chord["x"] + chord["width"] / 2, chord["top"]), "contém segmentos", "1", "N")
relationship(pdf, (project["x"] + project["width"] / 2, project["bottom"]), (score["x"] + score["width"] / 2, score["top"]), "possui cifra", "1", "1")

pdf.setFillColor(colors.HexColor("#f4f7fa"))
pdf.roundRect(42, 12, 748, 40, 5, fill=1, stroke=0)
pdf.setFillColor(colors.HexColor("#172033"))
pdf.setFont("Arial-Bold", 8)
pdf.drawString(54, 37, "Legenda")
pdf.setFont("Arial", 8)
pdf.drawString(54, 22, "PK = chave primária   |   FK = chave estrangeira   |   1:N = um para muitos   |   1:1 = um para um")
pdf.save()

print(PDF_PATH)
