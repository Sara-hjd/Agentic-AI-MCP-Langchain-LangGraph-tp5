"""Génère les données fictives du service après-vente (SAV) :

- manuels/figures/*.png       : pictogrammes de l'écran de l'imprimante PX-200
- manuels/manuel_PX-200.pdf   : manuel d'utilisation et de dépannage (texte + figures)
- images/img_px200_01.png     : photo de l'écran de l'imprimante envoyée par un client
- registre_garanties.csv      : registre des garanties (consulté en lecture seule)

Lancement :  uv run partie_D_synthese/donnees/generer_donnees.py
"""
import csv
import math
import random
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ICI = Path(__file__).resolve().parent
FIGURES = ICI / "manuels" / "figures"
ORANGE, NOIR, BLANC = (230, 120, 20), (30, 30, 30), (255, 255, 255)


def police(taille):
    return ImageFont.load_default(size=taille)


# ----------------------------------------------------------------------------- pictogrammes
def picto_bourrage_papier(d: ImageDraw.ImageDraw):
    """Feuille froissée (bord inférieur en zigzag) + point d'exclamation."""
    feuille = [(80, 50), (220, 50), (220, 200)]
    feuille += [(220 - i * 20, 200 + (15 if i % 2 else 0)) for i in range(8)]
    feuille += [(80, 50)]
    d.polygon(feuille, outline=NOIR, width=8)
    d.rectangle([140, 85, 160, 150], fill=ORANGE)
    d.ellipse([138, 163, 162, 187], fill=ORANGE)
    d.line([(40, 250), (260, 250)], fill=NOIR, width=8)  # bac papier


def picto_encre_vide(d: ImageDraw.ImageDraw):
    """Goutte d'encre vide, barrée."""
    d.polygon([(150, 40), (90, 160), (210, 160)], outline=NOIR, width=8)
    d.arc([88, 110, 212, 240], start=0, end=180, fill=NOIR, width=8)
    d.line([(60, 260), (240, 40)], fill=ORANGE, width=14)


def picto_wifi(d: ImageDraw.ImageDraw):
    """Arcs Wi-Fi + croix : connexion réseau perdue."""
    for r in (40, 80, 120):
        d.arc([150 - r, 200 - r, 150 + r, 200 + r], start=225, end=315, fill=NOIR, width=10)
    d.ellipse([138, 188, 162, 212], fill=NOIR)
    d.line([(190, 190), (260, 260)], fill=ORANGE, width=14)
    d.line([(260, 190), (190, 260)], fill=ORANGE, width=14)


def picto_capot_ouvert(d: ImageDraw.ImageDraw):
    """Imprimante avec capot relevé et flèche vers le bas."""
    d.rectangle([60, 150, 240, 250], outline=NOIR, width=8)
    d.line([(60, 150), (110, 60)], fill=NOIR, width=8)
    d.line([(110, 60), (260, 80)], fill=NOIR, width=8)
    d.polygon([(200, 95), (240, 95), (220, 135)], fill=ORANGE)


PICTOGRAMMES = [
    ("fig1_bourrage_papier", "Bourrage papier", picto_bourrage_papier),
    ("fig2_encre_vide", "Cartouche d'encre vide", picto_encre_vide),
    ("fig3_wifi_perdu", "Connexion Wi-Fi perdue", picto_wifi),
    ("fig4_capot_ouvert", "Capot ouvert", picto_capot_ouvert),
]


def dessiner_pictogramme(fonction, taille=300, fond=BLANC):
    img = Image.new("RGB", (taille, taille), fond)
    fonction(ImageDraw.Draw(img))
    return img


# ----------------------------------------------------------------------------- photo client
def photo_ecran_client(chemin: Path):
    """Simule une photo (smartphone) de l'écran de l'imprimante : écran LCD verdâtre affichant le
    pictogramme « bourrage papier », code E-02 et voyant orange allumé ; légère rotation + bruit."""
    random.seed(42)
    photo = Image.new("RGB", (640, 480), (70, 70, 75))  # boîtier gris
    d = ImageDraw.Draw(photo)
    d.rounded_rectangle([120, 60, 520, 380], radius=18, fill=(185, 205, 170), outline=(20, 20, 20), width=6)
    picto = dessiner_pictogramme(picto_bourrage_papier, fond=(185, 205, 170)).resize((240, 240))
    photo.paste(picto, (140, 100))
    d.text((395, 190), "E-02", fill=NOIR, font=police(46))
    d.text((150, 395), "PX-200", fill=(210, 210, 210), font=police(28))
    for r, alpha in ((26, 60), (18, 130), (11, 255)):  # voyant orange (halo)
        d.ellipse([560 - r, 220 - r, 560 + r, 220 + r], fill=(255, 150 + (255 - alpha) // 4, 40))
    photo = photo.rotate(4, resample=Image.BICUBIC, fillcolor=(40, 40, 45)).filter(ImageFilter.GaussianBlur(1.2))
    pixels = photo.load()
    for _ in range(25000):  # bruit de capteur
        x, y = random.randrange(640), random.randrange(480)
        p = pixels[x, y]
        bruit = random.randint(-25, 25)
        pixels[x, y] = tuple(max(0, min(255, c + bruit)) for c in p)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    photo.save(chemin)


# ----------------------------------------------------------------------------- manuel PDF
TEXTE_VOYANTS = [
    ("Vert fixe", "L'imprimante est prête."),
    ("Vert clignotant", "Impression ou préchauffage en cours."),
    ("Orange clignotant", "Une intervention de l'utilisateur est nécessaire : un pictogramme indique le "
                          "problème sur l'écran (voir section 3)."),
    ("Orange fixe", "Erreur matérielle : éteindre l'imprimante 30 secondes ; si l'erreur persiste, "
                    "contacter le service après-vente."),
]

TEXTE_PICTOS = {
    "fig1_bourrage_papier": "Figure 1 - Pictogramme « bourrage papier » : une feuille froissée avec un point "
        "d'exclamation orange au-dessus du bac. Il s'affiche avec le code E-02 et le voyant orange clignotant "
        "lorsqu'une feuille est coincée dans le chemin du papier. Suivre la procédure de dégagement de la "
        "section 4.",
    "fig2_encre_vide": "Figure 2 - Pictogramme « cartouche d'encre vide » : une goutte barrée. Code E-05. "
        "Remplacer la cartouche (référence INK-200) ; l'impression reprend automatiquement.",
    "fig3_wifi_perdu": "Figure 3 - Pictogramme « connexion Wi-Fi perdue » : des arcs Wi-Fi avec une croix. "
        "Code E-07. Vérifier le routeur puis relancer l'assistant réseau depuis le menu Paramètres > Réseau.",
    "fig4_capot_ouvert": "Figure 4 - Pictogramme « capot ouvert » : l'imprimante avec le capot relevé. Code E-01. "
        "Refermer le capot supérieur jusqu'au déclic.",
}

TEXTE_BOURRAGE = """4. Procédure de dégagement d'un bourrage papier (code E-02)
1) Éteindre l'imprimante et débrancher le câble d'alimentation.
2) Ouvrir le capot supérieur et retirer doucement la feuille coincée en la tirant dans le sens du papier.
3) Retirer le bac papier, vérifier qu'aucun morceau de papier ne reste à l'intérieur.
4) Aérer la pile de papier, ne pas dépasser le repère de remplissage (150 feuilles de 80 g/m²).
5) Remettre le bac, refermer le capot, rebrancher et rallumer : le voyant redevient vert fixe.
Si le pictogramme réapparaît après trois tentatives, contacter le service après-vente."""

TEXTE_GARANTIE = """5. Garantie
L'imprimante PX-200 est garantie 24 mois à compter de la date d'achat, pièces et main-d'oeuvre.
Le numéro de série (format PX200-XXXX) figure sous l'appareil et sur la facture ; il permet au service
après-vente de vérifier la garantie. Sont exclus : les consommables (cartouches), les dommages dus à une
chute ou à l'utilisation de papier non conforme."""


def generer_manuel(chemin: Path):
    doc = fitz.open()
    marge, largeur = 50, 595 - 100

    def page_texte(titre, corps):
        page = doc.new_page()
        page.insert_text((marge, 70), titre, fontsize=16, fontname="hebo")
        page.insert_textbox(fitz.Rect(marge, 90, marge + largeur, 800), corps, fontsize=11, fontname="helv")
        return page

    # page 1 : présentation
    page_texte("Imprimante PX-200 - Manuel d'utilisation et de dépannage",
               "1. Présentation\nLa PX-200 est une imprimante jet d'encre couleur Wi-Fi. Son panneau de "
               "commande comporte un écran qui affiche des pictogrammes et un voyant d'état bicolore "
               "(vert / orange).\n\nSommaire : 2. Voyants du panneau de commande - 3. Pictogrammes de "
               "l'écran - 4. Dégagement d'un bourrage papier - 5. Garantie.")
    # page 2 : voyants
    page_texte("2. Voyants du panneau de commande",
               "\n\n".join(f"Voyant {etat} : {sens}" for etat, sens in TEXTE_VOYANTS))
    # pages 3-4 : pictogrammes (2 figures par page)
    for i in range(0, len(PICTOGRAMMES), 2):
        page = doc.new_page()
        page.insert_text((marge, 70), "3. Pictogrammes de l'écran", fontsize=16, fontname="hebo")
        for j, (nom, _, _) in enumerate(PICTOGRAMMES[i:i + 2]):
            y = 100 + j * 360
            page.insert_image(fitz.Rect(marge, y, marge + 180, y + 180), filename=str(FIGURES / f"{nom}.png"))
            page.insert_textbox(fitz.Rect(marge, y + 190, marge + largeur, y + 330), TEXTE_PICTOS[nom],
                                fontsize=11, fontname="helv")
    # page 5 : procédure ; page 6 : garantie
    titre, corps = TEXTE_BOURRAGE.split("\n", 1)
    page_texte(titre, corps)
    titre, corps = TEXTE_GARANTIE.split("\n", 1)
    page_texte(titre, corps)
    chemin.parent.mkdir(parents=True, exist_ok=True)
    doc.save(chemin)


# ----------------------------------------------------------------------------- registre
REGISTRE = [
    ("numero_serie", "modele", "date_achat", "duree_garantie_mois", "client"),
    ("PX200-0042", "PX-200", "2025-06-12", "24", "C-1187"),
    ("PX200-0017", "PX-200", "2023-02-03", "24", "C-0954"),
    ("PX200-0105", "PX-200", "2026-09-28", "24", "C-1420"),
    ("PX300-0008", "PX-300", "2024-11-30", "36", "C-1011"),
]

if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    for nom, _, fonction in PICTOGRAMMES:
        dessiner_pictogramme(fonction).save(FIGURES / f"{nom}.png")
    generer_manuel(ICI / "manuels" / "manuel_PX-200.pdf")
    photo_ecran_client(ICI / "images" / "img_px200_01.png")
    with open(ICI / "registre_garanties.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(REGISTRE)
    print("Données générées dans", ICI)
