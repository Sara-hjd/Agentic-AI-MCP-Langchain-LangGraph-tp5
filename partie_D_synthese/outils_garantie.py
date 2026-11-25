"""Outil local de l'agent garantie : consultation du registre des garanties en LECTURE SEULE."""
import csv
from datetime import date
from pathlib import Path

from langchain.tools import tool

REGISTRE = Path(__file__).resolve().parent / "donnees" / "registre_garanties.csv"


def _ajouter_mois(d: date, mois: int) -> date:
    annee, m = divmod(d.month - 1 + mois, 12)
    return d.replace(year=d.year + annee, month=m + 1)


@tool
def consulter_garantie(numero_serie: str) -> dict:
    """Consulte le registre des garanties (lecture seule) à partir du numéro de série d'un appareil
    (format PX200-XXXX) et indique si l'appareil est encore sous garantie.
    Args:
        numero_serie: numéro de série de l'appareil, ex. "PX200-0042"
    """
    numero = numero_serie.strip().upper()
    with open(REGISTRE, encoding="utf-8", newline="") as f:  # mode "r" : aucune écriture possible
        for ligne in csv.DictReader(f):
            if ligne["numero_serie"] == numero:
                achat = date.fromisoformat(ligne["date_achat"])
                fin = _ajouter_mois(achat, int(ligne["duree_garantie_mois"]))
                aujourd_hui = date.today()
                return {
                    "numero_serie": numero, "modele": ligne["modele"],
                    "date_achat": achat.isoformat(), "duree_garantie_mois": int(ligne["duree_garantie_mois"]),
                    "fin_garantie": fin.isoformat(), "date_consultation": aujourd_hui.isoformat(),
                    "sous_garantie": aujourd_hui <= fin,
                    "jours_restants": max((fin - aujourd_hui).days, 0),
                }
    return {"numero_serie": numero, "erreur": "numéro de série introuvable dans le registre"}
