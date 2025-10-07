import requests
from bs4 import BeautifulSoup
import pandas as pd
import random
from datetime import datetime, timedelta
import time
import os

base_url = "http://quotes.toscrape.com"
start_url = base_url + "/page/1/"
output_csv = r"C:\Users\Ferie\Desktop\Avis_Client_Quotes.csv"  # Change if besoin

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
}

# --------------------------
# Utilitaires
# --------------------------
def random_date_within_last_two_years():
    """Retourne une date ISO (YYYY-MM-DD) aléatoire dans les 2 dernières années."""
    today = datetime.today()
    days_back = random.randint(0, 365 * 2)
    rand_date = today - timedelta(days=days_back)
    return rand_date.strftime("%Y-%m-%d")

# --------------------------
# Scraping loop (pagination)
# --------------------------
rows = []
url = start_url
session = requests.Session()
session.headers.update(headers)

while url:
    try:
        resp = session.get(url, timeout=15)
    except Exception as e:
        print("Erreur de requête :", e)
        break

    if resp.status_code != 200:
        print(f"Erreur HTTP {resp.status_code} pour {url}")
        break

    soup = BeautifulSoup(resp.text, "html.parser")

    quote_divs = soup.find_all("div", class_="quote")
    if not quote_divs:
        print("Aucune citation trouvée sur :", url)
        break

    for q in quote_divs:
        # Texte de la citation -> traité comme "Avis"
        text_tag = q.find("span", class_="text")
        avis = text_tag.get_text(strip=True) if text_tag else ""

        # Auteur -> traité comme "Nom client"
        author_tag = q.find("small", class_="author")
        nom = author_tag.get_text(strip=True) if author_tag else "Anonyme"

        # Générer une date réaliste aléatoire (option)
        date = random_date_within_last_two_years()

        if avis:
            rows.append({
                "Nom client": nom,
                "Avis": avis,
                "Date": date
            })

    # Pagination : chercher le lien "Next"
    next_link = soup.select_one("li.next > a")
    if next_link and next_link.get("href"):
        next_href = next_link["href"]
        # construire URL absolue
        if next_href.startswith("http"):
            url = next_href
        else:
            url = base_url + next_href
        # courte pause pour politesse (même si site test)
        time.sleep(0.5)
    else:
        url = None

# --------------------------
# Sauvegarde CSV
# --------------------------
if rows:
    # Créer dossier si nécessaire
    out_dir = os.path.dirname(output_csv)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame(rows, columns=["Nom client", "Avis", "Date"])
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"✅ {len(rows)} lignes sauvegardées dans : {output_csv}")
else:
    print("⚠️ Aucun avis/citation extrait.")

