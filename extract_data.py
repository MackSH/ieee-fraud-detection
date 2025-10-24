import os
import zipfile

def extract_dataset(zip_path="ieee-fraud-detection.zip", extract_to="Data"):
    """
    Extrait le dataset s'il n'est pas déjà extrait.

    - zip_path: chemin du fichier zip
    - extract_to: dossier où extraire les fichiers
    """
    # Vérifier si le fichier zip existe
    if not os.path.exists(zip_path):
        print(f"❌ Le fichier {zip_path} est introuvable.")
        return

    # Vérifier si le dossier Data existe déjà
    if os.path.exists(extract_to) and os.listdir(extract_to):
        print(f"✅ Les données semblent déjà extraites dans le dossier '{extract_to}'.")
        return

    # Créer le dossier Data si nécessaire
    os.makedirs(extract_to, exist_ok=True)

    # Extraction
    print(f"📦 Extraction de {zip_path} vers {extract_to} ...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    print(f"✅ Extraction terminée ! Les fichiers sont disponibles dans '{extract_to}'.")


if __name__ == "__main__":
    extract_dataset()
