import subprocess
import time
import os
import sys  # <-- on va utiliser python actuel
from PIL import Image
Image.MAX_IMAGE_PIXELS = None


def run_services():
    os.chdir(os.path.dirname(__file__))

    print("🚀 Lancement de l’API FastAPI...")
    # On utilise python -m uvicorn pour Windows
    api_process = subprocess.Popen([sys.executable, "-m", "uvicorn", "api:app", "--reload", "--host", "0.0.0.0", "--port", "3003"])

    time.sleep(3)

    print("📊 Lancement du dashboard Streamlit...")
    # Pareil pour streamlit, on utilise le python actuel
    dashboard_process = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "dashboard.py", "--server.maxUploadSize=800"])

    print("\n✅ Tous les services sont en cours d’exécution.")
    print("   ➤ API: http://127.0.0.1:8000/docs")
    print("   ➤ Dashboard: http://localhost:8501")

    try:
        api_process.wait()
        dashboard_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Arrêt des services...")
        api_process.terminate()
        dashboard_process.terminate()

if __name__ == "__main__":
    run_services()
