import os
import subprocess
from multiprocessing import Pool
from pathlib import Path

IMG_DIR = "/root/Documents/Cover"
SECRETO = "/root/Documents/theZoo/malware/Binaries/Ransomware.Matsnu/'Matsnu-MBRwipingRansomware_1B2D2A4B97C7C2727D571BBF9376F54F_Inkasso Rechnung vom 27.05.2013 .com_'"
OUTPUT_DIR = "/root/Documents/fotostego/matsnu"
NUM_CORES = 64

def process_image(img_path):
    output_path = os.path.join(OUTPUT_DIR, f"{Path(img_path).stem}_oculta.png")
    cmd = [
        "openstego", "embed", "-a", "randomlsb",
        "-mf", SECRETO, "-cf", img_path, "-sf", output_path
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return f"Procesada: {img_path}"
    except subprocess.CalledProcessError as e:
        return f"Error en {img_path}: {e.stderr}"

def main():
    # Crear directorio de salida
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Listar todas las imágenes JPG
    images = [str(p) for p in Path(IMG_DIR).glob("*.jpg")]
    
    # Procesar en paralelo
    with Pool(NUM_CORES) as pool:
        results = pool.map(process_image, images)
    
    # Imprimir resultados
    for result in results:
        print(result)

if __name__ == "__main__":
    main()
