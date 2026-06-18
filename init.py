"""
Script de Inicialización para Google Colab
------------------------------------------
Este script contiene los pasos requeridos para preparar el entorno de trabajo en Google Colab:
1. Monta la unidad de Google Drive para acceder al material de origen.
2. Copia el video original en resolución 4K desde Google Drive hacia el almacenamiento 
   local rápido de la máquina virtual de Colab (/content) para evitar desconexiones por
   red y caídas de ancho de banda de E/S.
3. Extrae la pista de audio original mediante FFmpeg y la guarda en un archivo individual (.aac).
4. Genera una copia del video original pero sin pista de audio (.mp4) para usarlo como la
   entrada de lectura limpia de los scripts de benchmark (secuencial, CPU y GPU).
"""

from google.colab import drive
import os
import subprocess
import shutil

print("=== PASO 1: Montando Google Drive ===")
# Montar Drive en el entorno de Colab
drive.mount('/content/drive')

# --- CONFIGURACIÓN DE RUTAS ---
# NOTA: Reemplaza los '...' con la ruta de tu carpeta real en Google Drive
ruta_drive_video = "/content/drive/MyDrive/.../inputVideo4k.mp4"
video_local = "/content/mi_video_local.mp4"
audio_output = "/content/audio_original.aac"
video_sin_audio = "/content/video_sin_audio.mp4"
# ------------------------------

print("\n=== PASO 2: Transfiriendo video a almacenamiento local ===")
print("Copiando archivo a /content/ para optimizar el rendimiento y prevenir errores de red...")
if os.path.exists(ruta_drive_video):
    # Se utiliza shutil.copy en lugar de !cp para evitar errores de sintaxis al ejecutar como script de Python
    shutil.copy(ruta_drive_video, video_local)
    print(f"¡Copia finalizada! Video local listo en: {video_local}")
else:
    print(f"[ERROR]: No se encontró el video en la ruta especificada de Drive: {ruta_drive_video}")
    print("Asegúrate de editar la variable 'ruta_drive_video' con tu ruta de Drive correcta.")

print("\n=== PASO 3: Separando audio y video original con FFmpeg ===")
if os.path.exists(video_local):
    # Extrae el audio de forma directa sin decodificar el video (-vn deshabilita el video)
    print("1. Extrayendo la pista de audio sin alterar el video...")
    command_audio = f'ffmpeg -y -i "{video_local}" -vn -acodec copy "{audio_output}"'
    subprocess.run(command_audio, shell=True)
    print(f"-> Audio extraído con éxito en: {audio_output}")

    # Remueve la pista de audio y copia el video intacto (-an deshabilita el audio)
    print("2. Generando archivo de video sin canal de audio...")
    command_video = f'ffmpeg -y -i "{video_local}" -an -c:v copy "{video_sin_audio}"'
    subprocess.run(command_video, shell=True)
    print(f"-> Video silencioso generado con éxito en: {video_sin_audio}")
else:
    print("[ERROR]: No se pudo realizar la separación porque el archivo local de video no existe.")