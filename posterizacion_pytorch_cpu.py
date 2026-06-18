import cv2
import timeit
import os
import numpy as np
import torch
import subprocess

# ------------------- CONFIGURACIÓN -------------------
video_entrada = "/content/video_sin_audio.mp4"
video_salida = "/content/video_procesado_pytorch_cpu.mp4"
ruta_audio = "/content/audio_original.aac"
NIVELES_POSTERIZACION = 4

# Hilos que usará PyTorch en la CPU. 
# Puedes variar este número (1, 2, 4, 8) en tus pruebas de benchmark para analizar el speed-up.
NUM_THREADS = 2

# Cantidad de corridas completas del benchmark (para obtener promedios)
RUNS = 3

# Ajusta esto para tus corridas.
# Coloca 'None' para procesar el video completo.
LIMITAR_FRAMES = 60 
# -----------------------------------------------------

# Configurar el número de hilos de CPU para PyTorch
torch.set_num_threads(NUM_THREADS)

# Abrir video de entrada temporalmente para obtener metadatos
cap = cv2.VideoCapture(video_entrada)
if not cap.isOpened():
    raise FileNotFoundError(f"No se pudo abrir el video en {video_entrada}")
fps_original = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

frames_a_procesar = LIMITAR_FRAMES if LIMITAR_FRAMES else total_frames_video

print("==================================================")
print("INICIANDO BENCHMARK PYTORCH CPU (SIN GPU)")
print(f"Resolución: {width}x{height}")
print(f"Hilos de CPU configurados: {NUM_THREADS}")
print(f"Cantidad de corridas (Runs): {RUNS}")
print(f"FPS Originales: {fps_original}")
print(f"Frames a procesar: {frames_a_procesar} (de {total_frames_video} en total)")
print("==================================================\n")

# Warmup de PyTorch (primer procesamiento para inicializar librerías y evitar sesgo en el tiempo)
dummy_frame = np.zeros((height, width, 3), dtype=np.uint8)
dummy_tensor = torch.from_numpy(dummy_frame)
_ = torch.div(dummy_tensor, 256 // NIVELES_POSTERIZACION, rounding_mode='floor') * (255 // (NIVELES_POSTERIZACION - 1))

# Listas para almacenar las métricas de todas las corridas
run_total_times = []
run_decoding_times = []
run_filtering_times = []
run_encoding_times = []
run_fps_efectivos = []

for r in range(1, RUNS + 1):
    print(f"---------- CORRIDA {r} de {RUNS} ----------")
    
    cap = cv2.VideoCapture(video_entrada)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_salida, fourcc, fps_original, (width, height))
    
    decoding_times = []
    filtering_times = []
    encoding_times = []
    
    run_start_time = timeit.default_timer()
    frame_count = 0
    
    try:
        div = 256 // NIVELES_POSTERIZACION
        factor = 255 // (NIVELES_POSTERIZACION - 1)
        
        for i in range(frames_a_procesar):
            # 1. TIEMPO DE LECTURA (Decodificación)
            t_start_read = timeit.default_timer()
            ret, frame = cap.read()
            if not ret:
                break
            decoding_times.append(timeit.default_timer() - t_start_read)
            
            # 2. TIEMPO DE FILTRADO (Posterización Vectorizada PyTorch CPU)
            t_start_filter = timeit.default_timer()
            
            # Convertir frame BGR (NumPy) a RGB (Tensor de PyTorch)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(frame_rgb)  # Comparte memoria con NumPy (costo de transferencia cero)
            
            # Aplicar posterización vectorizada
            processed_tensor = torch.div(tensor, div, rounding_mode='floor') * factor
            
            # Volver a BGR y NumPy
            processed_rgb = processed_tensor.byte().numpy()
            frame_procesado = cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR)
            
            filtering_times.append(timeit.default_timer() - t_start_filter)
            
            # 3. TIEMPO DE ESCRITURA (Codificación)
            t_start_write = timeit.default_timer()
            out.write(frame_procesado)
            encoding_times.append(timeit.default_timer() - t_start_write)
            
            frame_count += 1
            
            # Mostrar avance periódicamente
            if frame_count % 10 == 0 or frame_count == 1:
                tiempo_frame = decoding_times[-1] + filtering_times[-1] + encoding_times[-1]
                
    finally:
        cap.release()
        out.release()
        
    run_end_time = timeit.default_timer()
    run_total_time = run_end_time - run_start_time
    fps_efectivos = frame_count / run_total_time
    
    # Guardar métricas de esta corrida
    run_total_times.append(run_total_time)
    run_decoding_times.append(sum(decoding_times))
    run_filtering_times.append(sum(filtering_times))
    run_encoding_times.append(sum(encoding_times))
    run_fps_efectivos.append(fps_efectivos)
    
    print(f"Corrida {r} finalizada en {run_total_time:.4f} segundos.")
    print(f"  └─ Lectura: {run_decoding_times[-1]:.4f}s | Filtrado: {run_filtering_times[-1]:.4f}s | Escritura: {run_encoding_times[-1]:.4f}s")
    print(f"  └─ FPS Efectivos: {fps_efectivos:.4f} FPS\n")

# --- REPORTE FINAL DE METRICAS PROMEDIADAS ---
avg_total = np.mean(run_total_times)
avg_decoding = np.mean(run_decoding_times)
avg_filtering = np.mean(run_filtering_times)
avg_encoding = np.mean(run_encoding_times)
avg_fps = np.mean(run_fps_efectivos)

print("\n==================================================")
print("      INFORME CONSOLIDADO PYTORCH CPU FINAL       ")
print("==================================================")
print(f"Resolución del video: {width}x{height}")
print(f"Hilos de CPU utilizados: {NUM_THREADS}")
print(f"Cantidad total de corridas evaluadas: {RUNS}")
print(f"Frames procesados por corrida: {frame_count}")
print("--------------------------------------------------")
print(f"PROMEDIO de Tiempo Total: {avg_total:.4f} segundos")
print(f"  └─ PROMEDIO de Lectura/Decodificación: {avg_decoding:.4f} s (Promedio: {avg_decoding/frame_count:.4f} s/frame)")
print(f"  └─ PROMEDIO de Filtrado (PyTorch CPU): {avg_filtering:.4f} s (Promedio: {avg_filtering/frame_count:.4f} s/frame)")
print(f"  └─ PROMEDIO de Escritura/Codificación: {avg_encoding:.4f} s (Promedio: {avg_encoding/frame_count:.4f} s/frame)")
print(f"PROMEDIO de Frames por segundo efectivos: {avg_fps:.4f} FPS")
print(f"Archivo generado: {video_salida}")
print("==================================================")

# Proyección de tiempos para el video completo en caso de haber limitado los frames
if LIMITAR_FRAMES and total_frames_video > LIMITAR_FRAMES:
    tiempo_estimado_completo = (avg_total / LIMITAR_FRAMES) * total_frames_video
    print(f"\n[INFO PROYECCIÓN]: Para procesar el video completo ({total_frames_video} frames),")
    print(f"el pipeline PyTorch CPU tardaría aproximadamente: {tiempo_estimado_completo / 60:.2f} minutes.")

# Reincorporar audio al video final utilizando FFmpeg (fuera del tiempo medido del benchmark)
video_final = video_salida.replace(".mp4", "_con_audio.mp4")
if os.path.exists(ruta_audio):
    print("\nReincorporando el audio original al video procesado...")
    command_recombine = f'ffmpeg -y -i "{video_salida}" -i "{ruta_audio}" -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 "{video_final}"'
    subprocess.run(command_recombine, shell=True)
    print(f"¡Audio reincorporado con éxito! Video final guardado en: {video_final}")
else:
    print(f"\n[AVISO]: No se encontró el archivo de audio en '{ruta_audio}'.")
    print("Para reincorporar el audio, asegúrate de que el archivo exista en la ruta indicada.")
