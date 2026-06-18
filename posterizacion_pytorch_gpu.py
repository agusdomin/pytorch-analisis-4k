import cv2
import timeit
import os
import numpy as np
import torch
import subprocess

# ------------------- CONFIGURACIÓN -------------------
video_entrada = "/content/video_sin_audio.mp4"
video_salida = "/content/video_procesado_pytorch_gpu.mp4"
ruta_audio = "/content/audio_original.aac"
NIVELES_POSTERIZACION = 4

# Tamaño de lote (batch size)
# Determina cuántos frames se suben y procesan en la GPU al mismo tiempo.
# 4K ocupa mucha memoria, por lo que valores entre 4 y 16 son óptimos.
BATCH_SIZE = 8

# Cantidad de corridas completas del benchmark (para obtener promedios)
RUNS = 3

# Ajusta esto para limitar los frames en cada corrida.
# Coloca 'None' para procesar el video completo.
LIMITAR_FRAMES = None
# -----------------------------------------------------

# Configurar el dispositivo (usará CUDA si está disponible)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Verificar metadatos iniciales abriendo temporalmente el video
cap = cv2.VideoCapture(video_entrada)
if not cap.isOpened():
    raise FileNotFoundError(f"No se pudo abrir el video en {video_entrada}")
fps_original = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

frames_a_procesar = LIMITAR_FRAMES if LIMITAR_FRAMES else total_frames_video
div = 256 // NIVELES_POSTERIZACION
factor = 255 // (NIVELES_POSTERIZACION - 1)

print("==================================================")
print("INICIANDO SCRIPT PYTORCH GPU POR LOTES")
print(f"Resolución: {width}x{height}")
print(f"Dispositivo detectado: {device.type.upper()}")
print(f"Tamaño de lote (Batch Size): {BATCH_SIZE}")
print(f"Cantidad de corridas (Runs): {RUNS}")
print(f"Frames por corrida: {frames_a_procesar} (de {total_frames_video} en total)")
print("==================================================\n")

# Warmup en GPU si corresponde
if device.type == 'cuda':
    print("Realizando precalentamiento (warmup) de la GPU...")
    dummy_np = np.zeros((BATCH_SIZE, height, width, 3), dtype=np.uint8)
    dummy_tensor = torch.from_numpy(dummy_np).to(device)
    dummy_out = torch.div(dummy_tensor, div, rounding_mode='floor') * factor
    torch.cuda.synchronize()
    _ = dummy_out.cpu().numpy()
    print("Warmup completado.\n")

# Listas para almacenar las métricas consolidadas de todas las corridas
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
    
    # Estructura temporal para ir acumulando los frames del lote
    batch_frames = []
    accum_read_time = [0.0]
    
    def process_batch(frames):
        # Registrar el tiempo acumulado de lectura de este lote
        decoding_times.append(accum_read_time[0])
        accum_read_time[0] = 0.0  # Resetear acumulador
        
        # 1. TIEMPO DE FILTRADO (H2D -> Procesamiento -> Sincronización -> D2H)
        t_start_filter = timeit.default_timer()
        
        # Apilar las imágenes del lote en un único array de NumPy: (B, H, W, 3)
        batch_np = np.stack(frames)
        # Convertir a tensor de PyTorch y subir a GPU
        batch_tensor = torch.from_numpy(batch_np).to(device)
        
        # Operación vectorizada del filtro en GPU para todo el lote simultáneamente
        processed_batch = torch.div(batch_tensor, div, rounding_mode='floor') * factor
        
        # Sincronización obligatoria antes de parar el reloj
        if device.type == 'cuda':
            torch.cuda.synchronize()
            
        # Descargar el lote procesado de la GPU a la CPU (NumPy)
        processed_np = processed_batch.cpu().numpy()
        
        filtering_times.append(timeit.default_timer() - t_start_filter)
        
        # 2. TIEMPO DE ESCRITURA (Codificación y BGR)
        t_start_write = timeit.default_timer()
        for f_idx in range(processed_np.shape[0]):
            frame_rgb = processed_np[f_idx]
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)
        encoding_times.append(timeit.default_timer() - t_start_write)

    # Bucle principal de lectura y procesamiento por lotes
    frame_count = 0
    try:
        for i in range(frames_a_procesar):
            t_start_read = timeit.default_timer()
            ret, frame = cap.read()
            if not ret:
                break
            accum_read_time[0] += (timeit.default_timer() - t_start_read)
            
            # Pasar a RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            batch_frames.append(frame_rgb)
            frame_count += 1
            
            # Si completamos el tamaño de lote, procesamos
            if len(batch_frames) == BATCH_SIZE:
                process_batch(batch_frames)
                batch_frames = []
                
        # Procesar el último lote remanente si no quedó vacío
        if len(batch_frames) > 0:
            process_batch(batch_frames)
            
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
print("      INFORME CONSOLIDADO PYTORCH GPU (LOTES)      ")
print("==================================================")
print(f"Resolución del video: {width}x{height}")
print(f"Dispositivo de ejecución: {device.type.upper()}")
print(f"Tamaño de lote (Batch Size): {BATCH_SIZE}")
print(f"Cantidad total de corridas evaluadas: {RUNS}")
print(f"Frames procesados por corrida: {frame_count}")
print("--------------------------------------------------")
print(f"PROMEDIO de Tiempo Total: {avg_total:.4f} segundos")
print(f"  └─ PROMEDIO de Lectura/Decodificación: {avg_decoding:.4f} s (Promedio: {avg_decoding/frame_count:.4f} s/frame)")
print(f"  └─ PROMEDIO de Filtrado (GPU + H2D/D2H): {avg_filtering:.4f} s (Promedio: {avg_filtering/frame_count:.4f} s/frame)")
print(f"  └─ PROMEDIO de Escritura/Codificación: {avg_encoding:.4f} s (Promedio: {avg_encoding/frame_count:.4f} s/frame)")
print(f"PROMEDIO de Frames por segundo efectivos: {avg_fps:.4f} FPS")
print(f"Archivo generado: {video_salida}")
print("==================================================")

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
