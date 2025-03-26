# Laboratorio-Número-4-Procesamiento-Digital-de-Señales

La electromiografía (EMG) y los estudios de conducción nerviosa son pruebas que verifican qué tan bien están funcionando los músculos y los nervios que los controlan. Estos nervios controlan los músculos enviando señales eléctricas para que se muevan. A medida que los músculos reaccionan contrayéndose, emiten señales eléctricas, que luego se pueden medir.

Una prueba EMG analiza las señales eléctricas que emiten los músculos cuando están en reposo y cuando se usan.En esta práctica de laboratorio se tuvó como objetivo: aplicar el filtrado de señales continuas para procesar una señal electromigráfica y detectar la fatiga muscular a través del análisis espectral de la misma. 

A continuación, se describirá el proceso llevado a cabo para cumplir ocn el objetivo de la práctica:

# 1.Instalación de programas

# 2.Configuración del DAQ

 # 3.Conexiones del circuito

# 4.Adquisición de la señal EMG
Librerias utilizadas
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
```
```python
# Cargar la señal EMG desde un archivo CSV
file_path = "emg_signal.csv"  # Asegúrate de que el archivo esté en el mismo directorio
df = pd.read_csv(file_path)

# Extraer datos
tiempo = df.iloc[:, 0]  # Primera columna (Tiempo)
voltaje = df.iloc[:, 1]  # Segunda columna (Voltaje)

# Estimar la frecuencia de muestreo (fs)
fs_estimates = 1 / tiempo.diff().dropna().unique()
fs_mean = fs_estimates.mean()  # Tomar un valor promedio si hay variaciones

# Graficar la señal original
plt.figure(figsize=(10, 4))
plt.plot(tiempo, voltaje, label="Señal EMG", color="b")
plt.xlabel("Tiempo (s)")
plt.ylabel("Voltaje (V)")
plt.title("Señal EMG Original")
plt.legend()
plt.grid(True)
plt.show()

print(f"Frecuencia de muestreo estimada: {fs_mean:.2f} Hz")
```
![image](https://github.com/user-attachments/assets/362fba82-22e5-4f46-869a-8c635d2db889)
Frecuencia de muestreo estimada: 124.60 Hz
-Según el artículo "Extracción de 400ms de la señal EMG", publicado en ResearchGate, las señales EMG presentan una amplitud de naturaleza aleatoria que varía en el rango de [0-10] mV, con una energía útil en el rango de frecuencias de 20 a 500 Hz. De acuerdo con este artículo se definieron las frecuencias de corte de los filtros pasa altas y pasa bajas aplicados a continuación.
# 5.Filtrado de la señal
```python
# Función para diseñar y aplicar un filtro Butterworth
def butterworth_filter(data, cutoff, fs, filter_type, order=4):
    nyquist = 0.5 * fs  # Frecuencia de Nyquist
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)
    return filtfilt(b, a, data)

# Aplicar filtro pasa altas (20 Hz)
filtered_high = butterworth_filter(voltaje, 20, fs_mean, 'high')

# Aplicar filtro pasa bajas (60 Hz)
filtered_signal = butterworth_filter(filtered_high, 60, fs_mean, 'low')

# Graficar señal original vs filtrada
plt.figure(figsize=(10, 4))
plt.plot(tiempo, voltaje, label="Señal Original", alpha=0.5, color="gray")
plt.plot(tiempo, filtered_signal, label="Señal Filtrada", color="blue")
plt.xlabel("Tiempo (s)")
plt.ylabel("Voltaje (V)")
plt.title("Señal EMG antes y después del filtrado")
plt.legend()
plt.grid(True)
plt.show()
```
![image](https://github.com/user-attachments/assets/a0442f04-8bab-4597-a135-649b880e84e1)

Los filtros que se aplicaron a la señal de EMG fueron los siguientes:
-Filtro pasa altas:

El filtro pasa altas elimina frecuencias bajas , dejando pasar solo las altas.En una señal EMG, ayuda a eliminar el ruido de baja frecuencia como el movimiento de la línea base o interferencias musculares de baja frecuencia.

La frecuencia de corte ( 20 Hz):Cualquier frecuencia por debajo de 20 Hz será atenuada,elimina ruido de baja frecuencia, como el movimiento de los electrodos o artefactos biomecánicos.

-Filtro pasa bajas
Este filtro elimina las frecuencias altas, dejando pasar solo las bajas. En EMG, ayuda a eliminar el ruido electromagnético y la interferencia de alta frecuencia (como es el ruido de 50-60 Hz de la corriente eléctrica).

-La frecuencia de corte ( 60 Hz):El filtro deja pasar frecuencias entre 0 Hz y aproximadamente 60 Hz, con una ligera atenuación cerca del punto de corte.
# 6.Aventanamiento

La ventana de Hanning es una función matemática utilizada principalmente en el procesamiento de señales para suavizar los bordes de una señal,es un tipo de función de ventana que aplica una superposición ponderada a un segmento de datos, lo que ayuda a minimizar las discontinuidades abruptas en sus límites. Este efecto de suavizado es crucial en el análisis de señales, ya que reduce la fuga espectral (artefactos no deseados que pueden distorsionar el análisis).
En este caso se grafican hasta 5 ventanas para visualizar cómo se segmenta la señal y cada curva representa un fragmento de la señal original pero suavizada por la ventana de Hamming.
```python
# Definir tamaño de ventana en segundos
window_size = 1  # 1 segundo por ventana
samples_per_window = int(window_size * fs_mean)  # Convertir a muestras

# Aplicar aventanamiento
num_windows = len(filtered_signal) // samples_per_window
windows = [filtered_signal[i * samples_per_window:(i + 1) * samples_per_window] for i in range(num_windows)]

# Aplicar ventana de Hamming
windowed_signals = [w * np.hamming(len(w)) for w in windows]

# Graficar algunas ventanas
plt.figure(figsize=(10, 4))
for i in range(min(5, len(windowed_signals))):
    plt.plot(windowed_signals[i], label=f'Ventana {i+1}')
plt.xlabel("Muestras")
plt.ylabel("Voltaje (V)")
plt.title("Señales EMG con ventana de Hamming")
plt.legend()
plt.grid(True)
plt.show()
```
![image](https://github.com/user-attachments/assets/a1685bef-0dad-4ad2-bf1c-a76048c179ad)
Los beneficios de usar aventamiento en este caso son: 

Mejor análisis en el dominio de la frecuencia (reduce el ruido espectral).

Evita bordes bruscos que podrían introducir artefactos en la señal.
# 7.Análisis espectral

En este fragmento de código  se aplica la Transformada Rápida de Fourier (FFT) a las ventanas de la señal EMG y grafica su espectro de frecuencia. 
```python
# Aplicar Transformada de Fourier (FFT) a cada ventana
fft_results = [np.fft.fft(w) for w in windowed_signals]
frequencies = np.fft.fftfreq(samples_per_window, d=1/fs_mean)

# Tomar solo la mitad del espectro (parte positiva)
half_spectrum = samples_per_window // 2
frequencies = frequencies[:half_spectrum]
fft_magnitudes = [np.abs(fft[:half_spectrum]) for fft in fft_results]

# Graficar el espectro de frecuencia de algunas ventanas
plt.figure(figsize=(10, 4))
for i in range(min(5, len(fft_magnitudes))):
    plt.plot(frequencies, fft_magnitudes[i], label=f'Ventana {i+1}')
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.title("Espectro de Frecuencia de la Señal EMG")
plt.legend()
plt.grid(True)
plt.show()
```
![image](https://github.com/user-attachments/assets/dff544d8-b993-46be-af1a-0737d43af0b1)
La FFT permite analizar la distribución de energía en diferentes frecuencias,ayuda a identificar ruidos no deseados (como interferencia eléctrica en 50-60 Hz),y es útil para extraer características de la señal, como la frecuencia media o la frecuencia mediana en estudios de fatiga muscular.
# 8.Frecuencia media y mediana de cada ventana 
```python
# Calcular frecuencia media y mediana para cada ventana
freq_mean_values = []
freq_median_values = []

for i in range(len(fft_magnitudes)):
    magnitudes = fft_magnitudes[i]
    
    # Frecuencia media
    f_mean = np.sum(frequencies * magnitudes) / np.sum(magnitudes)
    freq_mean_values.append(f_mean)

    # Frecuencia mediana
    cumulative_energy = np.cumsum(magnitudes)  # Suma acumulada
    total_energy = cumulative_energy[-1]  # Energía total
    f_median = frequencies[np.where(cumulative_energy >= total_energy / 2)[0][0]]
    freq_median_values.append(f_median)

# Mostrar resultados
for i in range(min(5, len(freq_mean_values))):
    print(f"Ventana {i+1}: Frecuencia Media = {freq_mean_values[i]:.2f} Hz, Frecuencia Mediana = {freq_median_values[i]:.2f} Hz")
```
-Ventana 1: Frecuencia Media = 39.63 Hz, Frecuencia Mediana = 39.19 Hz

-Ventana 2: Frecuencia Media = 37.75 Hz, Frecuencia Mediana = 36.18 Hz

-Ventana 3: Frecuencia Media = 40.72 Hz, Frecuencia Mediana = 42.20 Hz

-Ventana 4: Frecuencia Media = 37.82 Hz, Frecuencia Mediana = 37.18 Hz

-Ventana 5: Frecuencia Media = 41.24 Hz, Frecuencia Mediana = 44.21 Hz

En el estudio de EMG estos calculos estadísticos son usados para:

Frecuencia Media: Se usa para analizar la fatiga muscular. Si disminuye con el tiempo, indica fatiga.

Frecuencia Mediana: También se usa en fatiga y en la caracterización de diferentes tipos de actividad muscular.

# 9
# Conclusiones
# Referencias
Electromiografía y estudios de conducción nerviosa. (n.d.). Medlineplus.gov. Retrieved March 26, 2025, from https://medlineplus.gov/spanish/pruebas-de-laboratorio/electromiografia-y-estudios-de-conduccion-nerviosa/

Entender la ventana de Hanning: una guía práctica para principiantes. (n.d.). Wray Castle. Retrieved March 26, 2025, from https://wraycastle.com/es/blogs/knowledge-base/hanning-window?srsltid=AfmBOoorDAgr8KgZZHGfSGRgx7zQtEtQnPjF
