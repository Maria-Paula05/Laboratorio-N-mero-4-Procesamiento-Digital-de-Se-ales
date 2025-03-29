# Laboratorio-Número-4-Procesamiento-Digital-de-Señales

La electromiografía (EMG) y los estudios de conducción nerviosa son pruebas que verifican qué tan bien están funcionando los músculos y los nervios que los controlan. Estos nervios controlan los músculos enviando señales eléctricas para que se muevan. A medida que los músculos reaccionan contrayéndose, emiten señales eléctricas, que luego se pueden medir.

Una prueba EMG analiza las señales eléctricas que emiten los músculos cuando están en reposo y cuando se usan.En esta práctica de laboratorio se tuvó como objetivo: aplicar el filtrado de señales continuas para procesar una señal electromigráfica y detectar la fatiga muscular a través del análisis espectral de la misma. 

A continuación, se describirá el proceso llevado a cabo para cumplir con el objetivo de la práctica:
# 1.Músculo medido:
<img src="https://github.com/user-attachments/assets/1b26a2f7-f252-4424-933f-6e6672c64e22" width="200">


El músculo escogido para la práctica fue el extensor de los dedos,los tres electrodos fueron ubicados como se puede observar en la imagen.

# 2.Configuración del DAQ:
En el presente laboratorio se utilizó un módulo DAQ el cual se encarga de :La adquisición de datos y es el proceso de medir un fenómeno eléctrico o físico como voltaje, corriente, temperatura, presión o sonido. Un sistema DAQ consiste de sensores, hardware de medidas DAQ y una PC con software programable.En este caso se utilizó para realizar adquisición de datos de una señal electromiográfica usando  un sensor de señal muscular.

# 3.Adquisición de la señal EMG
Para que este sistema de adquisición de datos(DAQ) funcionará se instaló una librería propia de DAQ en Matlab para captar la señal en tiempo real, con ayuda delsensor descrito y conexiones simples.

```python
% ======= CONFIGURACIÓN =======
device = 'Dev1';     % Nombre del DAQ
channel = 'ai0';     % Canal de entrada 
sampleRate = 1000;   % Frecuencia de muestreo (Hz)
duration = 60*4;       % Duración total (segundos)
outputFile = 'emg_signal.csv';  % Nombre del archivo a guardar

% ======= CREAR SESIÓN =======
d = daq("ni");  % Crear sesión para DAQ NI
addinput(d, device, channel, "Voltage");  % Agregar canal de entrada
d.Rate = sampleRate;

% ======= VARIABLES =======
timeVec = [];  % Vector de tiempo
signalVec = [];  % Vector de señal

% ======= CONFIGURAR GRÁFICA =======
figure('Name', 'Señal en Tiempo Real', 'NumberTitle', 'off');
h = plot(NaN, NaN);
xlabel('Tiempo (s)');
ylabel('Voltaje (V)');
title('Señal EMG en Tiempo Real');
xlim([0, duration]);
ylim([-0.5, 3]);  % Ajusta el rango de voltaje si es necesario
grid on;

% ======= ADQUISICIÓN Y GUARDADO =======
disp('Iniciando adquisición...');
startTime = datetime('now');

while seconds(datetime('now') - startTime) < duration
    % Leer una muestra
    [data, timestamp] = read(d, "OutputFormat", "Matrix");
    
    % Guardar datos en vectores
    t = seconds(datetime('now') - startTime);
    timeVec = [timeVec; t];
    signalVec = [signalVec; data];
    
    % Actualizar gráfica
    set(h, 'XData', timeVec, 'YData', signalVec);
    drawn
```
En este código, se definió:

Frecuencia de muestreo(samplerate):1000; este valor quiere decir que se toman 1000 muestras por segundo.

Tiempo de muestreo(duration):240 s lo que equivale a 4 minutos.

En cuanto a el procesamiento de la señal tomada se utilizaron las siguientes librerias:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.fftpack import fft
import scipy.stats as stats
```
Para graficar la señal de EMG tomada se usó el siguiente código:

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

# 4.Filtrado de la señal
A continuación, algunos calculos realizados para obtener filtros como se requieren:
Frecuencia de Nyquist:

![image](https://github.com/user-attachments/assets/cee2e0ab-9972-4ecb-8d36-f4cfc03aec98)

Frecuencias de corte normalizadas por Nyquist:

![image](https://github.com/user-attachments/assets/72ea20a0-a297-4674-a220-280e4382df26)

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

-Según el artículo "Extracción de 400ms de la señal EMG", publicado en ResearchGate, las señales EMG presentan una amplitud de naturaleza aleatoria que varía en el rango de [0-10] mV, con una energía útil en el rango de frecuencias de 20 a 500 Hz. De acuerdo con este artículo se definió las frecuencias de corte del filtros pasa altas.

# 5.Aventanamiento

La ventana de Hanning es una función matemática utilizada principalmente en el procesamiento de señales para suavizar los bordes de una señal,es un tipo de función de ventana que aplica una superposición ponderada a un segmento de datos, lo que ayuda a minimizar las discontinuidades abruptas en sus límites. Este efecto de suavizado es crucial en el análisis de señales, ya que reduce la fuga espectral (artefactos no deseados que pueden distorsionar el análisis).
En este caso se grafican tres ventanas: de la primera , la del medio y la última contracción muscular.

```python
# Definir tamaño de ventana en segundos
window_size1 = 2.5  # Reducida para captar frecuencias más altas
window_size2 = 2.2  # Mantiene el tamaño original
window_size3 = 2.2  # Aumentada para captar frecuencias más bajas

samples_per_window1 = int(window_size1 * fs_mean)  # Convertir a muestras
samples_per_window2 = int(window_size2 * fs_mean)
samples_per_window3 = int(window_size3 * fs_mean)

# Extraer las ventanas
first_window = filtered_signal[:samples_per_window1]
middle_window = filtered_signal[len(filtered_signal)//2 : len(filtered_signal)//2 + samples_per_window2]
last_window = filtered_signal[-samples_per_window3:]

# Aplicar ventana de Hamming a cada segmento
first_window_hamming = first_window * np.hamming(len(first_window))
middle_window_hamming = middle_window * np.hamming(len(middle_window))
last_window_hamming = last_window * np.hamming(len(last_window))

# Graficar solo las señales con Hamming
fig, axes = plt.subplots(3, 1, figsize=(10, 8))

# Primera ventana con Hamming
axes[0].plot(first_window_hamming, color='r', label="Primera Ventana")
axes[0].plot(np.hamming(len(first_window_hamming)) * max(first_window_hamming), '--', color='black', alpha=0.6, label="Hamming")
axes[0].set_title("Primera Ventana")
axes[0].set_xlabel("Muestras")
axes[0].set_ylabel("Voltaje (V)")
axes[0].legend()

# Ventana del medio con Hamming
axes[1].plot(middle_window_hamming, color='g', label="Ventana del Medio")
axes[1].plot(np.hamming(len(middle_window_hamming)) * max(middle_window_hamming), '--', color='black', alpha=0.6, label="Hamming")
axes[1].set_title("Ventana del Medio")
axes[1].set_xlabel("Muestras")
axes[1].set_ylabel("Voltaje (V)")
axes[1].legend()

# Última ventana con Hamming (Ajustada con 15 muestras más)
axes[2].plot(last_window_hamming, color='b', label="Última Ventana")
axes[2].plot(np.hamming(len(last_window_hamming)) * max(last_window_hamming), '--', color='black', alpha=0.6, label="Hamming")
axes[2].set_title("Última Ventana")
axes[2].set_xlabel("Muestras")
axes[2].set_ylabel("Voltaje (V)")
axes[2].legend()

plt.tight_layout()
plt.show()
```

![image](https://github.com/user-attachments/assets/59e156d9-2264-4653-829d-1678936114a9)

# 6.Análisis espectral

En este fragmento de código  se aplica la Transformada Rápida de Fourier (FFT) a las ventanas de la señal EMG y grafica su espectro de frecuencia. 
```python
# Función para calcular la FFT
def compute_fft(signal, fs):
    N = len(signal)  # Número de muestras
    fft_values = np.abs(fft(signal))[:N // 2]  # Magnitud de la FFT (mitad del espectro)
    freqs = np.linspace(0, fs / 2, N // 2)  # Escala de frecuencias
    return freqs, fft_values

# Calcular la FFT de cada ventana
freqs_first, fft_first = compute_fft(first_window_hamming, fs_mean)
freqs_middle, fft_middle = compute_fft(middle_window_hamming, fs_mean)
freqs_last, fft_last = compute_fft(last_window_hamming, fs_mean)

# Calcular la frecuencia media de cada ventana
freq_mean_first = np.sum(freqs_first * fft_first) / np.sum(fft_first)
freq_mean_middle = np.sum(freqs_middle * fft_middle) / np.sum(fft_middle)
freq_mean_last = np.sum(freqs_last * fft_last) / np.sum(fft_last)

# Calcular la magnitud total de cada ventana
magnitude_first = np.sum(fft_first)
magnitude_middle = np.sum(fft_middle)
magnitude_last = np.sum(fft_last)

# Imprimir resultados
print(f"Primera Ventana: Frecuencia Media: {freq_mean_first:.2f} Hz, Magnitud Total: {magnitude_first:.2f}")
print(f"Ventana del Medio: Frecuencia Media: {freq_mean_middle:.2f} Hz, Magnitud Total: {magnitude_middle:.2f}")
print(f"Última Ventana: Frecuencia Media: {freq_mean_last:.2f} Hz, Magnitud Total: {magnitude_last:.2f}")

# Graficar espectros de frecuencia
plt.figure(figsize=(12, 5))
plt.plot(freqs_first, fft_first, color='r', label="Primera Ventana")
plt.plot(freqs_middle, fft_middle, color='g', label="Ventana del Medio")
plt.plot(freqs_last, fft_last, color='b', label="Última Ventana")
plt.title("Espectros de Frecuencia de las Ventanas")
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("Magnitud")
plt.legend()
plt.grid(True)
plt.show()
```
![image](https://github.com/user-attachments/assets/50427bae-f726-40f3-8b99-6cafd61fa853)
Primera Ventana: Frecuencia Media: 38.70 Hz, Magnitud Total: 523.64

Ventana del Medio: Frecuencia Media: 39.45 Hz, Magnitud Total: 471.80

Última Ventana: Frecuencia Media: 40.38 Hz, Magnitud Total: 142.05

# 7 Prueba de hipótesis
````python
def compute_features(signal, fs):
    N = len(signal)
    fft_values = np.abs(np.fft.fft(signal))[:N // 2]  # Magnitud de la FFT
    freqs = np.linspace(0, fs / 2, N // 2)  # Frecuencias correspondientes

    freq_mean = np.sum(freqs * fft_values) / np.sum(fft_values)  # Frecuencia media
    magnitude_total = np.sum(fft_values)  # Magnitud total

    return freq_mean, magnitude_total

# Calcular características para la primera y última ventana
freq_mean_first, magnitude_first = compute_features(first_window_hamming, fs_mean)
freq_mean_last, magnitude_last = compute_features(last_window_hamming, fs_mean)

# Prueba t bilateral para frecuencia media
t_freq, p_value_freq = stats.ttest_ind(first_window_hamming, last_window_hamming, equal_var=False)

# Prueba t bilateral para magnitud total
t_mag, p_value_mag = stats.ttest_ind(np.abs(np.fft.fft(first_window_hamming)),
                                     np.abs(np.fft.fft(last_window_hamming)),
                                     equal_var=False)

# Imprimir resultados
print(f"P-value (Frecuencia Media, bilateral): {p_value_freq:.5f}")
print(f"P-value (Magnitud Total, bilateral): {p_value_mag:.5f}")

# Interpretación de los resultados
alpha = 0.05

if p_value_freq < alpha:
    print("La diferencia en la frecuencia media es estadísticamente significativa (p < 0.05).")
else:
    print("No hay suficiente evidencia para afirmar que la frecuencia media es diferente.")

if p_value_mag < alpha:
    print("La diferencia en la magnitud total es estadísticamente significativa (p < 0.05).")
else:
    print("No hay suficiente evidencia para afirmar que la magnitud total es diferente.")
````
P-value (Frecuencia Media, bilateral): 0.99433

P-value (Magnitud Total, bilateral): 0.000001

No hay suficiente evidencia para afirmar que la frecuencia media es diferente.

La diferencia en la magnitud total es estadísticamente significativa (p < 0.05).

# 8 Analisis de resultados  

- Se puede analizar de la señal original lo siguiente:se observa una señal con bastante variabilidad lo cual es normal porque las señales electromiográficas son variables y cambientes durante la contracción y relajación del músculo;también en la señal parece haber interferencias de alta frecuencia (picos muy rápidos),esta característica se da por las contracciones repetitivas que se realizaron sin descanso del músculo, y por último que la amplitud disminuye con el tiempo, lo cual podría indicar fatiga muscular.

- El filtrado de las señales es de suma importancia ya la EMG puede estar contaminado con movimientos involuntarios del sujeto, que generan frecuencias bajas (< 20 Hz)y eliminarlas ayuda a aislar la actividad muscular real además de que estudios en EMG indican que las señales musculares útiles están por encima de 20 Hz y gracias a este filtro pasa bajas se eliminan fluctuaciones lentas que no corresponden a la contracción muscular también se mantiene la información relevante de los potenciales de acción de la unidad motora.Y en cuanto al filtro pasa altas, la señal EMG puede estar contaminada con la interferencia eléctrica de 60 Hz y este filtro de igualmanera elimina ruidos electrónicos generados por el amplificador y otros equipos; este fiktro ayuda a  conservar la parte principal de la señal EMG, ya que la mayoría de la actividad muscular está entre 20 y 450 Hz, pero en estudios de fatiga, se priorizan frecuencias menores a 60 Hz.

- También se pueden observar los beneficios de usar aventamiento en este caso y son que se obtiene un mejor análisis en el dominio de la frecuencia (reduce el ruido espectral y evita bordes bruscos que podrían introducir artefactos en la señal.- Al dividir la señal en ventanas temporales y analizar su contenido espectral, se observó que la energía se redistribuye en diferentes frecuencias a medida que avanza el tiempo. Esto es un comportamiento esperable, ya que la activación muscular no es estática y puede verse influenciada por variaciones en la fuerza aplicada, el reclutamiento de unidades motoras y la adaptación neuromuscular. Sin embargo, los cambios no siguen un patrón claro de desplazamiento hacia frecuencias más bajas, lo cual suele ser característico de la fatiga muscular.

- En cuanto a la Transformada rápida de Fourier,esta herramienta matemática permite analizar la distribución de energía en diferentes frecuencias,ayuda a identificar ruidos no deseados (como interferencia eléctrica en 50-60 Hz),y es útil para extraer características de la señal, como la frecuencia media o la frecuencia mediana en estudios de fatiga muscular.
  
- El análisis espectral de una señal electromiográfica (EMG) es una herramienta clave en la evaluación de la fatiga muscular.La Transformada rápida de Fourier,es la herramienta utilizada que  permite analizar la distribución de energía en diferentes frecuencias,ayuda a identificar ruidos no deseados (como interferencia eléctrica en 50-60 Hz),y es útil para extraer características de la señal, como la frecuencia media o la frecuencia mediana en estudios de fatiga muscular. Se sabe que a medida que un músculo se fatiga, la distribución de energía en su espectro de frecuencia cambia debido a la disminución de la conducción de la fibra muscular y la acumulación de metabolitos que afectan la excitabilidad de la membrana. Un indicador comúnmente utilizado para evaluar este proceso es la frecuencia mediana del espectro de potencia, también se evaluó la frecuencia media en tres ventanas temporales de la señal EMG (inicio, mitad y final de la contracción).
y se puede analizar que no se evidenció una disminución clara de la frecuencia media a lo largo del experimento;aunque la magnitud total de la señal sí mostró una reducción, la frecuencia se mantuvo relativamente estable y esto sugiere que el músculo no alcanzó un estado de fatiga total durante el período de registro.- En un músculo fatigado, la disminución de la velocidad de conducción de las fibras musculares provoca que las señales EMG se desplacen hacia el rango de frecuencias más bajas. En este análisis, sin embargo, la evolución de la frecuencia mediana no mostró una disminución progresiva. Por el contrario, hacia el final del registro se observó un incremento en la frecuencia mediana.

- Para determinar si la reducción en la frecuencia mediana es estadísticamente significativa, se aplicó una prueba de Wilcoxon comparando las primeras cinco ventanas con las últimas cinco. El resultado obtenido (𝑝 = 0.197) indica que no existe evidencia suficiente para afirmar que la frecuencia mediana haya disminuido de manera significativa. Este resultado implica que no se puede concluir que haya una tendencia clara de fatiga en la señal analizada. No obstante, esto no significa necesariamente que el músculo no esté fatigado, sino que los cambios en la señal EMG podrían no haber sido lo suficientemente pronunciados o podrían estar enmascarados por otros factores.

# Conclusión

En este experimento, se analizó la señal EMG para evaluar la presencia de fatiga muscular mediante el procesamiento en el dominio del tiempo y la frecuencia. Si bien se esperaba una disminución progresiva de la frecuencia media como indicador de fatiga, los resultados mostraron que esta se mantuvo relativamente estable a lo largo de la prueba. Esto sugiere que el músculo no alcanzó un estado de fatiga total, posiblemente debido a que la duración de la contracción o la intensidad del esfuerzo no fueron lo suficientemente altas para generar una disminución notable en la conducción nerviosa y la activación de unidades motoras. Sin embargo, se observó una reducción en la magnitud total de la señal EMG, lo que podría indicar una ligera disminución en la activación muscular con el tiempo. Estos hallazgos resaltan la importancia de ajustar las condiciones experimentales para inducir una fatiga más evidente, como aumentar la carga muscular o extender la duración de la contracción. En futuras pruebas, estos ajustes permitirían una mejor caracterización de los efectos de la fatiga muscular y su impacto en la señal EMG, lo cual es fundamental para aplicaciones en rehabilitación, entrenamiento deportivo y diagnóstico neuromuscular.

# Referencias

Electromiografía y estudios de conducción nerviosa. (n.d.). Medlineplus.gov. Retrieved March 26, 2025, from https://medlineplus.gov/spanish/pruebas-de-laboratorio/electromiografia-y-estudios-de-conduccion-nerviosa/

Influencia en la fatiga muscular en la señal de músculos estimulados eléctricamente. Org.Co. Retrieved March 29, 2025, from http://www.scielo.org.co/scielo.php?script=sci_arttext&pid=S1794-12372007000100010

Entender la ventana de Hanning: una guía práctica para principiantes. (n.d.). Wray Castle. Retrieved March 26, 2025, from https://wraycastle.com/es/blogs/knowledge-base/hanning-window?srsltid=AfmBOoorDAgr8KgZZHGfSGRgx7zQtEtQnPjF
