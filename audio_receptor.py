import numpy as np
import pyaudio
import wave
import threading
import time
from scipy import signal
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from collections import deque
import json
from typing import List, Tuple, Dict, Any
from datetime import datetime

# Instalar dependencias: pip install pyaudio scipy matplotlib numpy

# Clase auxiliar para crear versión simplificada de MajoranaQubit si no está disponible
class SimpleMajoranaQubit:
    """Versión simplificada para demo"""
    def __init__(self, initial_state=None):
        if initial_state is None:
            self.state = np.array([1, 0], dtype=complex)
        else:
            self.state = np.array(initial_state, dtype=complex)
            # Normalizar el estado para asegurar que es un estado cuántico válido
            norm = np.linalg.norm(self.state)
            if norm > 1e-9:
                self.state = self.state / norm
    
    def get_bloch_vector(self) -> Tuple[float, float, float]:
        """Calcula y devuelve el vector de Bloch (x, y, z) para el estado del qubit."""
        alpha = self.state[0]
        beta = self.state[1]
        x = 2 * np.real(alpha * np.conj(beta))
        y = 2 * np.imag(alpha * np.conj(beta))
        z = np.abs(alpha)**2 - np.abs(beta)**2
        return (x, y, z)

class AudioQuantumStatePreparation:
    """Sistema que usa el micrófono para preparar estados cuánticos"""
    
    def __init__(self, 
                 sample_rate: int = 44100,
                 chunk_size: int = 1024,
                 channels: int = 1,
                 audio_format=pyaudio.paInt16):
        """
        Inicializar el sistema de captura de audio
        
        Args:
            sample_rate: Frecuencia de muestreo (Hz)
            chunk_size: Tamaño del buffer de audio
            channels: Número de canales (1 = mono)
            audio_format: Formato de audio
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.channels = channels
        self.audio_format = audio_format
        
        self.audio_buffer = deque(maxlen=sample_rate * 2)  # 2 segundos de audio
        
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        
        self.n_qubits = 5
        self.n_features = 32
        
        self.high_cutoff = 8000  # Hz
        self.low_cutoff = 200    # Hz
        
        self.quantum_state_history = deque(maxlen=100)
        self.prev_spectrum = {}

    def start_audio_capture(self):
        """Iniciar captura de audio en tiempo real"""
        try:
            self.stream = self.p.open(
                format=self.audio_format,
                channels=self.channels,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            self.is_recording = True
            self.stream.start_stream()
            print("🎤 Captura de audio iniciada")
            
        except Exception as e:
            print(f"Error al iniciar captura de audio: {e}")
    
    def stop_audio_capture(self):
        """Detener captura de audio"""
        self.is_recording = False
        if self.stream and self.stream.is_active():
            self.stream.stop_stream()
            self.stream.close()
        print("🛑 Captura de audio detenida")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback para procesar audio en tiempo real"""
        audio_data = np.frombuffer(in_data, dtype=np.int16)
        self.audio_buffer.extend(audio_data)
        return (None, pyaudio.paContinue)
    
    def get_current_audio_window(self, window_length: float = 0.1) -> np.ndarray:
        """Obtener ventana actual de audio"""
        if len(self.audio_buffer) == 0:
            return np.zeros(int(self.sample_rate * window_length))
        
        n_samples = int(self.sample_rate * window_length)
        
        if len(self.audio_buffer) >= n_samples:
            audio_window = np.array(list(self.audio_buffer)[-n_samples:])
        else:
            audio_window = np.array(list(self.audio_buffer))
            padding = n_samples - len(audio_window)
            audio_window = np.pad(audio_window, (padding, 0), mode='constant')
        
        return audio_window.astype(np.float32)
    
    def decimate_pressure_waves(self, audio_data: np.ndarray) -> Dict[str, np.ndarray]:
        """Diezmar ondas de presión en componentes de alta y baja frecuencia"""
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        nyquist = self.sample_rate / 2
        
        low_b, low_a = signal.butter(4, self.low_cutoff / nyquist, btype='low')
        low_freq_component = signal.filtfilt(low_b, low_a, audio_data)
        
        high_b, high_a = signal.butter(4, self.high_cutoff / nyquist, btype='high')
        high_freq_component = signal.filtfilt(high_b, high_a, audio_data)
        
        mid_b, mid_a = signal.butter(4, [self.low_cutoff / nyquist, self.high_cutoff / nyquist], btype='band')
        mid_freq_component = signal.filtfilt(mid_b, mid_a, audio_data)
        
        decimation_factor = 4
        low_decimated = signal.decimate(low_freq_component, decimation_factor)
        high_decimated = signal.decimate(high_freq_component, decimation_factor)
        mid_decimated = signal.decimate(mid_freq_component, decimation_factor)
        
        return {
            'low_freq': low_decimated,
            'high_freq': high_decimated,
            'mid_freq': mid_decimated,
            'original': audio_data
        }
    
    def extract_quantum_features(self, decimated_components: Dict[str, np.ndarray]) -> np.ndarray:
        """Extraer 32 características cuánticas usando FFT avanzada"""
        features = []
        
        for component_name, component_data in decimated_components.items():
            if component_name == 'original' or len(component_data) == 0:
                continue

            windowed_data = component_data * np.hamming(len(component_data))
            fft_complex = fft(windowed_data)
            fft_magnitude = np.abs(fft_complex)
            
            n_samples = len(component_data)
            freqs = fftfreq(n_samples, 1/self.sample_rate)
            positive_mask = freqs >= 0
            positive_freqs = freqs[positive_mask]
            positive_fft_mag = fft_magnitude[positive_mask]
            
            if np.sum(positive_fft_mag) > 0:
                normalized_spectrum = positive_fft_mag / np.sum(positive_fft_mag)
            else:
                normalized_spectrum = positive_fft_mag

            spectral_energy = np.sum(positive_fft_mag**2)
            features.append(spectral_energy)

            spectral_centroid = np.sum(positive_freqs * normalized_spectrum) if np.sum(normalized_spectrum) > 0 else 0
            features.append(spectral_centroid / (self.sample_rate / 2))

            if np.sum(normalized_spectrum) > 0:
                spectral_entropy = -np.sum(normalized_spectrum * np.log(normalized_spectrum + 1e-10))
            else:
                spectral_entropy = 0
            features.append(spectral_entropy / np.log(len(normalized_spectrum) + 1e-9))
        
        if len(decimated_components) >= 3:
            low_energy = np.sum(decimated_components['low_freq']**2)
            mid_energy = np.sum(decimated_components['mid_freq']**2)
            high_energy = np.sum(decimated_components['high_freq']**2)
            total_energy = low_energy + mid_energy + high_energy + 1e-10
            features.append(low_energy / total_energy)
            features.append(mid_energy / total_energy)
        
        while len(features) < self.n_features:
            features.append(0.0)
        
        features = features[:self.n_features]
        features = np.array(features, dtype=np.float32)
        features = np.clip(features, -10, 10)
        
        min_val, max_val = np.min(features), np.max(features)
        if max_val > min_val:
            features = 2 * (features - min_val) / (max_val - min_val) - 1
        
        return features

    def advanced_fft_analysis(self, audio_data: np.ndarray) -> Dict[str, Any]:
        """Análisis FFT avanzado con técnicas de procesamiento digital de señales"""
        if len(audio_data) == 0:
            return {'error': 'No hay datos de audio'}
        
        if np.max(np.abs(audio_data)) > 0:
            audio_normalized = audio_data / np.max(np.abs(audio_data))
        else:
            audio_normalized = audio_data
        
        pre_emphasis = 0.97
        emphasized_audio = np.append(audio_normalized[0], audio_normalized[1:] - pre_emphasis * audio_normalized[:-1])
        
        window = np.hamming(len(emphasized_audio))
        windowed_signal = emphasized_audio * window
        
        fft_complex = fft(windowed_signal)
        n_samples = len(windowed_signal)
        freqs = fftfreq(n_samples, 1/self.sample_rate)
        
        positive_mask = freqs >= 0
        positive_freqs = freqs[positive_mask]
        positive_fft = fft_complex[positive_mask]
        
        magnitude_spectrum = np.abs(positive_fft)
        
        return {
            'frequencies': positive_freqs.tolist(),
            'magnitude_spectrum': magnitude_spectrum.tolist(),
        }
        
    def features_to_qubit_amplitudes(self, features: np.ndarray) -> List[Tuple[complex, complex]]:
        """Convertir 32 características FFT en amplitudes de 5 qubits usando mapeo cuántico"""
        qubit_amplitudes = []
        feature_groups = np.array_split(features, self.n_qubits)
        
        for qubit_idx in range(self.n_qubits):
            qubit_features = feature_groups[qubit_idx]
            
            magnitude_features = qubit_features[::2]
            phase_features = qubit_features[1::2]
            
            magnitude_sum = np.sum(magnitude_features)
            alpha_magnitude_raw = 1 / (1 + np.exp(-magnitude_sum))
            
            entropy_factor = -np.sum(np.abs(magnitude_features) * np.log(np.abs(magnitude_features) + 1e-10))
            entropy_factor = np.clip(entropy_factor / 10.0, -0.5, 0.5)
            
            alpha_magnitude = np.sqrt(np.clip(alpha_magnitude_raw * (1 + 0.1 * entropy_factor), 0, 1))
            
            phase_sum = np.sum(phase_features)
            alpha_phase = phase_sum * 2 * np.pi
            alpha = alpha_magnitude * np.exp(1j * alpha_phase)
            
            beta_magnitude = np.sqrt(1 - alpha_magnitude**2)
            beta_phase = alpha_phase + np.pi / 2 + (0.3 * np.sin(alpha_phase))
            beta = beta_magnitude * np.exp(1j * beta_phase)
            
            qubit_amplitudes.append((alpha, beta))
        
        return qubit_amplitudes

    def prepare_quantum_state_from_audio(self) -> Dict[str, Any]:
        """Pipeline completo: Audio → Características → Estados Cuánticos"""
        audio_window = self.get_current_audio_window(window_length=0.1)
        decimated = self.decimate_pressure_waves(audio_window)
        quantum_features = self.extract_quantum_features(decimated)
        qubit_amplitudes = self.features_to_qubit_amplitudes(quantum_features)
        
        qubits = [SimpleMajoranaQubit(initial_state=[a, b]) for a, b in qubit_amplitudes]
        
        combined_state = qubits[0].state
        for i in range(1, len(qubits)):
            combined_state = np.kron(combined_state, qubits[i].state)
        
        quantum_state_data = {
            'timestamp': datetime.now().isoformat(),
            'qubits': qubits,
            'combined_state': combined_state.tolist(),
            'quantum_features': quantum_features.tolist(),
            'qubit_amplitudes': [(a.tolist(), b.tolist()) for a,b in qubit_amplitudes],
            'audio_metadata': {
                'rms_level': float(np.sqrt(np.mean(audio_window**2))),
                'peak_level': float(np.max(np.abs(audio_window))),
            }
        }
        
        self.quantum_state_history.append(quantum_state_data)
        return quantum_state_data
    
    def visualize_audio_quantum_mapping(self, quantum_state_data: Dict[str, Any]):
        """Visualizar el mapeo de audio a estados cuánticos"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("Mapeo de Audio a Estado Cuántico", fontsize=16)

        audio_window = self.get_current_audio_window()
        time_axis = np.linspace(0, len(audio_window)/self.sample_rate, len(audio_window))
        axes[0, 0].plot(time_axis, audio_window, color='c')
        axes[0, 0].set_title('Señal de Audio Original')
        
        f, t, Sxx = signal.spectrogram(audio_window, self.sample_rate)
        axes[0, 1].pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), cmap='magma')
        axes[0, 1].set_title('Espectrograma')
        
        features = quantum_state_data['quantum_features']
        axes[0, 2].bar(range(len(features)), features, color='m')
        axes[0, 2].set_title('32 Características Cuánticas')
        
        for i in range(self.n_qubits):
            ax_row, ax_col = 1, i
            if i >= 3: continue # Limitar a 3 visualizaciones de qubits
            ax = axes[ax_row, ax_col]
            qubit = quantum_state_data['qubits'][i]
            x, y, z = qubit.get_bloch_vector()
            
            ax.plot([0, x], [0, y], 'k-', lw=2)
            ax.scatter([x], [y], color='red', s=100, zorder=5, label=f'z={z:.2f}')
            ax.set_xlim([-1.1, 1.1]); ax.set_ylim([-1.1, 1.1])
            ax.set_aspect('equal'); ax.grid(True, alpha=0.3)
            ax.set_title(f'Qubit {i+1} (Proyección XY)')
            ax.legend(loc='upper right')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
        
    def continuous_quantum_monitoring(self, duration: int = 30, update_interval: float = 0.5):
        """Monitoreo continuo de estados cuánticos basados en audio"""
        print(f"🔄 Iniciando monitoreo continuo por {duration} segundos")
        if not self.is_recording: self.start_audio_capture()
        
        start_time = time.time()
        try:
            while time.time() - start_time < duration:
                state = self.prepare_quantum_state_from_audio()
                print(f"\r⏱️  {time.time() - start_time:.1f}s - "
                      f"RMS: {state['audio_metadata']['rms_level']:.4f} - "
                      f"Características: {np.linalg.norm(state['quantum_features']):.3f}", end='')
                time.sleep(update_interval)
        except KeyboardInterrupt:
            print("\n⏹️ Monitoreo interrumpido.")
        print(f"\n✅ Monitoreo completado.")

    def save_quantum_session(self, filename: str = None):
        """Guardar sesión de estados cuánticos"""
        if filename is None:
            filename = f"quantum_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(list(self.quantum_state_history), f, indent=2)
        print(f"💾 Sesión guardada en: {filename}")

def main():
    """Función principal de demostración"""
    print("🎵 PREPARACIÓN DE ESTADOS CUÁNTICOS CON AUDIO 🎵")
    print("=" * 60)
    
    quantum_prep = None
    try:
        quantum_prep = AudioQuantumStatePreparation()
        while True:
            print("\n🎮 OPCIONES:")
            print("1. 🎤 Capturar estado cuántico instantáneo")
            print("2. 🔄 Monitoreo continuo (15 segundos)")
            print("3. 📊 Visualizar último mapeo audio→cuántico")
            print("4. 💾 Guardar sesión de estados")
            print("5. ⚙️  Configurar filtros de frecuencia")
            print("0. 🚪 Salir")
            
            choice = input("\nSelecciona una opción (0-5): ").strip()
            
            if choice == '0':
                print("👋 ¡Hasta luego desde el mundo cuántico!")
                break
            
            elif choice == '1':
                print("\n🎤 Capturando audio... Haz algún ruido (habla, aplaude) en 3 segundos.")
                quantum_prep.start_audio_capture()
                time.sleep(3)
                state = quantum_prep.prepare_quantum_state_from_audio()
                quantum_prep.stop_audio_capture()
                print(f"✅ ¡Estado cuántico preparado! RMS Audio: {state['audio_metadata']['rms_level']:.4f}")
            
            elif choice == '2':
                quantum_prep.continuous_quantum_monitoring(duration=15, update_interval=0.5)
            
            elif choice == '3':
                if quantum_prep.quantum_state_history:
                    print("\n📊 Visualizando último mapeo...")
                    quantum_prep.visualize_audio_quantum_mapping(quantum_prep.quantum_state_history[-1])
                else:
                    print("❌ No hay estados para visualizar. Captura uno primero (opción 1).")
            
            elif choice == '4':
                if quantum_prep.quantum_state_history:
                    quantum_prep.save_quantum_session()
                else:
                    print("❌ No hay estados para guardar.")
            
            elif choice == '5':
                print("\n⚙️ CONFIGURACIÓN DE FILTROS:")
                try:
                    new_low = input(f"   Frec. de corte baja ({quantum_prep.low_cutoff} Hz, Enter para mantener): ").strip()
                    if new_low: quantum_prep.low_cutoff = int(new_low)
                    new_high = input(f"   Frec. de corte alta ({quantum_prep.high_cutoff} Hz, Enter para mantener): ").strip()
                    if new_high: quantum_prep.high_cutoff = int(new_high)
                    print(f"✅ Filtros actualizados: Baja < {quantum_prep.low_cutoff} Hz, Alta > {quantum_prep.high_cutoff} Hz")
                except ValueError:
                    print("❌ Entrada inválida. Introduce solo números enteros.")
            else:
                print("❌ Opción no válida. Inténtalo de nuevo.")

    except Exception as e:
        print(f"\n💥 Ocurrió un error crítico: {e}")
        print("Asegúrate de tener PyAudio instalado y un micrófono conectado.")
    finally:
        if quantum_prep and quantum_prep.is_recording:
            quantum_prep.stop_audio_capture()

if __name__ == "__main__":
    main()