import vosk
import pyaudio
import json
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import numpy as np
import os

CHUNK_SIZE = 2048
SAMPLE_RATE = 64000

class SpeechRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Speech Recognizer")

        # Label for audio devices
        self.device_label = ttk.Label(root, text="Select Audio Device:")
        self.device_label.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        # Dropdown menu for audio devices
        self.device_list = ttk.Combobox(root, state="readonly")
        self.device_list.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
        self.populate_device_list()

        # Label for models
        self.model_label = ttk.Label(root, text="Select Model:")
        self.model_label.grid(row=1, column=0, padx=10, pady=10, sticky="w")

        # Dropdown menu for models
        self.model_list = ttk.Combobox(root, state="readonly")
        self.model_list.grid(row=1, column=1, padx=10, pady=10, sticky="ew")
        self.populate_model_list()

        # Start and Stop buttons
        self.start_button = ttk.Button(root, text="Start", command=self.start_recognition)
        self.start_button.grid(row=2, column=0, padx=10, pady=10, sticky="w")

        self.stop_button = ttk.Button(root, text="Stop", command=self.stop_recognition, state=tk.DISABLED)
        self.stop_button.grid(row=2, column=1, padx=10, pady=10, sticky="e")

        # ScrolledText widget for displaying recognized text
        self.text_output = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Helvetica", 10), width=50, height=20)
        self.text_output.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

        # Progressbar for audio intensity
        self.progress = ttk.Progressbar(root, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.progress.grid(row=4, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

        # Configure row and column weights for resizing
        root.grid_columnconfigure(0, weight=1)
        root.grid_columnconfigure(1, weight=2)
        root.grid_rowconfigure(3, weight=1)

        # Initialize variables
        self.recognizer = None
        self.stream = None
        self.p = None
        self.model = None
        self.running = False

        # Variable to track selected device and model
        self.selected_device_index = self.device_list.current()
        self.selected_model = self.model_list.get()

        # Initialize the audio stream
        self.start_audio_stream()

    def populate_device_list(self):
        p = pyaudio.PyAudio()
        device_count = p.get_device_count()
        devices = []

        for i in range(device_count):
            device_info = p.get_device_info_by_index(i)
            
            if device_info['hostApi'] == 0:
                try:
                    device_name = device_info['name'].encode('cp1252').decode('utf-8')
                except UnicodeEncodeError:
                    device_name = device_info['name']
                devices.append((i, device_name))

        self.device_list['values'] = [name for idx, name in devices]
        self.device_list.current(0)
        self.device_list.bind("<<ComboboxSelected>>", self.on_device_selected)
        self.device_indices = [idx for idx, name in devices]

    def populate_model_list(self):
        models = [folder for folder in os.listdir('models') if os.path.isdir(os.path.join('models', folder))]
        self.model_list['values'] = models
        self.model_list.current(0)
        self.model_list.bind("<<ComboboxSelected>>", self.on_model_selected)

    def on_device_selected(self, event):
        self.selected_device_index = self.device_list.current()
        self.text_output.insert(tk.END, f"Device changed to: {self.device_list.get()}\n")
        self.text_output.see(tk.END)
        self.restart_audio_stream()

    def on_model_selected(self, event):
        self.selected_model = self.model_list.get()
        self.text_output.insert(tk.END, f"Model changed to: {self.selected_model}\n")
        self.text_output.see(tk.END)

    def start_audio_stream(self):
        try:
            # Initialize PyAudio and open the stream
            self.p = pyaudio.PyAudio()
            # Obtain the selected device's info
            device_info = self.p.get_device_info_by_index(self.selected_device_index)
            
            # Extract channels and sample rate from the selected device
            channels = device_info['maxInputChannels']
            rate = int(device_info['defaultSampleRate'])

            self.stream = self.p.open(format=pyaudio.paInt16,
                                      channels=channels,
                                      rate=rate,
                                      input=True,
                                      input_device_index=self.selected_device_index,
                                      frames_per_buffer=CHUNK_SIZE,
                                      )

            # Start a thread to monitor the audio intensity
            self.audio_monitor_thread = threading.Thread(target=self.monitor_audio_intensity)
            self.audio_monitor_thread.start()

        except Exception as e:
            self.text_output.insert(tk.END, f"Error initializing audio stream: {e}\n")
            self.text_output.see(tk.END)
            self.stream = None

    def restart_audio_stream(self):
        try:
            if self.stream is not None:
                self.stream.stop_stream()
                self.stream.close()
            self.start_audio_stream()
        except Exception as e:
            self.text_output.insert(tk.END, f"Error restarting audio stream: {e}\n")
            self.text_output.see(tk.END)

    def monitor_audio_intensity(self):
        while True:
            if self.stream is not None:
                try:
                    data = self.stream.read(CHUNK_SIZE, exception_on_overflow=False)
                    volume_level = np.frombuffer(data, dtype=np.int16).astype(np.float32).max()
                    self.progress['value'] = min(volume_level / 32767 * 100, 100)
                except Exception as e:
                    self.text_output.insert(tk.END, f"Error reading audio stream: {e}\n")
                    self.text_output.see(tk.END)


    def start_recognition(self):
        self.running = True
        self.device_list.config(state=tk.DISABLED)
        self.model_list.config(state=tk.DISABLED)
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.text_output.delete(1.0, tk.END)

        self.recognition_thread = threading.Thread(target=self.recognize_speech)
        self.recognition_thread.start()

    def recognize_speech(self):
        try:
            model_path = self.selected_model  # Usar el modelo seleccionado
            self.model = vosk.Model(f"models/{model_path}")

            self.recognizer = vosk.KaldiRecognizer(self.model, SAMPLE_RATE)

            self.text_output.insert(tk.END, "...")

            while self.running:
                data = self.stream.read(CHUNK_SIZE, exception_on_overflow=False)
                if len(data) == 0:
                    print("No data")
                    break

                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    recognized_text = result['text']
                    if recognized_text:  # Asegurarse de que no esté vacío
                        # Reemplazar la última línea con el texto finalizado
                        self.text_output.delete("end-1l", "end-1c")
                        self.text_output.insert(tk.END, recognized_text + "\n")
                        print("Completed: ", recognized_text)
                        self.text_output.see(tk.END)
                else:
                    result = json.loads(self.recognizer.PartialResult())
                    recognized_text = result['partial']
                    if recognized_text:  # Asegurarse de que no esté vacío
                        # Obtener la última línea
                        last_line = self.text_output.get("end-1l", "end-1c")
                        if last_line.endswith("..."):
                            # Si la última línea termina en "...", reemplazarla
                            self.text_output.delete("end-1l", "end-1c")
                        self.text_output.insert(tk.END, recognized_text + "...")
                        self.text_output.see(tk.END)

        except Exception as e:
            self.text_output.insert(tk.END, f"Error during recognition: {e}\n")
            self.text_output.see(tk.END)



    def stop_recognition(self):  
        self.running = False
        self.device_list.config(state="readonly")  # Enable device list}
        self.model_list.config(state="readonly")
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def on_close(self):
        # Handle application closing to stop the stream and terminate PyAudio
        if self.stream is not None:
            self.stream.stop_stream()
            self.stream.close()
        if self.p is not None:
            self.p.terminate()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("800x300") # Set the window size
    root.attributes("-topmost", True)
    app = SpeechRecognizerApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
