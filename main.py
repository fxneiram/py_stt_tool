import vosk
import pyaudio
import json
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import numpy as np

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

        # Start and Stop buttons
        self.start_button = ttk.Button(root, text="Start", command=self.start_recognition)
        self.start_button.grid(row=1, column=0, padx=10, pady=10, sticky="w")

        self.stop_button = ttk.Button(root, text="Stop", command=self.stop_recognition, state=tk.DISABLED)
        self.stop_button.grid(row=1, column=1, padx=10, pady=10, sticky="e")

        # ScrolledText widget for displaying recognized text
        self.text_output = scrolledtext.ScrolledText(root, wrap=tk.WORD, font=("Helvetica", 10), width=50, height=20)
        self.text_output.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

        # Progressbar for audio intensity
        self.progress = ttk.Progressbar(root, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.progress.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="ew")

        # Configure row and column weights for resizing
        root.grid_columnconfigure(0, weight=1)
        root.grid_columnconfigure(1, weight=2)
        root.grid_rowconfigure(2, weight=1)

        # Initialize variables
        self.recognizer = None
        self.stream = None
        self.p = None
        self.model = None
        self.running = False

    def populate_device_list(self):
        p = pyaudio.PyAudio()
        device_count = p.get_device_count()
        devices = []

        for i in range(device_count):
            device_info = p.get_device_info_by_index(i)
            try:
                device_name = device_info['name'].encode('cp1252').decode('utf-8')
            except UnicodeEncodeError:
                device_name = device_info['name']
            devices.append(device_name)

        self.device_list['values'] = devices
        self.device_list.current(0)  # Select the first device by default

    def start_recognition(self):
        selected_device_index = self.device_list.current()
        self.running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.text_output.delete(1.0, tk.END)

        self.recognition_thread = threading.Thread(target=self.recognize_speech, args=(selected_device_index,))
        self.recognition_thread.start()

    def recognize_speech(self, device_index):
        # Set the model path
        model_path = "models/vosk-model-en-us-0.42-gigaspeech"
        self.model = vosk.Model(model_path)

        # Create a recognizer
        self.recognizer = vosk.KaldiRecognizer(self.model, 16000)

        # Open the microphone stream
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=16000,
                                  input=True,
                                  input_device_index=device_index,
                                  frames_per_buffer=1024)

        self.text_output.insert(tk.END, "Listening for speech. Say 'Terminate' to stop.\n")

        while self.running:
            data = self.stream.read(1024, exception_on_overflow=False)
            # Compute the volume level and update the progress bar
            volume_level = np.frombuffer(data, dtype=np.int16).astype(np.float32).max()
            self.progress['value'] = min(volume_level / 32767 * 100, 100)

            if self.recognizer.AcceptWaveform(data):
                result = json.loads(self.recognizer.Result())
                recognized_text = result['text']
                self.text_output.insert(tk.END, recognized_text + "\n")
                self.text_output.see(tk.END)  # Scroll to the end of the text box

                if "terminate" in recognized_text.lower():
                    self.text_output.insert(tk.END, "Termination keyword detected. Stopping...\n")
                    self.stop_recognition()
                    break

        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

    def stop_recognition(self):
        self.running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = SpeechRecognizerApp(root)
    root.mainloop()
