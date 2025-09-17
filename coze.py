# -*- coding: utf-8 -*-
import os
import cv2
import time
import io
import threading
import queue
import numpy as np
import pyaudio
import wave
import keyboard
import customtkinter as ctk
from PIL import Image, ImageTk
import oss2
from pydub import AudioSegment
from pydub.playback import play
import dashscope
from dashscope.audio.tts_v2 import SpeechSynthesizer
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from datetime import datetime, timedelta
import re
from cozepy import Coze, TokenAuth, COZE_CN_BASE_URL
from tools.API_key import *
# ---------------- Configuration ----------------
# OSS Configuration

# SenseVoice ASR Configuration
MODEL_DIR = "iic/SenseVoiceSmall"

# Audio Recording Configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
WAVE_OUTPUT_FILENAME = "output.wav"

# ---------------- API Clients Initialization ----------------
# Coze API Client
coze = Coze(auth=TokenAuth(COZE_API_TOKEN), base_url=COZE_BASE_URL)

# ASR Model
asr_model = AutoModel(
    model=MODEL_DIR,
    trust_remote_code=True,
    remote_code="./model.py",
    vad_model="fsmn-vad",
    vad_kwargs={"max_single_segment_time": 30000},
    device="cpu"
)


# ---------------- Utility Functions ----------------
def extract_language_emotion_content(text):
    """Extract clean content from ASR output"""
    language_start = text.find("|") + 1
    language_end = text.find("|", language_start)
    language = text[language_start:language_end]

    emotion_start = text.find("|", language_end + 1) + 1
    emotion_end = text.find("|", emotion_start)
    emotion = text[emotion_start:emotion_end]

    content_start = text.find(">", emotion_end) + 1
    content = text[content_start:]

    while content.startswith("<|"):
        end_tag = content.find(">", 2) + 1
        content = content[end_tag:]

    return content.strip()


# Coze API Call Function
def call_coze(messages):
    """Call Coze API for chat completion"""
    try:
        response = coze.chat.stream(
            bot_id=COZE_BOT_ID,
            user_id="user123",  # 可根据需要替换为动态用户 ID
            additional_messages=messages
        )
        full_response = ""
        for event in response:
            if event.event == "conversation.message.delta":
                full_response += event.message.content
        return full_response
    except Exception as e:
        print(f"Coze API error: {e}")
        raise


# ---------------- Camera Display Window ----------------
class CameraWindow(ctk.CTkToplevel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title("Camera Feed")
        self.geometry("640x480")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.camera_frame = ctk.CTkFrame(self)
        self.camera_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.camera_label = ctk.CTkLabel(self.camera_frame, text="Starting camera...")
        self.camera_label.pack(fill="both", expand=True)

        self.current_image = None
        self.is_closed = False

    def update_frame(self, img):
        """Update camera frame with new image"""
        if self.is_closed:
            return

        try:
            if img:
                img_resized = img.copy()
                img_resized.thumbnail((640, 480))

                ctk_img = ctk.CTkImage(light_image=img_resized, dark_image=img_resized, size=(640, 480))
                self.camera_label.configure(image=ctk_img, text="")
                self.current_image = ctk_img
        except Exception as e:
            print(f"Error updating camera frame: {e}")

    def on_closing(self):
        """Handle window close event"""
        self.is_closed = True
        self.withdraw()


# ---------------- Core Functionality Classes ----------------
class AudioRecorder:
    def __init__(self, app):
        self.app = app
        self.recording = False
        self.stop_recording_flag = False
        self.audio_thread = None

    def start_recording(self):
        """Begin audio recording when 'r' key is pressed"""
        if not self.recording:
            self.recording = True
            self.stop_recording_flag = False
            self.audio_thread = threading.Thread(target=self._record_audio)
            self.audio_thread.daemon = True
            self.audio_thread.start()
            self.app.ui_update_queue.put(lambda: self.app.update_status("Recording..."))

    def stop_recording(self):
        """Stop audio recording when 's' key is pressed"""
        if self.recording:
            self.stop_recording_flag = True
            self.recording = False
            if self.audio_thread and self.audio_thread.is_alive():
                self.audio_thread.join(timeout=1.0)
            self.app.ui_update_queue.put(lambda: self.app.update_status("Processing audio..."))

    def _record_audio(self):
        """Record audio from microphone"""
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        frames = []

        while self.recording and not self.stop_recording_flag:
            try:
                data = stream.read(CHUNK)
                frames.append(data)
            except Exception as e:
                self.app.ui_update_queue.put(lambda: self.app.update_status(f"Error recording audio: {e}"))
                break

        stream.stop_stream()
        stream.close()
        p.terminate()

        if frames:
            try:
                wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))
                wf.close()

                self.app.transcribe_audio(WAVE_OUTPUT_FILENAME)
            except Exception as e:
                self.app.ui_update_queue.put(lambda: self.app.update_status(f"Error saving audio: {e}"))


class VoiceActivityDetector:
    def __init__(self, app):
        self.app = app
        self.running = False
        self.listening_thread = None
        self.detection_thread = None

        self.energy_threshold = 80
        self.dynamic_threshold = True
        self.silence_threshold = 0.8
        self.min_speech_duration = 0.3
        self.max_speech_duration = 30.0

        self.is_speaking = False
        self.speech_started = 0
        self.silence_started = 0
        self.speech_frames = []

        self.noise_levels = []
        self.max_noise_levels = 100

        self.audio = None
        self.stream = None

        self.debug = True

        self.is_calibrating = True
        self.calibration_duration = 3
        self.calibration_start_time = 0

    def start_monitoring(self):
        """Begin continuous voice monitoring"""
        if not self.running:
            self.running = True
            self.listening_thread = threading.Thread(target=self._monitor_audio)
            self.listening_thread.daemon = True
            self.listening_thread.start()
            self.app.ui_update_queue.put(lambda: self.app.update_status("语音监测启动中... 正在校准麦克风"))

    def stop_monitoring(self):
        """Stop voice monitoring"""
        self.running = False
        if self.listening_thread and self.listening_thread.is_alive():
            self.listening_thread.join(timeout=1.0)
        if self.audio and self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()
            self.audio = None
            self.stream = None

    def _get_energy(self, audio_data):
        """Calculate audio energy level"""
        try:
            data = np.frombuffer(audio_data, dtype=np.int16)

            if len(data) == 0 or np.all(data == 0):
                return 0.0

            energy = np.mean(np.abs(data))
            return energy
        except Exception as e:
            print(f"Error calculating energy: {e}")
            return 0.0

    def _is_speech(self, audio_data, energy=None):
        """Detect if audio chunk contains speech based on energy level"""
        try:
            if hasattr(self.app, 'is_playing_audio') and self.app.is_playing_audio:
                if self.debug and time.time() % 2 < 0.1:
                    print("语音监测暂停中 - 正在播放系统语音")
                return False

            if energy is None:
                energy = self._get_energy(audio_data)

            if self.is_calibrating:
                self.noise_levels.append(energy)
                return False

            threshold = self.energy_threshold
            if self.dynamic_threshold and len(self.noise_levels) > 0:
                noise_avg = sum(self.noise_levels) / len(self.noise_levels)
                dynamic_threshold = noise_avg * 2.5
                threshold = max(threshold, dynamic_threshold)

            if self.debug and time.time() % 1 < 0.1:
                pass
            return energy > threshold
        except Exception as e:
            print(f"Error in speech detection: {e}")
            return False

    def _calibrate_microphone(self):
        """Calibrate microphone by measuring background noise"""
        try:
            if not self.stream or not self.audio:
                raise RuntimeError("音频流或音频对象未初始化")

            self.calibration_start_time = time.time()
            self.is_calibrating = True
            self.noise_levels = []

            print("开始麦克风校准...")
            self.app.ui_update_queue.put(lambda: self.app.update_status("校准麦克风中，请保持安静..."))

            while self.is_calibrating and time.time() - self.calibration_start_time < self.calibration_duration:
                try:
                    audio_data = self.stream.read(CHUNK, exception_on_overflow=False)
                    energy = self._get_energy(audio_data)
                    self.noise_levels.append(energy)
                    time.sleep(0.1)
                except Exception as e:
                    print(f"校准期间读取音频错误: {e}")

            if len(self.noise_levels) > 0:
                avg_noise = sum(self.noise_levels) / len(self.noise_levels)
                self.energy_threshold = max(100, avg_noise * 2.5)

                print(f"麦克风校准完成: 平均噪音级别 {avg_noise:.1f}, 阈值设为 {self.energy_threshold:.1f}")
                self.app.ui_update_queue.put(
                    lambda: self.app.update_status(f"语音监测已启动 (阈值: {self.energy_threshold:.1f})"))
            else:
                print("校准失败: 没有收集到噪音样本")
                self.app.ui_update_queue.put(lambda: self.app.update_status("语音监测已启动，但校准失败"))

            self.is_calibrating = False
        except Exception as e:
            print(f"麦克风校准错误: {e}")
            self.is_calibrating = False
            self.app.ui_update_queue.put(lambda: self.app.update_status(f"语音监测已启动，但校准出错: {e}"))

    def _monitor_audio(self):
        """Continuously monitor audio for speech"""
        try:
            self.audio = pyaudio.PyAudio()
            self.stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK
            )

            self._calibrate_microphone()

            while self.running:
                try:
                    audio_data = self.stream.read(CHUNK, exception_on_overflow=False)
                    energy = self._get_energy(audio_data)

                    if not self.is_speaking and len(self.noise_levels) < self.max_noise_levels:
                        self.noise_levels.append(energy)
                        if len(self.noise_levels) > self.max_noise_levels:
                            self.noise_levels.pop(0)

                    if self._is_speech(audio_data, energy):
                        if not self.is_speaking:
                            self.is_speaking = True
                            self.speech_started = time.time()
                            self.speech_frames = []
                            print("语音开始检测中...")
                            self.app.ui_update_queue.put(lambda: self.app.update_status("检测到语音输入..."))

                        self.silence_started = 0
                        self.speech_frames.append(audio_data)

                        if time.time() - self.speech_started > self.max_speech_duration:
                            print(f"达到最大语音长度 ({self.max_speech_duration}s)，开始处理")
                            self._process_speech()

                    elif self.is_speaking:
                        if self.silence_started == 0:
                            self.silence_started = time.time()
                            print(f"检测到语音之后的静音")

                        self.speech_frames.append(audio_data)

                        silence_duration = time.time() - self.silence_started
                        if silence_duration > self.silence_threshold:
                            print(
                                f"静音时长达到阈值 ({silence_duration:.2f}s > {self.silence_threshold}s)，开始处理语音")
                            self._process_speech()

                    time.sleep(0.01)

                except Exception as e:
                    error_msg = f"音频监测错误: {e}"
                    print(error_msg)
                    self.app.ui_update_queue.put(lambda: self.app.update_status(error_msg))
                    time.sleep(0.5)

        except Exception as e:
            error_msg = f"语音监测失败: {e}"
            print(error_msg)
            self.app.ui_update_queue.put(lambda: self.app.update_status(error_msg))
        finally:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            if self.audio:
                self.audio.terminate()

    def _process_speech(self):
        """Process detected speech segment"""
        speech_duration = time.time() - self.speech_started

        if speech_duration >= self.min_speech_duration and len(self.speech_frames) > 0:
            print(f"处理语音片段: {speech_duration:.2f}秒, {len(self.speech_frames)} 帧")

            is_speaking_was = self.is_speaking
            self.is_speaking = False
            self.silence_started = 0

            frames_copy = self.speech_frames.copy()
            self.speech_frames = []

            if is_speaking_was and speech_duration > 0.5:
                self.detection_thread = threading.Thread(
                    target=self._save_and_transcribe,
                    args=(frames_copy,)
                )
                self.detection_thread.daemon = True
                self.detection_thread.start()
            else:
                print(f"语音太短或者无效: {speech_duration:.2f}秒")
                self.app.ui_update_queue.put(lambda: self.app.update_status("Ready"))
        else:
            print(f"语音太短 ({speech_duration:.2f}秒 < {self.min_speech_duration}秒)，忽略")
            self.is_speaking = False
            self.silence_started = 0
            self.speech_frames = []
            self.app.ui_update_queue.put(lambda: self.app.update_status("Ready"))

    def _save_and_transcribe(self, frames):
        """Save speech frames to file and start transcription"""
        try:
            temp_filename = f"./speech/speech_{int(time.time())}.wav"
            print(f"保存语音到 {temp_filename}")

            if not self.audio:
                print("错误: 音频对象不存在，无法保存语音")
                self.app.ui_update_queue.put(lambda: self.app.update_status("错误: 音频对象不存在"))
                return

            if not frames or len(frames) == 0:
                print("错误: 没有语音帧可以保存")
                self.app.ui_update_queue.put(lambda: self.app.update_status("错误: 没有语音帧"))
                return

            wf = wave.open(temp_filename, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            if os.path.exists(temp_filename) and os.path.getsize(temp_filename) > 0:
                print(f"语音文件已保存: {temp_filename}, 大小: {os.path.getsize(temp_filename)} 字节")
            else:
                print(f"保存语音文件失败: {temp_filename}")
                self.app.ui_update_queue.put(lambda: self.app.update_status(f"保存语音文件失败: {temp_filename}"))
                return

            self.app.ui_update_queue.put(lambda: self.app.update_status(f"正在转写语音: {temp_filename}"))
            self.app.after(100, lambda: self._send_for_transcription(temp_filename))

        except Exception as e:
            error_msg = f"处理语音出错: {e}"
            print(error_msg)
            self.app.ui_update_queue.put(lambda: self.app.update_status(error_msg))

    def _send_for_transcription(self, audio_file):
        """Send audio file for transcription after UI is updated"""
        try:
            print(f"发送语音文件进行转写: {audio_file}")
            self.app.transcribe_audio(audio_file, priority=True)
        except Exception as e:
            error_msg = f"发送转写请求时出错: {e}"
            print(error_msg)
            self.app.ui_update_queue.put(lambda: self.app.update_status(error_msg))


class WebcamHandler:
    def __init__(self, app):
        self.app = app
        self.running = False
        self.paused = False
        self.processing = False
        self.cap = None
        self.webcam_thread = None
        self.last_webcam_image = None
        self.debug = True

        self.camera_window = None

    def start(self):
        """Start webcam capture process"""
        if not self.running:
            try:
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    self.app.ui_update_queue.put(lambda: self.app.update_status("Cannot open webcam"))
                    return False

                self.running = True

                self.create_camera_window()

                self.webcam_thread = threading.Thread(target=self._process_webcam)
                self.webcam_thread.daemon = True
                self.webcam_thread.start()

                return True
            except Exception as e:
                self.app.ui_update_queue.put(lambda: self.app.update_status(f"Error starting webcam: {e}"))
                return False
        return False

    def create_camera_window(self):
        """Create a window to display the camera feed"""
        if not self.camera_window or self.camera_window.is_closed:
            self.camera_window = CameraWindow(self.app)
            self.camera_window.title("Camera Feed")
            main_x = self.app.winfo_x()
            main_y = self.app.winfo_y()
            self.camera_window.geometry(f"640x480+{main_x + self.app.winfo_width() + 10}+{main_y}")

    def stop(self):
        """Stop webcam capture process"""
        self.running = False
        if self.cap:
            self.cap.release()

        if self.camera_window:
            self.camera_window.destroy()
            self.camera_window = None

    def _process_webcam(self):
        """Main webcam processing loop - just keeps the most recent frame"""
        last_ui_update_time = 0
        ui_update_interval = 0.05

        while self.running:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    self.app.ui_update_queue.put(lambda: self.app.update_status("Failed to capture frame"))
                    time.sleep(0.1)
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)

                self.last_webcam_image = img

                current_time = time.time()
                if self.camera_window and not self.camera_window.is_closed and current_time - last_ui_update_time >= ui_update_interval:
                    self.camera_window.update_frame(img)
                    last_ui_update_time = current_time

                time.sleep(0.03)
            except Exception as e:
                error_msg = f"Webcam error: {e}"
                print(error_msg)
                self.app.ui_update_queue.put(lambda: self.app.update_status(error_msg))
                time.sleep(1)

    def toggle_pause(self):
        """Toggle pause state for webcam"""
        self.paused = not self.paused
        status = "paused" if self.paused else "resumed"
        self.app.ui_update_queue.put(lambda: self.app.update_status(f"Webcam {status}"))


class AudioPlayer:
    def __init__(self, app):
        self.app = app
        self.playing = False
        self.skip_flag = False
        self.play_thread = None
        self.tts_queue = queue.PriorityQueue()

    def play_text(self, text, priority=2):
        """Add text to playback queue with priority"""
        self.tts_queue.put((priority, text))
        if not self.playing:
            self.play_thread = threading.Thread(target=self._process_queue)
            self.play_thread.daemon = True
            self.play_thread.start()

    def _process_queue(self):
        """Process TTS queue"""
        self.playing = True
        while not self.tts_queue.empty() and not self.skip_flag:
            _, text = self.tts_queue.get()
            self._synthesize_and_play(text)
        self.playing = False
        self.skip_flag = False

    def _synthesize_and_play(self, text):
        """Synthesize and play text using TTS"""
        try:
            result = SpeechSynthesizer.call(model=TTS_MODEL,
                                            text=text,
                                            voice=TTS_VOICE,
                                            format='mp3')
            if result.get_audio_data() is not None:
                audio_data = io.BytesIO(result.get_audio_data())
                sound = AudioSegment.from_file(audio_data, format="mp3")
                play(sound)
        except Exception as e:
            print(f"TTS error: {e}")

    def skip_current(self):
        """Skip current playback"""
        self.skip_flag = True
        self.tts_queue = queue.PriorityQueue()


class MultimediaAssistantApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Multimedia Assistant")
        self.geometry("800x600")

        self.ui_update_queue = queue.Queue()

        self.is_playing_audio = False
        self.chat_context = []
        self.message_queue = queue.PriorityQueue()
        self.placeholder_map = {}
        self.message_id = 0
        self.chat_row = 0
        self.last_timestamp = 0
        self.timestamp_interval = 300

        self.message_font = ctk.CTkFont(family="Arial", size=12)
        self.timestamp_font = ctk.CTkFont(family="Arial", size=10, slant="italic")

        self.setup_ui()

        self.audio_recorder = AudioRecorder(self)
        self.voice_detector = VoiceActivityDetector(self)
        self.webcam_handler = WebcamHandler(self)
        self.audio_player = AudioPlayer(self)

        self.process_ui_updates()

        # 延迟启动语音检测以确保主线程初始化完成
        self.after(1000, self.voice_detector.start_monitoring)
        self.webcam_handler.start()
        self.start_processing_thread()
        self.setup_key_bindings()
        self.check_timestamp()

        self.ai_avatar = ctk.CTkImage(light_image=Image.new('RGB', (50, 50), color='blue'), size=(50, 50))
        self.user_avatar = ctk.CTkImage(light_image=Image.new('RGB', (50, 50), color='green'), size=(50, 50))

        self.add_ai_message("欢迎使用助手! 请直接说话。")

    def setup_ui(self):
        """Setup the main UI components"""
        self.chat_scrollable = ctk.CTkScrollableFrame(self)
        self.chat_scrollable.pack(fill="both", expand=True, padx=10, pady=10)

        self.chat_frame = ctk.CTkFrame(self.chat_scrollable)
        self.chat_frame.pack(fill="both", expand=True)

        self.controls_frame = ctk.CTkFrame(self)
        self.controls_frame.pack(fill="x", padx=10, pady=10)

        self.toggle_button = ctk.CTkButton(self.controls_frame, text="暂停分析", command=self.toggle_analysis)
        self.toggle_button.grid(row=0, column=0, padx=10, pady=5)

        self.toggle_camera_button = ctk.CTkButton(self.controls_frame, text="C", command=self.toggle_camera)
        self.toggle_camera_button.grid(row=0, column=1, padx=10, pady=5)

        self.audio_button = ctk.CTkButton(self.controls_frame, text="A", command=self.audio_action)
        self.audio_button.grid(row=0, column=2, padx=10, pady=5)

        self.video_button = ctk.CTkButton(self.controls_frame, text="V", command=self.video_action)
        self.video_button.grid(row=0, column=3, padx=10, pady=5)

        self.status_label = ctk.CTkLabel(self, text="Ready")
        self.status_label.pack(pady=5)

    def toggle_camera(self):
        """Toggle camera window"""
        if self.webcam_handler.camera_window and not self.webcam_handler.camera_window.is_closed:
            self.webcam_handler.camera_window.on_closing()
        else:
            self.webcam_handler.create_camera_window()

    def toggle_analysis(self):
        """Toggle webcam pause"""
        self.webcam_handler.toggle_pause()
        new_text = "恢复分析" if self.webcam_handler.paused else "暂停分析"
        self.toggle_button.configure(text=new_text)

    def audio_action(self):
        self.audio_recorder.start_recording()

    def video_action(self):
        self.toggle_analysis()

    def setup_key_bindings(self):
        """Set up keyboard shortcuts"""
        self.bind("<r>", lambda e: self.audio_recorder.start_recording())
        self.bind("<s>", lambda e: self.audio_recorder.stop_recording())
        self.bind("<v>", lambda e: self.toggle_analysis())
        self.bind("<space>", lambda e: self.skip_audio())

    def start_processing_thread(self):
        """Start background processing thread"""
        self.processing_running = True
        self.processing_thread = threading.Thread(target=self.process_message_queue)
        self.processing_thread.daemon = True
        self.processing_thread.start()

    def process_message_queue(self):
        """Process messages from queue"""
        while self.processing_running:
            if not self.message_queue.empty():
                priority, msg_id, message = self.message_queue.get()
                self.handle_message(message, msg_id)
                self.message_queue.task_done()
            time.sleep(0.05)

    def handle_message(self, message, msg_id):
        """Handle message types"""
        if message["type"] == "voice_input":
            self.process_voice_input(message["content"], message.get("placeholder_id"))

    def process_voice_input(self, text, placeholder_id=None):
        """Process voice input with Coze"""
        self.add_user_message(text)
        try:
            self.chat_context.append({"role": "user", "content": text})
            response = call_coze(self.chat_context)
            self.add_ai_message(response)
            self.audio_player.play_text(response, priority=1)
            self.chat_context.append({"role": "assistant", "content": response})
        except Exception as e:
            self.ui_update_queue.put(lambda: self.update_status(f"Coze error: {e}"))

    def transcribe_audio(self, audio_file, priority=False, placeholder_id=None):
        """Transcribe recorded audio"""
        try:
            res = asr_model.generate(input=audio_file, cache={}, language="auto", use_itn=False, ban_emo_unk=True,
                                     batch_size_s=60, merge_vad=True, merge_length_s=15)
            if res and "text" in res[0]:
                text = extract_language_emotion_content(res[0]["text"])
                if text.strip() and len(text.strip()) >= 2 and text.strip().startswith('小乐'):
                    priority_level = 1 if priority else 2
                    self.message_queue.put((priority_level, self.message_id,
                                            {"type": "voice_input", "content": text, "placeholder_id": placeholder_id}))
                    self.message_id += 1
        except Exception as e:
            self.ui_update_queue.put(lambda: self.update_status(f"Transcribe error: {e}"))

    def process_ui_updates(self):
        """Process UI update queue"""
        try:
            while not self.ui_update_queue.empty():
                update_func = self.ui_update_queue.get_nowait()
                update_func()
                self.ui_update_queue.task_done()
        except queue.Empty:
            pass
        self.after(50, self.process_ui_updates)

    def update_status(self, text):
        """Update the status message in the main thread"""
        self.status_label.configure(text=text)

    def add_ai_message(self, text):
        """Add AI message to UI"""
        self.ui_update_queue.put(lambda: self._add_ai_message(text))

    def _add_ai_message(self, text):
        frame = ctk.CTkFrame(self.chat_frame)
        frame.grid(row=self.chat_row, column=0, sticky="w", pady=5)
        ctk.CTkLabel(frame, text=text).pack()
        self.chat_row += 1
        self.scroll_to_bottom()

    def add_user_message(self, text):
        """Add user message to UI"""
        self.ui_update_queue.put(lambda: self._add_user_message(text))

    def _add_user_message(self, text):
        frame = ctk.CTkFrame(self.chat_frame)
        frame.grid(row=self.chat_row, column=0, sticky="e", pady=5)
        ctk.CTkLabel(frame, text=text).pack()
        self.chat_row += 1
        self.scroll_to_bottom()

    def scroll_to_bottom(self):
        """Scroll chat to bottom"""
        self.chat_scrollable._parent_canvas.yview_moveto(1.0)

    def skip_audio(self):
        """Skip currently playing audio and toggle analysis pause when spacebar is pressed"""
        self.audio_player.skip_current()
        self.webcam_handler.toggle_pause()

        if self.webcam_handler.camera_window and self.webcam_handler.camera_window.is_closed:
            self.webcam_handler.create_camera_window()
        elif self.webcam_handler.camera_window and not self.webcam_handler.camera_window.is_closed:
            self.webcam_handler.camera_window.on_closing()

    def check_timestamp(self):
        """Check if we need to display a new timestamp"""
        current_time = time.time()
        if current_time - self.last_timestamp >= self.timestamp_interval:
            now = datetime.now().strftime("%m月%d日 %H:%M")
            self.add_ai_message(now)
            self.last_timestamp = current_time

        self.after(5000, self.check_timestamp)


# ---------------- Main Function ----------------
def main():
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")

    app = MultimediaAssistantApp()
    app.protocol("WM_DELETE_WINDOW", lambda: quit_app(app))
    app.mainloop()


def quit_app(app):
    """Clean shutdown of the application"""
    if hasattr(app, 'webcam_handler'):
        app.webcam_handler.stop()

    if hasattr(app, 'voice_detector'):
        app.voice_detector.stop_monitoring()

    if hasattr(app, 'processing_running'):
        app.processing_running = False

    if hasattr(app, 'audio_player'):
        app.audio_player.skip_current()

    try:
        for file in os.listdir():
            if file.startswith("output") and (file.endswith(".mp3") or file.endswith(".wav")):
                os.remove(file)
                print(f"删除临时文件: {file}")
    except Exception as e:
        print(f"清理临时文件时出错: {e}")

    app.destroy()


if __name__ == "__main__":
    main()