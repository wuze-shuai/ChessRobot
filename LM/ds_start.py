import pyaudio
import wave
import time
import threading
import os
import subprocess
import queue
import logging
import numpy as np
import re
from funasr import AutoModel  # SenseVoice 模型依赖 funasr 库

# 配置
MODEL_DIR = "iic/SenseVoiceSmall"  # SenseVoice 模型路径，与 dscamera.py 一致
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
WAVE_OUTPUT_FILENAME = "temp_audio.wav"
SILENCE_THRESHOLD = 0.8  # 静音阈值（秒）
MIN_SPEECH_DURATION = 0.3  # 最小语音持续时间（秒）
CALIBRATION_DURATION = 3.0  # 麦克风校准时间（秒）
LOG_FILE = "ds_start_log.txt"
ENERGY_THRESHOLD = 80  # 初始能量阈值，与 dscamera.py 一致
MAX_NOISE_LEVELS = 100  # 最大噪声样本数

# 日志配置
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


class VoiceWakeUp:
    def __init__(self):
        self.running = False
        self.listening_thread = None
        self.audio = None
        self.stream = None
        self.is_speaking = False
        self.speech_started = 0
        self.silence_started = 0
        self.speech_frames = []
        self.message_queue = queue.PriorityQueue()
        self.processing_thread = None
        self.dscamera_process = None
        self.energy_threshold = ENERGY_THRESHOLD
        self.noise_levels = []
        self.is_calibrating = True
        self.calibration_start_time = 0

        # 初始化 SenseVoice 模型，与 dscamera.py 一致
        try:
            self.asr_model = AutoModel(
                model=MODEL_DIR,
                trust_remote_code=True,
                remote_code="./model.py",
                vad_model="fsmn-vad",
                vad_kwargs={"max_single_segment_time": 30000},
                device="cuda:0",
            )
            print("SenseVoice 模型加载成功")
            logging.info("SenseVoice 模型加载成功")
        except Exception as e:
            print(f"SenseVoice 模型加载失败: {e}")
            logging.error(f"SenseVoice 模型加载失败: {e}")
            raise

    def _calibrate_microphone(self):
        """校准麦克风，测量背景噪声"""
        print("开始麦克风校准，请保持安静...")
        logging.info("开始麦克风校准")
        self.noise_levels = []
        self.calibration_start_time = time.time()
        self.is_calibrating = True

        try:
            while time.time() - self.calibration_start_time < CALIBRATION_DURATION:
                audio_data = self.stream.read(CHUNK, exception_on_overflow=False)
                energy = self._get_energy(audio_data)
                self.noise_levels.append(energy)
                time.sleep(0.01)

            if self.noise_levels:
                avg_noise = sum(self.noise_levels) / len(self.noise_levels)
                self.energy_threshold = max(ENERGY_THRESHOLD, avg_noise * 2.5)
                print(f"校准完成，平均噪音: {avg_noise:.1f}, 能量阈值设为: {self.energy_threshold:.1f}")
                logging.info(f"校准完成，平均噪音: {avg_noise:.1f}, 能量阈值设为: {self.energy_threshold:.1f}")
            else:
                print("校准失败，使用默认能量阈值: {ENERGY_THRESHOLD}")
                logging.warning(f"校准失败，使用默认能量阈值: {ENERGY_THRESHOLD}")
                self.energy_threshold = ENERGY_THRESHOLD

            self.is_calibrating = False
        except Exception as e:
            print(f"麦克风校准失败: {e}")
            logging.error(f"麦克风校准失败: {e}")
            self.is_calibrating = False
            self.energy_threshold = ENERGY_THRESHOLD

    def start_listening(self):
        """开始语音监听"""
        if not self.running:
            self.running = True
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
                self.listening_thread = threading.Thread(target=self._monitor_audio)
                self.listening_thread.daemon = True
                self.listening_thread.start()
                self.processing_thread = threading.Thread(target=self._process_messages)
                self.processing_thread.daemon = True
                self.processing_thread.start()
                print("语音唤醒系统已启动，正在监听...")
                logging.info("语音唤醒系统已启动")
            except Exception as e:
                print(f"启动语音监听失败: {e}")
                logging.error(f"启动语音监听失败: {e}")
                self.stop_listening()

    def stop_listening(self):
        """停止语音监听"""
        self.running = False
        if self.listening_thread and self.listening_thread.is_alive():
            self.listening_thread.join(timeout=1.0)
        if self.audio and self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.audio.terminate()
        if self.dscamera_process and self.dscamera_process.poll() is None:
            self.dscamera_process.terminate()
        print("语音唤醒系统已停止")
        logging.info("语音唤醒系统已停止")

    def _get_energy(self, audio_data):
        """计算音频能量值"""
        try:
            data = np.frombuffer(audio_data, dtype=np.int16)
            if len(data) == 0:
                return 0.0
            return np.mean(np.abs(data))
        except Exception as e:
            logging.error(f"计算能量值错误: {e}")
            print(f"计算能量值错误: {e}")
            return 0.0

    def _monitor_audio(self):
        """持续监测音频以检测语音"""
        try:
            while self.running:
                try:
                    audio_data = self.stream.read(CHUNK, exception_on_overflow=False)
                    energy = self._get_energy(audio_data)

                    if self.is_calibrating:
                        self.noise_levels.append(energy)
                        continue

                    # 动态调整阈值
                    dynamic_threshold = self.energy_threshold
                    if len(self.noise_levels) > 0:
                        noise_avg = sum(self.noise_levels) / len(self.noise_levels)
                        dynamic_threshold = max(ENERGY_THRESHOLD, noise_avg * 2.5)

                    # 调试输出能量值
                    if time.time() % 1 < 0.1:
                        print(f"能量: {energy:.1f}, 阈值: {dynamic_threshold:.1f}")

                    if energy > dynamic_threshold:
                        if not self.is_speaking:
                            self.is_speaking = True
                            self.speech_started = time.time()
                            self.speech_frames = []
                            print("检测到语音开始...")
                            logging.info("检测到语音开始")
                        self.silence_started = 0
                        self.speech_frames.append(audio_data)
                    elif self.is_speaking:
                        if self.silence_started == 0:
                            self.silence_started = time.time()
                        self.speech_frames.append(audio_data)
                        if time.time() - self.silence_started > SILENCE_THRESHOLD:
                            print("检测到静音，开始处理语音...")
                            logging.info("检测到静音，开始处理语音")
                            self._process_speech()
                    time.sleep(0.01)
                except Exception as e:
                    logging.error(f"音频监测错误: {e}")
                    print(f"音频监测错误: {e}")
                    time.sleep(0.5)
        except Exception as e:
            logging.error(f"音频监测线程错误: {e}")
            print(f"音频监测线程错误: {e}")
        finally:
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
            if self.audio:
                self.audio.terminate()

    def _process_speech(self):
        """处理检测到的语音片段"""
        speech_duration = time.time() - self.speech_started
        if speech_duration >= MIN_SPEECH_DURATION and len(self.speech_frames) > 0:
            frames_copy = self.speech_frames.copy()
            self.is_speaking = False
            self.silence_started = 0
            self.speech_frames = []
            temp_filename = f"speech_{int(time.time())}.wav"
            try:
                wf = wave.open(temp_filename, 'wb')
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(self.audio.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames_copy))
                wf.close()
                self.message_queue.put((1, temp_filename))
                print(f"已保存音频到 {temp_filename}")
                logging.info(f"已保存音频到 {temp_filename}")
            except Exception as e:
                logging.error(f"保存音频错误: {e}")
                print(f"保存音频错误: {e}")
        else:
            self.is_speaking = False
            self.silence_started = 0
            self.speech_frames = []
            print("语音片段太短，忽略")
            logging.info("语音片段太短，忽略")

    def _process_messages(self):
        """处理消息队列中的音频文件"""
        while self.running:
            try:
                if not self.message_queue.empty():
                    print("处理消息队列...")
                    logging.info("处理消息队列")
                    priority, audio_file = self.message_queue.get()
                    self._transcribe_and_analyze(audio_file)
                    self.message_queue.task_done()
                else:
                    time.sleep(0.05)
            except Exception as e:
                logging.error(f"处理消息错误: {e}")
                print(f"处理消息错误: {e}")

    def _transcribe_and_analyze(self, audio_file):
        """转录音频并分析是否包含唤醒词"""
        try:
            # 检查音频文件是否存在且不为空
            if not os.path.exists(audio_file):
                logging.error(f"音频文件不存在: {audio_file}")
                print(f"音频文件不存在: {audio_file}")
                return
            if os.path.getsize(audio_file) == 0:
                logging.error(f"音频文件为空: {audio_file}")
                print(f"音频文件为空: {audio_file}")
                return

            # 使用 SenseVoice 模型进行语音转录，与 dscamera.py 一致
            res = self.asr_model.generate(
                input=audio_file,
                cache={},
                language="auto",
                use_itn=False,
                ban_emo_unk=True,
                batch_size_s=60,
                merge_vad=True,
                merge_length_s=15,
            )

            # 提取转录文本
            text = res[0]["text"] if res and len(res) > 0 else ""
            if text:
                # 清理 SenseVoice 输出中的语言和情感标签
                language_start = text.find("|") + 1
                language_end = text.find("|", language_start)
                emotion_start = text.find("|", language_end + 1) + 1
                emotion_end = text.find("|", emotion_start)
                content_start = text.find(">", emotion_end) + 1
                extracted_text = text[content_start:].strip()
                while extracted_text.startswith("<|"):
                    end_tag = extracted_text.find(">", 2) + 1
                    extracted_text = extracted_text[end_tag:].strip()

                print(f"转录文本: {extracted_text}")
                logging.info(f"转录文本: {extracted_text}")

                # 检查是否为空或噪音
                if not extracted_text or len(extracted_text.strip()) < 2:
                    print(f"检测到空语音或噪音: '{extracted_text}'，忽略")
                    logging.info(f"检测到空语音或噪音: '{extracted_text}'，忽略")
                    return

                # 使用正则表达式匹配“小乐小乐”
                if re.search(r'小乐.*小乐', extracted_text):
                    result = "start: 1"
                    self._start_dscamera()
                else:
                    result = "start: 0"
                print(f"匹配结果: {result}")
                logging.info(f"匹配结果: {result}")
            else:
                print("未检测到语音或转录失败")
                logging.info("未检测到语音或转录失败")

            # 删除临时文件
            if os.path.exists(audio_file):
                os.remove(audio_file)
                print(f"已删除临时文件: {audio_file}")
                logging.info(f"已删除临时文件: {audio_file}")
        except Exception as e:
            logging.error(f"转录或分析错误: {e}")
            print(f"转录或分析错误: {e}")

    def _start_dscamera(self):
        """启动 dscamera.py"""
        try:
            if not self.dscamera_process or self.dscamera_process.poll() is not None:
                if os.path.exists("dscamera.py"):
                    self.dscamera_process = subprocess.Popen(
                        ["python", "dscamera.py"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )
                    print("已启动 dscamera.py")
                    logging.info("已启动 dscamera.py")
                else:
                    logging.error("未找到 dscamera.py")
                    print("错误: 未找到 dscamera.py")
        except Exception as e:
            logging.error(f"启动 dscamera.py 错误: {e}")
            print(f"启动 dscamera.py 错误: {e}")


def main():
    voice_wakeup = VoiceWakeUp()
    try:
        voice_wakeup.start_listening()
        while True:
            print("正在监听语音输入...")
            time.sleep(2)
    except KeyboardInterrupt:
        voice_wakeup.stop_listening()
        print("程序被用户终止")
        logging.info("程序被用户终止")


if __name__ == "__main__":
    main()