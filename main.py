import sounddevice as sd
import numpy as np
from openai import OpenAI
import queue
import threading
import time
import os
import sys
import tempfile
import soundfile as sf
from collections import deque
from dotenv import load_dotenv
import asyncio
import logging
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaRelay
import whisper

load_dotenv()

os.environ["OPEN_API_KEY"] = os.getenv("OPENAI_API_KEY")

# OpenAI Whisper API 클라이언트 생성
client = OpenAI()

# 큐 설정
audio_queue = queue.Queue()

# 오디오 설정
samplerate = 16000
block_size = 4000  # 0.25초 분량

# 자막 관리를 위한 설정
caption_history = deque(maxlen=5)  # 최근 5개 문장 저장
current_caption = ""
caption_lock = threading.Lock()

# Logging 설정
logging.basicConfig(level=logging.INFO)

# Whisper 모델 로드
model = whisper.load_model("base")

# WebRTC 연결을 위한 PeerConnection
pcs = set()
relay = MediaRelay()

class AudioTrack(MediaStreamTrack):
    """
    클라이언트로부터 수신된 오디오 트랙을 처리하는 클래스
    """
    kind = "audio"

    def __init__(self, track):
        super().__init__()  # MediaStreamTrack 초기화
        self.track = track

    async def recv(self):
        frame = await self.track.recv()

        # Whisper를 사용하여 음성 인식 처리
        audio_data = frame.to_ndarray()
        result = model.transcribe(audio_data)
        logging.info(f"Transcription: {result['text']}")

        return frame

# 오디오 콜백
def audio_callback(indata, frames, time, status):
    if status:
        print(f"상태: {status}", file=sys.stderr)
    audio_queue.put(indata.copy())

# 화면 지우기 함수
def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

# 자막 출력 함수
def update_captions():
    clear_screen()
    print("\n\n\n")
    print("=" * 60)
    print("🎙️ 실시간 음성 인식 자막 (Ctrl+C로 종료)")
    print("=" * 60)

    for prev in list(caption_history)[:-1]:
        print(f"\033[90m{prev}\033[0m")

    if caption_history:
        print(list(caption_history)[-1])

    if current_caption:
        print(f"\033[1m{current_caption}\033[0m", end="▋\n")
    else:
        print("▋")
    print("=" * 60)

# 오디오 수집 스레드
def audio_collection_thread():
    try:
        with sd.InputStream(samplerate=samplerate, channels=1, 
                          callback=audio_callback, blocksize=block_size):
            print("🎙️ 실시간 STT 시작 중... 잠시만 기다려주세요.")
            while True:
                time.sleep(0.1)
    except Exception as e:
        print(f"오디오 스트림 오류: {e}", file=sys.stderr)
    except KeyboardInterrupt:
        pass

# STT 처리 스레드 (OpenAI Whisper API 최신 버전)
def stt_processing_thread():
    global current_caption
    buffer = np.zeros((0, 1), dtype=np.float32)
    max_buffer_size = samplerate * 5

    try:
        while True:
            try:
                data = audio_queue.get(timeout=1)
                buffer = np.concatenate((buffer, data), axis=0)

                if len(buffer) > max_buffer_size:
                    buffer = buffer[-max_buffer_size:]

                chunk_size = int(samplerate * 3.0)
                if len(buffer) >= chunk_size:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                        sf.write(f.name, buffer[:chunk_size], samplerate)
                        audio_file = open(f.name, "rb")
                        response = client.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_file,
                            language="ko"
                        ,
                            prompt="회의 중입니다. 또박또박 말하는 내용을 받아적어.")
                        audio_file.close()
                        os.unlink(f.name)

                    text = response.text.strip()
                    if text:
                        with caption_lock:
                            if not current_caption or text[0].isupper() or any(current_caption.endswith(p) for p in ['.', '!', '?', '。', '！', '？']):
                                if current_caption:
                                    caption_history.append(current_caption)
                                current_caption = text
                            else:
                                current_caption += " " + text
                        update_captions()
                        buffer = np.zeros((0, 1), dtype=np.float32)

                audio_queue.task_done()
            except queue.Empty:
                continue
    except KeyboardInterrupt:
        pass

async def offer(request):
    """
    클라이언트로부터 SDP Offer를 처리하고 응답을 반환
    """
    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    pc = RTCPeerConnection()
    pcs.add(pc)

    @pc.on("track")
    async def on_track(track):
        logging.info(f"Track {track.kind} received")
        if track.kind == "audio":
            pc.addTrack(AudioTrack(relay.subscribe(track)))

    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    return {
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type
    }

async def cleanup():
    """
    WebRTC 연결 정리
    """
    while True:
        await asyncio.sleep(10)
        for pc in pcs:
            if pc.connectionState == "closed":
                pcs.discard(pc)

# 메인 실행
if __name__ == "__main__":
    try:
        clear_screen()

        t1 = threading.Thread(target=audio_collection_thread)
        t2 = threading.Thread(target=stt_processing_thread)

        t1.daemon = True
        t2.daemon = True

        t1.start()
        t2.start()

        update_captions()

        while True:
            time.sleep(0.1)

    except KeyboardInterrupt:
        clear_screen()
        print("\n🛑 프로그램 종료...")
        time.sleep(0.5)
        print("👋 종료 완료")