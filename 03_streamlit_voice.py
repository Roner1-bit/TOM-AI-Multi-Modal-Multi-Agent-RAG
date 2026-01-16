import streamlit as st
import threading, queue, time
import sounddevice as sd
import soundfile as sf
import numpy as np


def record_until_stop(stop_evt, out_q):
    dev = sd.query_devices(sd.default.device[0], "input")
    sr = int(dev.get("default_samplerate", 44100))

    frames = []
    q = queue.Queue()

    def cb(indata, frames_count, time_info, status):
        q.put(indata.copy())

    with sd.InputStream(samplerate=sr, channels=1, dtype="float32", callback=cb):
        while not stop_evt.is_set():
            try:
                frames.append(q.get(timeout=0.1))
            except queue.Empty:
                pass

    audio = np.concatenate(frames, axis=0) if frames else np.zeros((0, 1), dtype="float32")
    sf.write("streamlit_voice.wav", audio, sr)
    out_q.put(("streamlit_voice.wav", sr))


st.title("Voice Start/Stop (Minimal)")

if "is_rec" not in st.session_state:
    st.session_state.is_rec = False
if "stop_evt" not in st.session_state:
    st.session_state.stop_evt = None
if "thread" not in st.session_state:
    st.session_state.thread = None
if "out_q" not in st.session_state:
    st.session_state.out_q = queue.Queue()
if "awaiting" not in st.session_state:
    st.session_state.awaiting = False
if "await_start" not in st.session_state:
    st.session_state.await_start = 0.0

# auto-rerun while waiting
if st.session_state.awaiting and st.session_state.thread is not None:
    if time.time() - st.session_state.await_start > 20:
        st.session_state.awaiting = False
        st.error("Timed out waiting for recording to finish.")
    elif st.session_state.thread.is_alive():
        time.sleep(0.2)
        st.rerun()
    else:
        st.session_state.awaiting = False
        st.rerun()

# consume queue
if (not st.session_state.is_rec) and (not st.session_state.out_q.empty()):
    wav, sr = st.session_state.out_q.get_nowait()
    st.success(f"Saved: {wav} @ {sr} Hz")

c1, c2 = st.columns(2)
with c1:
    if st.button("Start", disabled=st.session_state.is_rec):
        st.session_state.stop_evt = threading.Event()
        st.session_state.is_rec = True
        st.session_state.thread = threading.Thread(
            target=record_until_stop,
            args=(st.session_state.stop_evt, st.session_state.out_q),
            daemon=True,
        )
        st.session_state.thread.start()
        st.rerun()

with c2:
    if st.button("Stop", disabled=not st.session_state.is_rec):
        st.session_state.stop_evt.set()
        st.session_state.is_rec = False
        st.session_state.awaiting = True
        st.session_state.await_start = time.time()
        st.rerun()

if st.session_state.is_rec:
    st.info("Recording...")
elif st.session_state.awaiting:
    st.info("Finishing...")
