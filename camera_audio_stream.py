# yamcam CLI - CeC November 2024
#
# camera_audio_stream.py --> audio streaming class
#
#
#  Class: CameraAudioStream - each sound source is analyzed in a separate thread
#
#  Methods:
#
#         __init__(self, camera_name, rtsp_url, analyze_callback)
#             Set up thread
#
#         start(self)
#             Start thread - set up FFMPEG to stream with proper settings
#
#         stop(self)
#             Stop thread
#
#         read_stream(self)
#             Continuously pull data from FFMPEG stream.  When a 31,200 byte segment
#             is in hand, convert to a form that YAMNet can classify.
#             Pass the waveform to analyze_callback (in yamnet.py) which
#             in turn calls rank_scores (in yamnet_functions.py) and returns
#             results that can be sent (via the report function in yamnet_functions.py)
#             to Home Assistant via MQTT.
#             
#         read_stderr(self)
#             Monitor stderr for messages from FFMPEG which can be informational
#             or errors, but FFMPEG does not provide a code to differentiate between them.

import threading
import subprocess
import time
import logging
import fcntl
import os

import numpy as np


logger = logging.getLogger(__name__)

class CameraAudioStream:
    def __init__(self, camera_name, rtsp_url, analyze_callback, buffer_size, shutdown_event):
        self.camera_name = camera_name
        self.rtsp_url = rtsp_url
        self.analyze_callback = analyze_callback
        self.buffer_size = buffer_size
        self.shutdown_event = shutdown_event
        self.process = None
        self.command = []
        self.read_thread = None
        self.error_thread = None
        self.timeout_thread = None
        self.running = False
        self.lock = threading.Lock()

    def start(self):
        with self.lock:
            if self.running:
                logger.warning(f"{self.camera_name}: Stream already running.")
                return
            self.running = True
            logger.debug(f"START audio stream: {self.camera_name}.")

            # Prepare the FFmpeg command with the RTSP URL including the timeout parameter
            rtsp_url_with_timeout = self._construct_rtsp_url_with_timeout()

            # FFmpeg command
            self.command = [
                'ffmpeg',
                '-rtsp_transport', 'tcp',
                '-i', rtsp_url_with_timeout,
                '-vn',  # Disable video processing
                '-f', 's16le',
                '-acodec', 'pcm_s16le',
                '-ac', '1',
                '-ar', '16000',
                '-reorder_queue_size', '0',
                '-use_wallclock_as_timestamps', '1',
                '-probesize', '50M',
                '-analyzeduration', '10M',
                '-max_delay', '500000',
                '-flags', 'low_delay',
                '-fflags', 'nobuffer',
                '-'
            ]

            logger.debug(f"{self.camera_name}: FFmpeg command: {' '.join(self.command)}")

            # Start the FFmpeg process
            self.process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                bufsize=0
            )

            # Set stdout and stderr to non-blocking mode
            self._set_non_blocking(self.process.stdout)
            self._set_non_blocking(self.process.stderr)

            # Start reading threads
            self.read_thread = threading.Thread(target=self.read_stream, name=f"ReadThread-{self.camera_name}")
            self.error_thread = threading.Thread(target=self.read_stderr, name=f"ErrorThread-{self.camera_name}")
            self.read_thread.start()
            self.error_thread.start()

            # Start the timeout monitor thread
            self.timeout_thread = threading.Thread(target=self._timeout_monitor, name=f"TimeoutThread-{self.camera_name}")
            self.timeout_thread.start()

    def _construct_rtsp_url_with_timeout(self):
        if '?' in self.rtsp_url:
            return f"{self.rtsp_url}&timeout=30000000"  # 30 seconds in microseconds
        else:
            return f"{self.rtsp_url}?timeout=30000000"

    def _set_non_blocking(self, fd):
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

    def _timeout_monitor(self):
        timeout_duration = 30  # seconds
        start_time = time.time()
        while self.running and not self.shutdown_event.is_set():
            if self.process.poll() is not None:
                # Process has terminated
                logger.debug(f"{self.camera_name}: FFmpeg process has terminated.")
                break
            if time.time() - start_time > timeout_duration:
                logger.warning(f"{self.camera_name}: FFmpeg process did not start within {timeout_duration} seconds. Terminating.")
                self.stop()
                break
            time.sleep(1)

    def read_stream(self):
        raw_audio = b""
        while self.running and not self.shutdown_event.is_set():
            try:
                chunk = self.process.stdout.read(self.buffer_size - len(raw_audio))
                if chunk:
                    raw_audio += chunk
                    if len(raw_audio) >= self.buffer_size:
                        # Convert raw PCM data to float32 array in the range [-1, 1]
                        waveform = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32)
                        waveform /= 32768.0  # Normalize to range [-1, 1]
                        self.analyze_callback(waveform, self.camera_name)
                        raw_audio = b""  # Reset buffer after processing
                else:
                    time.sleep(0.1)  # No data; wait briefly
            except BlockingIOError:
                time.sleep(0.1)  # No data available; wait briefly
            except Exception as e:
                logger.error(f"{self.camera_name}: Exception in read_stream: {e}", exc_info=True)
                break
        logger.debug(f"{self.camera_name}: Exiting read_stream.")

    def read_stderr(self):
        # Check if the FFmpeg process was created successfully
        if not self.process or self.process.poll() is not None:
            logger.error(f"{self.camera_name}: FFmpeg process failed to start or exited unexpectedly.")
            return  # Exit early if the process didn't start or has already exited
        while self.running and not self.shutdown_event.is_set():
            try:
                # Ensure stderr is still available
                if self.process.stderr:
                    line = self.process.stderr.readline()
                    if line:
                        self._handle_stderr_line(line)
                    else:
                        time.sleep(0.1)  # Brief wait to avoid tight loop if no data
                else:
                    logger.warning(f"{self.camera_name}: FFmpeg stderr is not available.")
                    break  # Exit if stderr becomes unavailable
            except (OSError, BlockingIOError):
                time.sleep(0.1)  # No data available; wait briefly
            except Exception as e:
                logger.error(f"{self.camera_name}: Exception in read_stderr: {e}", exc_info=True)
                break
        logger.debug(f"{self.camera_name}: Exiting read_stderr.")


        
    def _handle_stderr_line(self, line):
        line_decoded = line.decode('utf-8', errors='ignore').strip()
        logger.debug(f"FFmpeg stderr ({self.camera_name}): {line_decoded}")

        if "Connection timed out" in line_decoded:
            logger.warning(f"{self.camera_name}: Connection timed out.")
            self.stop()
        elif "404 Not Found" in line_decoded:
            logger.warning(f"{self.camera_name}: Stream not found (404).")
            self.stop()
        elif "Immediate exit requested" in line_decoded:
            logger.debug(f"{self.camera_name}: Immediate exit requested.")
            self.stop()
        # Add more error handling as needed

    def stop(self):
        with self.lock:
            if not self.running:
                return
            self.running = False
            self.shutdown_event.set()
            logger.debug(f"{self.camera_name}: Stopping audio stream.")
            if self.process:
                try:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        logger.warning(f"{self.camera_name}: FFmpeg process did not terminate. Killing it.")
                        self.process.kill()
                        self.process.wait()
                except Exception as e:
                    logger.error(f"{self.camera_name}: Exception while terminating FFmpeg process: {e}", exc_info=True)
                finally:
                    # Close stdout and stderr
                    if self.process.stdout:
                        self.process.stdout.close()
                    if self.process.stderr:
                        self.process.stderr.close()
                    self.process = None
            # Wait for threads to finish
            if self.read_thread and self.read_thread.is_alive():
                self.read_thread.join(timeout=5)
            if self.error_thread and self.error_thread.is_alive():
                self.error_thread.join(timeout=5)
            if self.timeout_thread and self.timeout_thread.is_alive():
                self.timeout_thread.join(timeout=5)
            logger.debug(f"{self.camera_name}: Audio stream stopped.")


