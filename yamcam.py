#
# Yamcam Sound Profiler (YSP) - CeC November 2024
#
# yamcam.py 
#
import time
import threading 
import logging
import signal
import sys
import traceback
import faulthandler
from yamcam_functions import (
    analyze_audio_waveform,
    rank_sounds, update_sound_window,
    shutdown_event
)
import yamcam_config  # all setup and config happens here
from yamcam_config import logger
from yamcam_supervisor import CameraStreamSupervisor  # Import the supervisor
from yamcam_config import interpreter, input_details, output_details

# Not using at the moment...but handy to resurrect for any thorny bugs in future
# Enable faulthandler to dump tracebacks
faulthandler.enable()

def dump_all_thread_traces():
    for thread in threading.enumerate():
        if thread == threading.current_thread():
            continue  # Skip the current thread to avoid clutter
        thread_name = thread.name
        thread_id = thread.ident
        logger.debug(f"Thread {thread_name} (ID: {thread_id}) stack trace:")
        stack = sys._current_frames()[thread_id]
        traceback.print_stack(stack)

# -------- PULL FROM CONFIG FILE
camera_settings = yamcam_config.camera_settings

# -------- GLOBALS
running = True

# -------- SHUT-DOWN HANDLER
def shutdown(signum, frame):
    global running
    logger.info(f"Received shutdown signal (signal {signum}), shutting down...")
    #faulthandler.dump_traceback()  # Dump stack traces of all threads
    #dump_all_thread_traces()
    running = False  # Set the running flag to False to exit the main loop
    shutdown_event.set()  # Signal all threads to shut down
    supervisor.shutdown_event.set()  # Signal the supervisor to shut down
    supervisor.stop_all_streams()  # Stop all camera streams
    sys.exit(0)  # Exit the program

# -------- REGISTER SHUTDOWN HANDLER
signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)

#                                                #
### ---------- SOUND ANALYSIS HUB -------------###
#                                                #

def analyze_callback(waveform, camera_name):
    try:
        if shutdown_event.is_set():
            return

        # Use the shared interpreter, input, and output details from yamcam_config
        scores = analyze_audio_waveform(waveform, camera_name, interpreter, input_details, output_details)

        if shutdown_event.is_set():
            return
        if scores is not None:
            results = rank_sounds(scores, camera_name)
            if shutdown_event.is_set():
                return
            detected_sounds = [
                result['class']
                for result in results
                if result['class'] in yamcam_config.sounds_to_track
            ]
            update_sound_window(camera_name, detected_sounds)
        else:
            if not shutdown_event.is_set():
                logger.error(f"FAILED to analyze audio: {camera_name}")
    except Exception as e:
        logger.error(f"Exception in analyze_callback for {camera_name}: {e}", exc_info=True)


#                                                #
### ---------- START STREAMS ------------------###
#                                                #

# Create and start streams using the supervisor
supervisor = CameraStreamSupervisor(camera_settings, analyze_callback, shutdown_event)
supervisor.start_all_streams()

#                                                #
### ---------- MAIN ---------------------------###
#                                                #

try:
    while running and not shutdown_event.is_set():
        time.sleep(1)  # Keep the main thread running
except KeyboardInterrupt:
    shutdown(signal.SIGINT, None)
finally:
    try:
        logger.debug("******------> STOPPING ALL audio streams...")
        supervisor.stop_all_streams()
        logger.debug("All audio streams stopped. Exiting.")

        # Ensure all non-daemon threads are terminated
        for t in threading.enumerate():
            if t is not threading.main_thread():
                logger.debug(f"Thread still alive: {t.name}, daemon={t.daemon}")
                t.join(timeout=5)
        logging.shutdown()  # Make sure all logs are flushed
        sys.exit(0)
    except Exception as e:
        logger.error(f"Exception in finally block: {e}", exc_info=True)
        sys.exit(1)

