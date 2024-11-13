#
# yamcam CLI - CeC November 2024
#
# yamcam_functions.py - Functions for yamcam3
# 
#
#  ### Analyse the waveform using YAMNet
#
#         analyze_audio_waveform(waveform, camera_name, interpreter, input_details, output_details)
#             Check waveform for compatibility with YAMNet interpreter, invoke the
#             intepreter, and return scores (a [1,521] array of scores, ordered per the
#             YAMNet class map CSV (files/yamnet_class_map.csv)
#
#  ### Ranking and Scoring Sounds
#
#         rank_sounds(scores, camera_name)
#             Use noise_threshold to toss out very low scores; take the top_k highest
#             scores, return a [2,521] array with pairs of class names (from class map CSV)
#             and scores.  Calls group_scores to group these class name/score pairs by
#             group, in turn calling calculate_composite_scores to create scores for these
#             groups. A modified yamnet_class_map.csv prepends each Yamnet display name
#             with a group name (people, music, birds, etc.) for this purpose.
#
#         group_scores_by_prefix(filtered_scores, class_names)
#             Organize filtered scores into groups according to the prefix of each class
#             name in (modified) files/yamnet_class_map.csv
#
#         calculate_composite_scores(group_scores_dict)
#             To report by group (vs. individual classes), take the individual scores from
#             each group (within the filtered scores) and use a simple algorithm to
#             score the group.  If any individual class score within the group is above 0.7,
#             that score will be used for the entire group.  Otherwise, take the highest
#             score within the group and add a confidence credit (0.05) for each individual
#             class within that group that made it through the filtering process.  Max 
#             composite score is 0.95 (unless the highest scoring class within the group 
#             is higher).
#
#  ### Sound Event Detection
#
#          update_sound_window(camera_name, detected_sounds)
#             Set up sliding window for detecting start/end sound events
#             
#          report_event(camera_name, sound_class, event_type, timestamp)
#
#
# ###  Misc
#
#          close_sound_log_file()
#             Make sure the sound_log CSV file is closed at exit
#
# yamcam_functions.py - Functions for yamcam3
#


import time
import os
import atexit
import csv
from datetime import datetime
import threading 
from collections import deque
import numpy as np
import json
import yamcam_config
from yamcam_config import (
        interpreter, input_details, output_details, logger,
        sound_log, sound_log_dir, shutdown_event
)

logger = yamcam_config.logger

#                                                #
### ---------- SOUND LOG CSV SETUP --------------###
#                                                #

sound_log_lock = threading.Lock()  # lock the file when writing since we have multiple threads writing

if sound_log:

    timestamp = datetime.now().strftime('%Y%m%d-%H%M') # timestamp for filename
    sound_log_path = os.path.join(sound_log_dir, f"{timestamp}.csv") # create the log file
    logger.debug(f"Creating {sound_log_path} for sound history analysis.")
    logger.info(f"Yamnet Sound Profiler logging sounds to {sound_log_path}. Press ^C to stop")

    try:
        sound_log_file = open(sound_log_path, 'a', newline='')
        sound_log_writer = csv.writer(sound_log_file)
        # write header row
        header = ["datetime", "sound_source", "group_name", "group_score", "class_name",
                  "class_score", "event_start", "event_end"]
        with sound_log_lock:
            sound_log_writer.writerow(header)
            sound_log_file.flush()

    except Exception as e:
        logger.warning(f"Could not create {sound_log_path}: {e}")
        sound_log_file = None
        sound_log_writer = None
else:
    sound_log_file = None
    sound_log_writer = None



     # -------- MAKE SURE WE CLOSE CSV AT EXIT

def close_sound_log_file():
    if sound_log_file is not None:
        sound_log_file.close()
        logger.debug("Sound log file closed.")

atexit.register(close_sound_log_file)

#                                                #
### ---------- REPORTING SETUP ----------------###
#                                                #

     # -------- GLOBALS FOR SUMMARY REPORTING
sound_event_tracker = {}
sound_event_lock = threading.Lock()
event_counts = {}

     # -------- DATA STRUCTS FOR EVENTS
# Decay counter data structure for detecting sound event termination
# Stores decay counters for each active sound class per camera
decay_counters = {}  # Structure: {camera_name: {sound_class: remaining_chunks}}

# State management for sound event detection
sound_windows = {}        # {camera_name: {sound_class: deque}}
active_sounds = {}        # {camera_name: {sound_class: bool}}
last_detection_time = {}  # {camera_name: {sound_class: timestamp}}

state_lock = threading.Lock()


     # -------- LOG SOUND EVENT
def report_event(camera_name, sound_class, event_type, timestamp):

    # CSV logging (events)
    if sound_log_writer is not None:
        log_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')  # Use current time for CSV log
        if event_type == 'start':   # column 7 is start
            row = [log_timestamp, camera_name, '', '', '', '', sound_class, '']
        else:                       # column 8 is end
            row = [log_timestamp, camera_name, '', '', '', '', '', sound_class]

        try:
            with sound_log_lock:
                sound_log_writer.writerow(row)
                sound_log_file.flush()
        except Exception as e:
            logger.error(f"Error writing to CSV: {e}", exc_info=True)

#                                                #
### ---------- SOUND FUNCTIONS ----------------###
#                                                #

     # -------- ANALYZE Waveform using YAMNet  
def analyze_audio_waveform(waveform, camera_name, interpreter, input_details, output_details):

    if shutdown_event.is_set():
        return None

    try:
        # Ensure waveform is a 1D array of float32 values between -1 and 1
        waveform = np.squeeze(waveform).astype(np.float32)
        if waveform.ndim != 1:
            logger.error(f"{camera_name}: Waveform must be a 1D array.")
            return None

        # Invoke the YAMNET inference engine 
        try:
            # Set input tensor and invoke interpreter
            interpreter.set_tensor(input_details[0]['index'], waveform)
            interpreter.invoke()

            # Get output scores; convert to a copy to avoid holding internal references
            scores = np.copy(interpreter.get_tensor(output_details[0]['index']))  

            if scores.size == 0:
                logger.warning(f"{camera_name}: No scores available to analyze.")
                return None

        except Exception as e:
            logger.error(f"{camera_name}: Error during interpreter invocation: {e}")
            return None

        return scores

    except Exception as e:
        logger.error(f"{camera_name}: Error during waveform analysis: {e}")
        return None


     # -------- Calculate, Group, and Filter Scores  

def rank_sounds(scores, camera_name):
    if shutdown_event.is_set():
        return []

    # Get config settings
    default_min_score = yamcam_config.default_min_score
    top_k = yamcam_config.top_k
    noise_threshold = yamcam_config.noise_threshold
    class_names = yamcam_config.class_names
    sounds_filters = yamcam_config.sounds_filters
    sounds_to_track = yamcam_config.sounds_to_track
    log_everything = yamcam_config.log_everything

    # Step 1: Filter scores based on noise threshold
    filtered_scores = [
        (i, score) for i, score in enumerate(scores[0]) if score >= noise_threshold
    ]

    # Track the number of classes that are not in sounds_to_track
    not_tracked_count = 0

    # Log each detected class, and count those not tracked
    for i, score in filtered_scores:
        class_name = class_names[i]
        group = class_name.split('.')[0]  # Extract the group prefix

        # Check if the group is in sounds_to_track
        is_tracked = group in sounds_to_track
        if not is_tracked:
            not_tracked_count += 1  # Increment for untracked classes

        # Log each class based on log_everything setting
        if log_everything or is_tracked:
            logger.debug(f"{camera_name}:--> {class_name}: {score:.2f}")

            # Log to CSV only for tracked groups or when log_everything is True
            if (log_everything or is_tracked) and sound_log_writer is not None:
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                row = [timestamp, camera_name, '', '', class_name, f"{score:.2f}", '', '']
                with sound_log_lock:
                    sound_log_writer.writerow(row)
                    sound_log_file.flush()

    # Report the number of found classes and those not tracked
    logger.debug(f"{camera_name}: {len(filtered_scores)} classes found"
                 f" ({not_tracked_count} not tracked)")

    if not filtered_scores:
        return []

    # Step 2: Group scores by prefix if we are applying filters
    if not log_everything:
        group_scores_dict = group_scores_by_prefix(filtered_scores, class_names)
        composite_scores = calculate_composite_scores(group_scores_dict)
        sorted_composite_scores = sorted(composite_scores, key=lambda x: x[1], reverse=True)
        limited_composite_scores = sorted_composite_scores[:top_k]

        # Step 3: Apply min_score filters for tracked groups
        results = []
        for group, score in limited_composite_scores:
            if group in sounds_to_track:
                min_score = sounds_filters.get(group, {}).get('min_score', default_min_score)
                if score >= min_score:
                    results.append({'class': group, 'score': score})

        return results

    # If log_everything is True, skip filtering and return all classes detected
    return [{'class': class_names[i].split('.')[0], 'score': score} for i, score in filtered_scores]



     # -------- Combine filtered class/score Pairs into Groups  
     # Group scores by prefix (e.g., 'music.*'), and keep track 
     # of the individual class scores.
def group_scores_by_prefix(filtered_scores, class_names):
    group_scores_dict = {}

    for i, score in filtered_scores:
        class_name = class_names[i]
        group = class_name.split('.')[0]  # Get the group prefix before the first period '.'

        if group not in group_scores_dict:
            group_scores_dict[group] = []

        group_scores_dict[group].append(score)

    return group_scores_dict


     # -------- Calculate Composite Scores for Groups 
     # Group scores by prefix (e.g., 'music.*'), and keep track of the individual class scores.
     # Algorithm to create a group score using the scores of the component classes from that group
     # - If max score in group is > 0.7, use this as the group composite score.
     # - Otherwise, boost score with credit based on number of group classes that were found:
     #   Max score + 0.05 * number of classes in the group (Cap Max score at 0.95).
def calculate_composite_scores(group_scores_dict):

    composite_scores = []

    for group, scores in group_scores_dict.items():
        max_score = max(scores)
        if max_score > 0.7:
            composite_score = max_score
        else:
            composite_score = min(max_score + 0.05 * len(scores), 0.95)

        composite_scores.append((group, composite_score))

    return composite_scores

     # -------- Manage Sound Event Window 
def update_sound_window(camera_name, detected_sounds):

    if shutdown_event.is_set():
        return

    current_time = time.time()
    with state_lock:

        # Initialize if not present
        if camera_name not in sound_windows:
            sound_windows[camera_name] = {}
            active_sounds[camera_name] = {}
            last_detection_time[camera_name] = {}
            decay_counters[camera_name] = {}  # Initialize decay_counters for the camera
            event_counts[camera_name] = {}    # Initialize event_counts for the camera

        window = sound_windows[camera_name]
        active = active_sounds[camera_name]
        last_time = last_detection_time[camera_name]
        decay_camera = decay_counters[camera_name]
        counts = event_counts[camera_name]

        for sound_class in yamcam_config.sounds_to_track:
            # Initialize deque for sound class
            if sound_class not in window:
                window[sound_class] = deque(maxlen=yamcam_config.window_detect)

            # Update detections
            is_detected = sound_class in detected_sounds
            window[sound_class].append(is_detected)

            # Update last detection time
            if is_detected:
                last_time[sound_class] = current_time

            # Check for start event
            if window[sound_class].count(True) >= yamcam_config.persistence:
                if not active.get(sound_class, False):
                    active[sound_class] = True
                    decay_camera[sound_class] = yamcam_config.decay
                    # Increment the event count for this sound_class
                    counts[sound_class] = counts.get(sound_class, 0) + 1
                    report_event(camera_name, sound_class, 'start', current_time)
                    if not shutdown_event.is_set():
                        logger.debug(f"{camera_name}: Sound '{sound_class}' started.")
            else:
                # Check for stop event using decay counters
                if active.get(sound_class, False):
                    if sound_class in detected_sounds:
                        # Reset decay counter if sound is detected
                        decay_camera[sound_class] = yamcam_config.decay
                    else:
                        # Decrement decay counter if sound is not detected
                        decay_camera[sound_class] -= 1
                        if decay_camera[sound_class] <= 0:
                            active[sound_class] = False
                            report_event(camera_name, sound_class, 'stop', current_time)
                            if not shutdown_event.is_set():
                                logger.debug(f"{camera_name}: Sound '{sound_class}' stopped.")
