#
# yamcam-CLI - CeC - November 2024
#
# yamcam_config.py 
#

import yaml
import csv
import logging
import tensorflow as tf
import time
import threading
import os
import sys
from datetime import datetime

# File paths

config_path = './microphones.yaml'
class_map_path = './files/yamnet_class_map.csv'
model_path = './files/yamnet.tflite'
log_dir = './logs'
sound_log_dir = './logs'

# Global shutdown event
shutdown_event = threading.Event()

#                                              #
### --------------- FUNCTIONS ---------------###
#                                              #
# 
#    check_for_log_dir()
#          Make sure the log directory exists
#    shutdownFilter(logging.Filter) and filter(self, record)
#          Smooth shutdown, minimize dangling log msgs
#    validate_camera_config(camera_settings)
#          Check for common errors in RTSP path spec`
#    format_input_details(details)
#          For debug, nicely format specs for what the model expects
#    validate_boolean(var_name, var_value)
#          Check that booleans are all correctly set, default to False
#

# -------- MAKE SURE THE LOG DIRECTORY EXISTS

def check_for_log_dir():
    try:
        print(f"Creating log file directory {sound_log_dir}")
        os.makedirs(sound_log_dir, exist_ok=True)
    except OSError as e:
        # Use print since logging is not configured yet
        print(f"Error: Failed to create logging directory '{sound_log_dir}': {e}")
        print(f"STOPPING the add-on. Use *Terminal* or SSH CLI to manually create {sound_log_dir} "
               "or set sound_log to false")

        sys.exit(1)  # Exit with a non-zero code to indicate failure


# -------- SHUT_DOWN HANDLING 

class ShutdownFilter(logging.Filter):
    def filter(self, record):
        return not shutdown_event.is_set()
    
# -------- VALIDATE CAMERA CONFIGURATION

def validate_camera_config(camera_settings):
    for camera_name, camera_config in camera_settings.items():
        ffmpeg_config = camera_config.get('ffmpeg')
        if not ffmpeg_config or not isinstance(ffmpeg_config, dict):
            raise ValueError(f"STOPPING. Camera '{camera_name}': "
                              "'ffmpeg' section is missing or invalid.")

        inputs = ffmpeg_config.get('inputs')
        if not inputs or not isinstance(inputs, list) or len(inputs) == 0:
            raise ValueError(f"Camera '{camera_name}': "
                              "'inputs' section is missing or invalid.")

        rtsp_url = inputs[0].get('path')
        if not rtsp_url or not isinstance(rtsp_url, str):
            raise ValueError(f"Camera '{camera_name}': "
                              "RTSP path is missing or invalid.")

# -------- LOG DETAILS FOR DEBUG

def format_input_details(details):
    formatted_details = "Input Details:\n"
    for detail in details:
        formatted_details += "  -\n"
        for key, value in detail.items():
            formatted_details += f"    {key}: {value}\n"
    return formatted_details

# -------- VALIDATE BOOLEAN PARAMETERS

def validate_boolean(var_name, var_value):
    if isinstance(var_value, str):
        var_lower = var_value.lower()
        if var_lower == "true":
            return True
        elif var_lower == "false":
            return False
        else:
            logger.warning(f"Invalid boolean value '{var_value}' "
                           f"for {var_name}. Defaulting to False.")
            return False
    elif isinstance(var_value, bool):
        return var_value
    else:
        logger.warning(f"Invalid type '{type(var_value).__name__}' "
                       f"for boolean {var_name}. Defaulting to False.")
        return False

#                                              #
### --------------- STARTUP ---------------###
#                                              #

# -------- SET INITIAL LOGGING FORMAT

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# -------- ASSIGN HANDLERS

for handler in logger.handlers:
    handler.addFilter(ShutdownFilter())

logger.debug("\n\n-------- YAMCAM3 Started-------- \n")

# -------- OPEN YAML CONFIG FILE

try:
    with open(config_path) as f:
        config = yaml.safe_load(f)
except yaml.YAMLError as e:
    logger.error(f"Error reading YAML file {config_path}: {e}")
    raise

# -------- GENERAL SETTINGS

try:
    general_settings = config['general']
except KeyError as e:
    logger.error(f"Missing general settings in the configuration file: {e}")
    raise

log_level            = general_settings.get('log_level', 'INFO').upper()
logfile              = general_settings.get('logfile', False)
sound_log            = general_settings.get('sound_log', False)
log_everything       = general_settings.get('log_everything', False)
ffmpeg_debug         = general_settings.get('ffmpeg_debug', False)
default_min_score    = general_settings.get('default_min_score', 0.5)
noise_threshold      = general_settings.get('noise_threshold', 0.1)   
top_k                = general_settings.get('top_k', 10)

# --------- VERIFY GENERAL SETTINGS

logfile = validate_boolean("logfile", logfile)
sound_log = validate_boolean("sound_log", sound_log)
ffmpeg_debug = validate_boolean("ffmpeg_debug", ffmpeg_debug)
log_everything = validate_boolean("log_everything", log_everything)



# DEFAULT_MIN_SCORE must be between 0 and 1
if not (0.0 <= default_min_score <= 1.0):
    logger.warning(f"Invalid default_min_score '{default_min_score}'"
                    "Should be between 0.0 and 1.0. Defaulting to 0.5."
    )
    default_min_score = 0.5

# NOISE_THRESHOLD must be between 0 and 1
if not (0.0 <= noise_threshold <= 1.0):
    logger.warning(f"Invalid noise_threshold '{noise_threshold}'"
                    "Should be between 0.0 and 1.0. Defaulting to 0.1."
    )
    noise_threshold = 0.1

# TOP_K cannot exceed 521 (more than about 20 is silly)
if not (1 <= top_k <= 20):
    logger.warning(f"Invalid top_k '{top_k}'"
                    "Should be between 1 and 20. Defaulting to 10."
    )
    top_k = 10
        

# -------- SET UP LOGGING TO FILE FOR DEBUG ANALYSIS if logfile=True:

check_for_log_dir() # make sure /media/yamcam exists

if logfile:
    timestamp = datetime.now().strftime('%Y%m%d-%H%M') # timestamp for filename
    log_path = os.path.join(log_dir, f"{timestamp}.log")
    logger.debug(f"Creating {log_path} for longitudinal sound analysis.")

    if log_everything:
        logger.debug("Logging all classes and groups (log_everything = True)")
    try:
        file_handler = logging.FileHandler(log_path, mode='a')  # always append
        file_handler.setLevel(logging.DEBUG)  # hard coding logfile to DEBUG
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler) # Add the file handler to the logger
    except Exception as e:
        logger.error(f"Could not create or open the log file at {log_path}: {e}")


# -------- SOUND EVENT PARAMETERS 

try:
    events_settings = config['events']
except KeyError:
    logger.warning("Missing events settings in the configuration file. Using default values.")
    events_settings = {
        'window_detect': 5,  # Default value
        'persistence': 3,    # Default value
        'decay': 15          # Default value
    }

window_detect = events_settings.get('window_detect', 5)
persistence = events_settings.get('persistence', 3)
decay = events_settings.get('decay', 15)

# -------- SOUND GROUPS TO WATCH; MIN_SCORES (optional)

try:
    sounds = config['sounds']
except KeyError:
    logger.warning("Missing sounds settings in the configuration file. Using default values.")
    sounds = {} # in case none are configured.

sounds_to_track = sounds.get('track', [])
sounds_filters = sounds.get('filters', {})

# min_score values also need to be between 0 and 1
for group, settings in sounds_filters.items():
    min_score = settings.get('min_score')
    if not (0.0 <= min_score <= 1.0):
        logger.warning(f"Invalid min_score '{min_score}' for group '{group}'."
                        "Should be between 0.0 and 1.0. Defaulting to default_min_score."
        )
        settings['min_score'] = default_min_score 

# -------- CAMS (SOUND SOURCES)

try:
    camera_settings = config['cameras']
    validate_camera_config(camera_settings)
except KeyError as e:
    logger.error(f"Missing camera settings in the configuration file: {e}")
    sys.exit(1)
except ValueError as e:
    logger.error(f"Configuration error: {e}")
    sys.exit(1)

# -------- LOG LEVEL

log_levels = {
    'DEBUG'    : logging.DEBUG,
    'INFO'     : logging.INFO,
    'WARNING'  : logging.WARNING,
    'ERROR'    : logging.ERROR,
    'CRITICAL' : logging.CRITICAL
}
if log_level in log_levels:
    logger.setLevel(log_levels[log_level])
    for handler in logger.handlers: #add for file log to stay at DEBUG
        if not isinstance(handler, logging.FileHandler):  # Skip file handler
            handler.setLevel(log_levels[log_level])
    logger.debug(f"Logging level: {log_level}")
else:
    logger.warning(f"Invalid log level {log_level}; Defaulting to INFO.")
    logger.setLevel(logging.INFO)
    for handler in logger.handlers:
        handler.setLevel(logging.INFO)

#                                              #
### ---------- SET UP YAMNET MODEL ----------###
#                                              #

# -------- LOAD MODEL (using TensorFLow Lite)

logger.debug("Loading YAMNet model")
interpreter    = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
logger.debug("YAMNet model loaded.")
logger.debug(format_input_details(input_details))


# -------- BUILD CLASS NAMES DICTIONARY

class_names = []
with open(class_map_path, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Skip the header
    for row in reader:
        class_names.append(row[2].strip('"'))

