# YAMNet-based Sound Profiler (YSP)

This project uses YAMNet to profile sounds heard by netwokred microphones,
producing a log file (csv format) that can be visualized using
[SoundViz](https://github.com/cecat/soundviz).

This code is a work in progress and has only been tested with MacOS.

Please report any issues
[here](https://github.com/cecat/CeC-HA-Addons/issues). 

This project uses TensorFlow and the
[YAMNet sound model](https://www.tensorflow.org/hub/tutorials/yamnet)
to characterize sounds deteced by  microphones on networked cameras.
*It does not record or keep any sound samples after analyzing them*. It
continually takes 0.975s samples from RTSP feeds, using FFMPEG, and
pushes these samples to the YAMNet sound classifier model, which
returns scores for each of its 521 sound classes.

The code uses a configuration file, *microphones.yaml*, to 
specify sound sourceas (RTSP feeds) and to set up a number
of options described below.
where the parameters for determining the start and stop of a sound event
are configurable.

The code does the following (*italics* parameters are configurable):

1. Analyze sound (in 0.975s chunks) using YAMNet, which produces scores for each
   of 521 sound classes. A **score** is YAMNet's certainty, from 0 to 1, that
   the sound class is present in the sound sample waveform.
2. Filter out all but the *top_k* sound classes - those whose scores
   exceed *noise_threshold*.
3. Aggregate those *top_k* scoring sound classes into groups such as "people," "music",
   "insects," or "birds." This uses a modified yamnet_class_map.csv where each
   of the 521 native YAMNet classes has been grouped and renamed as groupname.classname.
4. Assign a composite score to each group with classes that appear in the *top_k*,
   based on the individual scores of the classes from that group that are detected
   in the *top_k*. 
5. Detects the start and end of "sound events" defined by three configurable
   parameters (see configuration instructions below).
6. Logs detected sounds, groups, and sound events in a .csv file for analysis,
   e.g., using 
    [SoundViz](https://github.com/cecat/soundviz),
   whichvisualizes events over time for each sound source, the distribution of 
events by group, and the distribution of classes detected with each group.

### Modify the Configuration File (*microphones.yaml*)

Here you will configure specifics including and RTSP feeds and various 
parameters. Note that RTSP links have embedded credentials
(so treat this as a secrets file). 


```
# SAMPLE CONFIG FILE TO GET STARTED
# 
general:
  noise_threshold: 0.1       # Filter out very very low scores
  default_min_score: 0.5     # Default threshold for group scores (default 0.5)
                             # (must exceed this level for a sound event to be
                             # detected as starting or continuing)
  top_k: 10                  # Number of top scoring classes to analyze (default 10)
  log_level: INFO            # Default INFO. In order of decreasing verbosity:
                             # DEBUG->INFO->WARNING->ERROR->CRITICAL 
  log_everything: true       # Log classes/groups even if they are not specified in sounds/track below
  ffmpeg_debug: true         # Log ffmpeg stderr (a firehose - includes errors and info)
                             #   Must also have log_level set to DEBUG

# EVENTS (define 'event' parameters)
events:
  window_detect: 5           # Number of samples (~1s each) to examine to determine 
                             #   if a sound is persistent.
  persistence: 2             # Number of detections within window_detect to consider
                             #   a sound event has started.
  decay: 10                  # Number of waveforms without the sound to consider 
                             #   the sound event has stopped.


# SOURCES

cameras:
  myCam:                     # example of a UniFi NVR rtsp feed
    ffmpeg:
      inputs:
      - path: "rtsp://<IP_address>:7447/65dd5a1900f4cb70dffa2143_1"

# sound groups to listen for, and optional individual thresholds (will override
# reporting_threshold above in general settings). 
sounds:                     
  track:                    
    - people
    - birds
    - alert
  filters:
    people:
      min_score: 0.60
    birds:
      min_score: 0.70
    alert:
      min_score: 0.5

```

#### Configuration Parameters

**General**

- **noise_threshold**: Default 0.1 - Many sound classes will have very low scores, so we filter these 
out before processing the composite score for a sound group.
- **default_min_score**: Default 0.4 - When reporting to scores we ignore any groups with
scores below this value.
- **top_k**: YAMNet scores all 520 classes, we analyze the top_k highest scoring classes,
ignoring classes with scores below *noise_threshold*.
- **log_level**: Level of detail to be logged. Levels are
DEBUG->INFO->WARNING->ERROR->CRITICAL
in order of decreasing verbosity.
- **log_everything**: Log *all* classes/groups even if they are not specified in sounds/track below
- **ffmpeg_debug**: Logs all ffmpeg stderr messages, which have no codes nor does ffmpeg
differentiate between info and errors - so it's a firehose (coming from all n sources).

**Events**

- **window_detect**: Number of samples (~1s each) to examine to determine if a sound is persistent.
- **persistence**:   Number of detections within window_detect to consider a sound event has started.
- **decay**:         Number of waveforms without the sound to consider the sound event has stopped.

**Cameras**

- This is where you name your cameras (assuming they have microphones) and specify the RTSP path.
  It is always a good idea to make sure your paths are correct by opening them with an app like
  VLC Media Player.  For debugging you can also configure just with the problem camera and
  set log_level=DEBUG and ffmpeg_debug=true.

**Sounds**

Note that *Nothing will be reported if no sound
groups are listed here (unless you set log_everything to true).*
If no *min_score* is set for a group, the *default_min_score*
(General parameters above) is used.

Available sound groups are:
- aircraft
- alert (e.g., sirens, alarms, loud bangs...)
- animals
- birds
- boats
- construction (e.g., banging, sawing...)
- cooking
- domestic (e.g., vacuum cleaner, door squeek...)
- environment (e.g., wind, white noise, rustling leaves...)
- insects
- music
- people (e.g., laughter, coughing, speaking, ringtones, doorbells...)
- silence
- trains
- vehicles
- water
- weather
- yardcare (e.g., chainsaw, lawn mower)


More on sound groups below.

## Modified YAMNet Sound Class Scheme for Convenience Integrating with Home Assistant.

In the addon's directory is a *files* subdirectory, which contains the YAMNet *tflite* model
and a CSV file that maps YAMNet output codes to the human-readable names for those classes.
These are available at the
[TensorFlow hub](https://www.kaggle.com/models/google/yamnet/tfLite/classification-tflite/1?lite-format=tflite&tfhub-redirect=true).

The *yamnet_class_map.csv* used here is modified (without losing the original class names).
Yamnet has a whopping 521 sound classes, so if we are looking for human-related sounds there are
many related classes (e.g., *giggle, hiccup, laughter, shout, sigh,* not
to mention *speech, crowd, or sneeze*...). Similarly, if we want to detect that
music is playing there are over 150 related classes (*reggae, rimshot, rock and roll,
strum, tabla*...).

For this addon to be more useful it seemed reasonable 
to group these 521 classes, so that, for example, an automation can check for "people" rather than
all 60-80 classes related to people.  The grouping is implemented by using a modified version
of *yamnet_class_map.csv* that prepends a group name to each of the 521 sound class names.
For example, the classes *fiddle*, *snareDrum*, and *Middle Eastern Music*
are replaced with *music.fiddle, music.snareDrum* and * music.middleEasternMusic*;
and the classes *Tire Squeel*, *Motorcycle*, and
*Vehicle Horn Honking, Car Horn Honking* are replaced with 
*vehicles.tireSqueel*, *vehicles.motorcycle*, and
*vehicles.vehicleHornCarHornHonking*

Read more about groupings (and how you can customize)
[here](https://github.com/cecat/CeC-HA-Addons/tree/main/addons/yamcam3/files).

The code pulls the *top_k* classes with the highest scores (assuming there are 
at least *top_k* classes that exceed *noise_threshold*), then calculates a
group score from these.  The composite (i.e., group) score is calculated as follows:
- If the highest score (*max_score*) among those in the same group >= 0.7,
group_score = max_score
- Else group_score = max_score + 0.05(group_count), where group_count is the
number of classes from that group are in the top_k scores. 
- group_score is capped at 0.95.

The code will create a csv file in *./logs*. If this
directory does not exist, the code will create it.   The CSV format is 8 columns:

*(datetime), (camera), (group), (group_score), (class), (class_score), (group start), (group end)*

Each row records one of three items in additon to date/time and camera_name:
- a group name and score from among the *top_k*, 
- a class name and score from among the scores > *min_threshold*, or
- a group name for an event that started (group_name in column 7) or ended (group_name in column 8).

This sound log will give you a feel for what sounds and sound events are being detected
by each of your cameras. As importantly,
it will give you a sense for what kind of scores you are seeing, so that you can decide
how you wish to set both *default_min_score* and individual *min_score*s for the groups
you want to detect.


## Tech Info

- Languages & Frameworks:
  - Python
  - Home Assistant Framework


### Other Notes

This code has only been tested using RTSP feeds from Amcrest and UniFi (via NVR)
cameras. It's been tested on a MacBook with Apple M2 silicon running MacOS 15.1 (Sequoia).

