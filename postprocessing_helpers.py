from collections import OrderedDict

"""These are the default field names that are written to the .osu file."""
# GeneralValues
AudioFilename = "audio.mp3"
AudioLeadIn = "0"
PreviewTime = "-1"
Countdown = "0"
SampleSet = "Normal"
StackLeniency = "0"
Mode = "1"
LetterboxInBreaks = "0"
WidescreenStoryboard = "0"
# EditorValues
DistanceSpacing = "0.1"
BeatDivisor = "4"
GridSize = "32"
TimelineZoom = "2"
# MetadataValues
Title = ""
TitleUnicode = ""
Artist = ""
ArtistUnicode = ""
Creator = "TaikoMapper"
Version = "Taiko"
Source = ""
Tags = ""
BeatmapID = "0"
BeatmapSetID = "-1"
# DifficultyValues
HPDrainRate = "8" # Recommended: 8,7,6,5 for kantan, futsuu, muzu, oni diffs respectively
CircleSize = "2"
OverallDifficulty = "2" # Recommended: 2,3,4,5 for kantan, futsuu, muzu, oni diffs respectively
ApproachRate = "10"
SliderMultiplier = "1.4"
SliderTickRate = "1"
# TimingPoints
tp_time = ""
tp_beatLength = ""
tp_meter = "4"
tp_sampleSet = "1"
tp_sampleIndex = "0"
tp_volume = "50"
tp_uninherited = "1"
tp_effects = "0"
# HitObjects
ho_x = "256"
ho_y = "192"
ho_type = "1"
ho_hitSample = "0:0:0:0:"

# EventsValues
EventString = "////Background and Video events\n////Break Periods\n////Storyboard Layer 0 (Background)\n////Storyboard Layer 1 (Fail)\n////Storyboard Layer 2 (Pass)\n////Storyboard Layer 3 (Foreground)\n////Storyboard Layer 4 (Overlay)\n////Storyboard Sound Samples"

# Global Dict holding the current set values
current_vals = {
    'AudioFilename': AudioFilename,
    'AudioLeadIn': AudioLeadIn,
    'PreviewTime': PreviewTime,
    'Countdown': Countdown,
    'SampleSet': SampleSet,
    'StackLeniency': StackLeniency,
    'Mode': Mode,
    'LetterboxInBreaks': LetterboxInBreaks,
    'WidescreenStoryboard': WidescreenStoryboard,
    'DistanceSpacing': DistanceSpacing,
    'BeatDivisor': BeatDivisor,
    'GridSize': GridSize,
    'TimelineZoom': TimelineZoom,
    'Title': Title,
    'TitleUnicode': TitleUnicode,
    'Artist': Artist,
    'ArtistUnicode': ArtistUnicode,
    'Creator': Creator,
    'Version': Version,
    'Source': Source,
    'Tags': Tags,
    'BeatmapID': BeatmapID,
    'BeatmapSetID': BeatmapSetID,
    'HPDrainRate': HPDrainRate,
    'CircleSize': CircleSize,
    'OverallDifficulty': OverallDifficulty,
    'ApproachRate': ApproachRate,
    'SliderMultiplier': SliderMultiplier,
    'SliderTickRate': SliderTickRate,
    'EventString': EventString,
    'tp_time': tp_time,
    'tp_beatLength': tp_beatLength,
    'tp_meter': tp_meter,
    'tp_sampleSet': tp_sampleSet,
    'tp_sampleIndex': tp_sampleIndex,
    'tp_volume': tp_volume,
    'tp_uninherited': tp_uninherited,
    'tp_effects': tp_effects, 
    'ho_x': ho_x,
    'ho_y': ho_y,
    'ho_type': ho_type,
    'ho_hitSample': ho_hitSample,   
}

"""
Sets current parameters for song data, overwriting default data.

Parameters:
    pairs (list of tuple): A list containing all the changes to be made to the parameters
"""
def set_values(pairs):
    global current_vals
    for key,value in pairs:
        current_vals[key] = value
    
"""
Returns all key value pairs for general parameters.
"""
def get_general_param():
    global current_vals
    d = {}
    d['AudioFilename'] = current_vals['AudioFilename']
    d['AudioLeadIn'] = current_vals['AudioLeadIn']
    d['PreviewTime'] = current_vals['PreviewTime']
    d['Countdown'] = current_vals['Countdown']
    d['SampleSet'] = current_vals['SampleSet']
    d['StackLeniency'] = current_vals['StackLeniency']
    d['Mode'] = current_vals['Mode']
    d['LetterboxInBreaks'] = current_vals['LetterboxInBreaks']
    d['WidescreenStoryboard'] = current_vals['WidescreenStoryboard']
    return d

"""
Returns all key value pairs for Editor parameters.
"""
def get_editor_param():
    global current_vals
    d = {}
    d['DistanceSpacing'] =  current_vals['DistanceSpacing']
    d['BeatDivisor'] = current_vals['BeatDivisor']
    d['GridSize'] = current_vals['GridSize']
    d['TimelineZoom'] = current_vals['TimelineZoom']
    return d

"""
Returns all key value pairs for MetadataValues parameters.
"""
def get_metadata_param():
    global current_vals
    d = {}
    d['Title'] =  current_vals['Title']
    d['TitleUnicode'] = current_vals['TitleUnicode']
    d['Artist'] = current_vals['Artist']
    d['ArtistUnicode'] = current_vals['ArtistUnicode']
    d['Creator'] = current_vals['Creator']
    d['Version'] = current_vals['Version']
    d['Source'] = current_vals['Source']
    d['Tags'] = current_vals['Tags']
    d['BeatmapID'] = current_vals['BeatmapID']
    d['BeatmapSetID'] = current_vals['BeatmapSetID']
    return d

"""
Returns all key value pairs for MetadataValues parameters.
"""
def get_difficulty_param():
    global current_vals
    d = {}
    d['HPDrainRate'] = current_vals['HPDrainRate']
    d['CircleSize'] = current_vals['CircleSize']
    d['OverallDifficulty'] = current_vals['OverallDifficulty']
    d['ApproachRate'] = current_vals['ApproachRate']
    d['SliderMultiplier'] = current_vals['SliderMultiplier']
    d['SliderTickRate'] = current_vals['SliderTickRate']
    return d

"""
Returns all key value pairs for TimingPoints parameters.
"""
def get_timing_point_param():
    global current_vals
    ret = [
        current_vals['tp_time'],
        current_vals['tp_beatLength'],
        current_vals['tp_meter'],
        current_vals['tp_sampleSet'],
        current_vals['tp_sampleIndex'],
        current_vals['tp_volume'],
        current_vals['tp_uninherited'],
        current_vals['tp_effects']
    ]
    return ret
    
"""
Returns all key value pairs for HitObjects parameters.
"""
def get_hit_objects_param():
    global current_vals
    ret = [
        current_vals['ho_x'],
        current_vals['ho_y'],
        current_vals['ho_type'],
        current_vals['ho_hitSample']
    ]
    return ret

"""
Returns event values.
"""
def get_event_values(): 
    global current_vals
    return [current_vals['EventString']]