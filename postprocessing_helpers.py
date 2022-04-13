from collections import OrderedDict

##### Default Values ####
# GeneralValues
AudioFilename = "audio.mp3"
AudioLeadIn = "0"
PreviewTime = "0"
Countdown = "0"
SampleSet = "Normal"
StackLeniency = "0.7"
Mode = "1"
LetterboxInBreaks = "0"
WidescreenStoryboard = "0"
# EditorValues
DistanceSpacing = "1.8"
BeatDivisor = "4"
GridSize = "32"
TimelineZoom = "2"
# MetadataValues
Title = ""
TitleUnicode = ""
Artist = ""
ArtistUnicode = ""
Creator = ""
Version = "Normal"
Source = ""
Tags = ""
BeatmapID = "0"
BeatmapSetID = "0"
# DifficultyValues
HPDrainRate = "4.5" # 8,7,6,5 for kantan, futsuu, muzu, oni diffs respectively
CircleSize = "2"
OverallDifficulty = "4" # 2,3,4,5 for kantan, futsuu, muzu, oni diffs respectively
ApproachRate = "10"
SliderMultiplier = "1.4"
SliderTickRate = "1"
# TimingPoints
tp_time = ""
tp_beatLength = ""
tp_meter = "4"
tp_sampleSet = "0"
tp_sampleIndex = "0"
tp_volume = "100"
tp_uninherited = "1"
tp_effects = "0"
# HitObjects
ho_x = "256"
ho_y = "192"
ho_type = "1"
ho_objectParams = "0"
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
    'ho_objectParams': ho_objectParams,
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
    od = OrderedDict()
    od['AudioFilename'] = current_vals['AudioFilename']
    od['AudioLeadIn'] = current_vals['AudioLeadIn']
    od['PreviewTime'] = current_vals['PreviewTime']
    od['Countdown'] = current_vals['Countdown']
    od['SampleSet'] = current_vals['SampleSet']
    od['StackLeniency'] = current_vals['StackLeniency']
    od['Mode'] = current_vals['Mode']
    od['LetterboxInBreaks'] = current_vals['LetterboxInBreaks']
    od['WidescreenStoryboard'] = current_vals['WidescreenStoryboard']
    return od.items()

"""
Returns all key value pairs for Editor parameters.
"""
def get_editor_param():
    global current_vals
    od = OrderedDict()
    od['DistanceSpacing'] =  current_vals['DistanceSpacing']
    od['BeatDivisor'] = current_vals['BeatDivisor']
    od['GridSize'] = current_vals['GridSize']
    od['TimelineZoom'] = current_vals['TimelineZoom']
    return od.items()

"""
Returns all key value pairs for MetadataValues parameters.
"""
def get_metadata_param():
    global current_vals
    od = OrderedDict()
    od['Title'] =  current_vals['Title']
    od['TitleUnicode'] = current_vals['TitleUnicode']
    od['Artist'] = current_vals['Artist']
    od['ArtistUnicode'] = current_vals['ArtistUnicode']
    od['Creator'] = current_vals['Creator']
    od['Version'] = current_vals['Version']
    od['Source'] = current_vals['Source']
    od['Tags'] = current_vals['Tags']
    od['BeatmapID'] = current_vals['BeatmapID']
    od['BeatmapSetID'] = current_vals['BeatmapSetID']
    return od.items()

"""
Returns all key value pairs for MetadataValues parameters.
"""
def get_difficulty_param():
    global current_vals
    od = OrderedDict()
    od['HPDrainRate'] = current_vals['HPDrainRate']
    od['CircleSize'] = current_vals['CircleSize']
    od['OverallDifficulty'] = current_vals['OverallDifficulty']
    od['ApproachRate'] = current_vals['ApproachRate']
    od['SliderMultiplier'] = current_vals['SliderMultiplier']
    od['SliderTickRate'] = current_vals['SliderTickRate']
    return od.items()

"""
Returns all key value pairs for TimingPoints paramters
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
        current_vals['ho_objectParams'],
        current_vals['ho_hitSample']
    ]
    return ret

"""
Returns event values
"""
def get_event_values(): 
    global current_vals
    return [current_vals['EventString']]