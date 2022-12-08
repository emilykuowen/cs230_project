# trying for acoustic guitar

# Objective: 
# Reorganizing MedleyDB dataset into file structure expected by nussl's MixSourceFolder dataset
# note: change instrument_label parameter based on which instrument you want to separate
# this will get the dataset ready for source separation for that instrument

# Breakdown:
# 0. make baseline file directory as expected by Nussl's MixSourceFolder
# 1. look through all tracks in MedleyDB_sample/ Audio/ file
# 2. locate and read through _METADATA.yaml file for each track
# 3. identify if it contains 'acoustic guitar' as an 'instrument'
# 4. if it does, keep track name and put appropriate .wavs into directory expected by Nussl's MixSourceFolder (/mix and /acoustic guitar)
# 5. make 'other' track via mix - guitar or summing up all other sources stems that aren't guitar. put in /other

# imports
import os
import os.path
from os import path
import yaml
from pathlib import Path
#import shutil
import numpy as np
#import wavio as wv
import nussl
#import soundfile
#import wave
#from scipy.io import wavfile
#from scipy.io.wavfile import write
#from pydub import AudioSegment, effects

# track directory
dir_in_str = 'medleydb/tracks/'
# .yaml directory
ydir = 'medleydb/metadata/'

# change label depending on which instrument we want to separate
instrument_label = 'acoustic guitar' 

# prepare new directory structure
new_dir = './mix_source_folder' + '_' + instrument_label
new_mix_path = os.path.join(new_dir, 'mix')
new_instr_path = os.path.join(new_dir, instrument_label)
new_other_path = os.path.join(new_dir, 'other')
if not os.path.exists(new_dir):
    os.mkdir(new_dir)
    os.mkdir(new_mix_path)
    os.mkdir(new_instr_path)
    os.mkdir(new_other_path)

# parse through yaml filepath
# loop through all yaml files
# access yaml file
# convert to dict and search for instrument source
for yfilename in os.listdir(ydir):
    if yfilename != '.DS_Store' and yfilename != 'AimeeNorwich_Child_METADATA.yaml':
        print(yfilename)
        # looping through all yaml files
        f = os.path.join(ydir, yfilename)
        if os.path.isfile(f):
            yfile = open(f, 'r')
            # creates dictionary of .yaml file contents
            ydict = yaml.load(yfile, Loader = yaml.Loader)
            # create and look through 'stems' dict
            if 'stems' in ydict:
                sdict = ydict['stems']
                # stem = S0# (stem number), slabel = stem key labels (we care about 'instrument')
                stem_nos = []
                trackname = ''
                for stem, slabel in sdict.items():
                    for key in slabel:
                        if slabel[key] == instrument_label:
                            # trackname given by yaml file name
                            trackname = yfilename.partition('_METADATA.')[0]
                            # also need stem number in yaml
                            stem_nos.append(stem[-1])
                            # make path in medleydb folder with trackname
                print(len(stem_nos))
                # make sure track folder exists in medleydb
                f = os.path.join(dir_in_str, trackname)
                if path.exists(f):
                    #print('track exists for .yaml')
                    # special case, in which multiple stems have same instrument label
                    if (len(stem_nos) > 1): 
                        print('len > 1')
                        # initialize list to store the multiple paths that has the instrument stems
                        instr_paths = []
                        # iterate through stem_nos
                        for s in range(len(stem_nos)):
                            instr_paths.append(os.path.join(dir_in_str, trackname, trackname + '_STEMS', trackname + '_STEM_0' + stem_nos[s] + '.wav'))

                        mix_path = os.path.join(dir_in_str, trackname, trackname + '_MIX.wav')
      
                        #fs2, data2 = wavfile.read(mix_path)
                        mix_data = nussl.AudioSignal(mix_path)

                        # get individual stem .wav files and store in data1 list 
                        sdata = []
                        for ss in instr_paths:
                            data = nussl.AudioSignal(ss).audio_data
                            sdata.append(data)
                        # add all stems audio arrays corresponding to instrument together
                        stems_data = [sum(x) for x in zip(*sdata)] # numpy array
                        # convert to audio signal object
                        sums_stems_data = nussl.AudioSignal(audio_data_array=np.array(stems_data), sample_rate=44100) 
                        # normalize to avoid clipping
                        sums_stems_data = sums_stems_data.peak_normalize() 
                        # write to new instr dir
                        sums_stems_data.write_audio_to_file(os.path.join(new_instr_path, trackname + '.wav')) 

                        # calculate 'other' .wav file
                        other_data = mix_data - sums_stems_data
                        # normalize to avoid clipping
                        other_data = other_data.peak_normalize() 
                        # write to new other dir
                        other_data.write_audio_to_file(os.path.join(new_other_path, trackname + '.wav')) 
                        
                        # write 'mix' .wav file into new dir 
                        mix_data.write_audio_to_file(os.path.join(new_mix_path, trackname+ '.wav'))

                    # usual case where we only have one stem file corresponding to instrument
                    elif (len(stem_nos) == 1):
                        print('len == 1')
                        instr_path = os.path.join(dir_in_str, trackname, trackname + '_STEMS', trackname + '_STEM_0' + stem_nos[0] + '.wav')
                        mix_path = os.path.join(dir_in_str, trackname, trackname + '_MIX.wav')

                        # get audio data for instr
                        stem_data = nussl.AudioSignal(instr_path)
                        # get audio data for mix
                        mix_data = nussl.AudioSignal(mix_path)

                        # calculate 'other' .wav file
                        other_data = mix_data-stem_data
                        # normalize to avoid clipping
                        other_data = other_data.peak_normalize()
                        # write to new other dir
                        other_data.write_audio_to_file(os.path.join(new_other_path, trackname + '.wav'))

                        # write to new instr dir
                        stem_data.write_audio_to_file(os.path.join(new_instr_path, trackname + '.wav'))
                        # write to new mix dir
                        mix_data.write_audio_to_file(os.path.join(new_mix_path, trackname + '.wav'))