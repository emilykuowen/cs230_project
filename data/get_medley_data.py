import nussl
from pathlib import Path
import os
import glob
import scaper
import numpy as np
import matplotlib.pyplot as plt
import warnings
import shutil

""" def download_musdb18(fg_folder, bg_folder):
    # Run this command to download X 7-second clips from MUSDB18
    musdb = nussl.datasets.MUSDB18(download=True)
    musdb_train = nussl.datasets.MUSDB18(subsets=['train'])
    print(musdb_train.items)

    idx = 0
    item = musdb_train[idx]
    print(item.keys())
    print(item['metadata'])
    print(item['sources'].keys())

    # create foreground folder
    fg_folder = Path(fg_folder).expanduser()  
    fg_folder.mkdir(parents=True, exist_ok=True)                             

    # create background folder - we need to provide one even if we don't use it
    bg_folder = Path(bg_folder).expanduser()
    bg_folder.mkdir(parents=True, exist_ok=True)

    # For each item (track) in the train set, iterate over its sources (stems),
    # create a folder for the stem if it doesn't exist already (drums, bass, vocals, other) 
    # and place the stem audio file in this folder, using the song name as the filename
    for item in musdb_train:
        song_name = item['mix'].file_name
        for key, val in item['sources'].items():
            src_path = fg_folder / key 
            src_path.mkdir(exist_ok=True)
            src_path = str(src_path / song_name) + '.wav'
            val.write_audio_to_file(src_path)
    
    for folder in os.listdir(fg_folder):
        if folder[0] != '.':  # ignore system folders
            stem_files = os.listdir(os.path.join(fg_folder, folder))
            print(f"\n{folder}\tfolder contains {len(stem_files)} audio files:\n")
            for sf in sorted(stem_files)[:5]:
                print(f"\t\t{sf}")
            print("\t\t...")

    return musdb """

def get_medley(fg_folder, bg_folder):
    #ms_folder = '~/data/medleydb_' + 'acoustic_guitar'
    ms_folder = '../../data/medleydb_' + 'acoustic_guitar'
    #ms_folder = './mix_source_folder' + '_' + 'acoustic_guitar'
    # create MixSourceFolder-style dataset 
    medley = nussl.datasets.MixSourceFolder(ms_folder)
    #fg_folder = '~/data/medleydb_' + 'acoustic_guitar'
    #bg_folder = '~/data/bg_' + 'acoustic_guitar'
    # create foreground folder
    # fg_folder = Path(fg_folder).expanduser()  
    # fg_folder.mkdir(parents=True, exist_ok=True) 
    #dir_mix = os.path.join(ms_folder, 'mix')                            
    #for f in os.listdir(dir_mix):
        #os.remove(os.path.join(dir_mix,f))
    #os.rmdir(dir_mix)
    # create background folder - we need to provide one even if we don't use it
    #bg_folder = Path(bg_folder).expanduser()
    bg_folder = '~/bg'
    bg_folder.mkdir(parents=True, exist_ok=True)

    return medley


def incoherent(fg_folder, bg_folder, event_template, seed):
    """
    This function takes the paths to the MUSDB18 source materials, an event template, 
    and a random seed, and returns an INCOHERENT mixture (audio + annotations). 
    
    Stems in INCOHERENT mixtures may come from different songs and are not temporally
    aligned.
    
    Parameters
    ----------
    fg_folder : str
        Path to the foreground source material for MUSDB18
    bg_folder : str
        Path to the background material for MUSDB18 (empty folder)
    event_template: dict
        Dictionary containing a template of probabilistic event parameters
    seed : int or np.random.RandomState()
        Seed for setting the Scaper object's random state. Different seeds will 
        generate different mixtures for the same source material and event template.
        
    Returns
    -------
    mixture_audio : np.ndarray
        Audio signal for the mixture
    mixture_jams : np.ndarray
        JAMS annotation for the mixture
    annotation_list : list
        Simple annotation in list format
    stem_audio_list : list
        List containing the audio signals of the stems that comprise the mixture
    """
    
    # Create scaper object and seed random state
    sc = scaper.Scaper(
        duration=5.0,
        fg_path=str(fg_folder),
        bg_path=str(bg_folder),
        random_state=seed
    )
    
    # Set sample rate, reference dB, and channels (mono)
    sc.sr = 44100
    sc.ref_db = -20
    sc.n_channels = 1
    
    # Copy the template so we can change it
    event_parameters = event_template.copy()
    
    # Iterate over stem types and add INCOHERENT events
    # labels = ['vocals', 'drums', 'bass', 'other']
    labels = ['acoustic guitar', 'other']
    for label in labels:
        event_parameters['label'] = ('const', label)
        sc.add_event(**event_parameters)
    
    # Return the generated mixture audio + annotations 
    # while ensuring we prevent audio clipping
    return sc.generate(fix_clipping=True)


def coherent(fg_folder, bg_folder, event_template, seed):
    """
    This function takes the paths to the MUSDB18 source materials and a random seed,
    and returns an COHERENT mixture (audio + annotations).
    
    Stems in COHERENT mixtures come from the same song and are temporally aligned.
    
    Parameters
    ----------
    fg_folder : str
        Path to the foreground source material for MUSDB18
    bg_folder : str
        Path to the background material for MUSDB18 (empty folder)
    event_template: dict
        Dictionary containing a template of probabilistic event parameters
    seed : int or np.random.RandomState()
        Seed for setting the Scaper object's random state. Different seeds will 
        generate different mixtures for the same source material and event template.
        
    Returns
    -------
    mixture_audio : np.ndarray
        Audio signal for the mixture
    mixture_jams : np.ndarray
        JAMS annotation for the mixture
    annotation_list : list
        Simple annotation in list format
    stem_audio_list : list
        List containing the audio signals of the stems that comprise the mixture
    """
        
    # Create scaper object and seed random state
    sc = scaper.Scaper(
        duration=5.0,
        fg_path=str(fg_folder),
        bg_path=str(bg_folder),
        random_state=seed
    )
    
    # Set sample rate, reference dB, and channels (mono)
    sc.sr = 44100
    sc.ref_db = -20
    sc.n_channels = 1
    
    # Copy the template so we can change it
    event_parameters = event_template.copy()    
    
    # Instatiate the template once to randomly choose a song,   
    # a start time for the sources, a pitch shift and a time    
    # stretch. These values must remain COHERENT across all stems
    sc.add_event(**event_parameters)
    event = sc._instantiate_event(sc.fg_spec[0])
    
    # Reset the Scaper object's the event specification
    sc.reset_fg_event_spec()
    
    # Replace the distributions for source time, pitch shift and 
    # time stretch with the constant values we just sampled, to  
    # ensure our added events (stems) are coherent.              
    event_parameters['source_time'] = ('const', event.source_time)
    event_parameters['pitch_shift'] = ('const', event.pitch_shift)
    event_parameters['time_stretch'] = ('const', event.time_stretch)

    # Iterate over the four stems (vocals, drums, bass, other) and 
    # add COHERENT events.                                         
    # labels = ['vocals', 'drums', 'bass', 'other']
    labels = ['acoustic guitar', 'other']
    for label in labels:
        
        # Set the label to the stem we are adding
        event_parameters['label'] = ('const', label)
        
        # To ensure coherent source files (all from the same song), we leverage
        # the fact that all the stems from the same song have the same filename.
        # All we have to do is replace the stem file's parent folder name from "vocals" 
        # to the label we are adding in this iteration of the loop, which will give the 
        # correct path to the stem source file for this current label.
        coherent_source_file = event.source_file.replace('acoustic guitar', label)
        event_parameters['source_file'] = ('const', coherent_source_file)
        # Add the event using the modified, COHERENT, event parameters
        sc.add_event(**event_parameters)
    
    # Generate and return the mixture audio, stem audio, and annotations
    return sc.generate(fix_clipping=True)


def generate_mixture(dataset, fg_folder, bg_folder, event_template, seed):
    
    # hide warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        
        # flip a coint to choose coherent or incoherent mixing
        random_state = np.random.RandomState(seed)
        
        # generate mixture
        if random_state.rand() > .5:
            data = coherent(fg_folder, bg_folder, event_template, seed)
        else:
            data = incoherent(fg_folder, bg_folder, event_template, seed)
            
    # unpack the data
    mixture_audio, mixture_jam, annotation_list, stem_audio_list = data
    
    # convert mixture to nussl format
    mix = dataset._load_audio_from_array(
        audio_data=mixture_audio, sample_rate=dataset.sample_rate
    )
    
    # convert stems to nussl format
    sources = {}
    ann = mixture_jam.annotations.search(namespace='scaper')[0]
    for obs, stem_audio in zip(ann.data, stem_audio_list):
        key = obs.value['label']
        sources[key] = dataset._load_audio_from_array(
            audio_data=stem_audio, sample_rate=dataset.sample_rate
        )
    
    # store the mixture, stems and JAMS annotation in the format expected by nussl
    output = {
        'mix': mix,
        'sources': sources,
        'metadata': mixture_jam
    }
    return output

# Convenience class so we don't need to enter the fg_folder, bg_folder, and template each time
class MixClosure:
    
    def __init__(self, fg_folder, bg_folder, event_template):
        self.fg_folder = fg_folder
        self.bg_folder = bg_folder
        self.event_template = event_template
        
    def __call__(self, dataset, seed):
        return generate_mixture(dataset, self.fg_folder, self.bg_folder, self.event_template, seed)


if __name__ == "__main__":
    # only need to run download_musdb18() once
    # fg_folder = '~/.nussl/ismir2020-tutorial/foreground'
    # bg_folder = '~/.nussl/ismir2020-tutorial/background'
    fg_folder = '~/data/medleydb_' + 'acoustic_guitar'
    bg_folder = '~/data' + '/bg'

    # musdb = download_musdb18(fg_folder, bg_folder)
    medley = get_medley(fg_folder, bg_folder)
    # Create a template of probabilistic event parameters
    template_event_parameters = {
        'label': ('const', 'vocals'),
        'source_file': ('choose', []),
        'source_time': ('uniform', 0, 7),
        'event_time': ('const', 0),
        'event_duration': ('const', 5.0),
        'snr': ('uniform', -5, 5),
        'pitch_shift': ('uniform', -2, 2),
        'time_stretch': ('uniform', 0.8, 1.2)
    }

    # # Generate 3 coherent mixtures
    # for seed in [1, 2, 3]:
    #     mixture_audio, mixture_jam, annotation_list, stem_audio_list = coherent(
    #         fg_folder, 
    #         bg_folder, 
    #         template_event_parameters, 
    #         seed)
    
    # # Generate 3 incoherent mixtures
    # for seed in [1, 2, 3]:
    #     mixture_audio, mixture_jam, annotation_list, stem_audio_list = incoherent(
    #         fg_folder, 
    #         bg_folder, 
    #         template_event_parameters, 
    #         seed)

    # Initialize our mixing function with our specific source material and event template
    mix_func = MixClosure(fg_folder, bg_folder, template_event_parameters)

    # Create a nussle OnTheFly data generator
    on_the_fly = nussl.datasets.OnTheFly(
        num_mixtures=1000,
        mix_closure=mix_func
    )

    for i in range(3):
        item = on_the_fly[i]
        mix = item['mix']
        print(mix)
        mix.write_audio_to_file('./gen_mix.wav')
        sources = item['sources']['acoustic guitar']
        print(sources)
        sources.write_audio_to_file('./gen_acoustic_guitar.wav') 