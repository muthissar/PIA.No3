import csv
import json
import math
import os
import re
import shutil

import torch
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from DatasetManager.arrangement.instrumentation import get_instrumentation
from tqdm import tqdm
import music21
import numpy as np
from DatasetManager.helpers import START_SYMBOL, END_SYMBOL

from DatasetManager.music_dataset import MusicDataset
import DatasetManager.arrangement.nw_align as nw_align

from DatasetManager.arrangement.arrangement_helper import note_to_midiPitch, score_to_pianoroll, \
    midiPitch_to_octave_pc, quantize_and_filter_music21_element


class ArrangementFrameDataset(MusicDataset):
    """
    Class for all arrangement dataset
    It is highly recommended to run arrangement_statistics before building the database
    """

    def __init__(self,
                 corpus_it_gen,
                 name,
                 subdivision=2,
                 transpose_to_sounding_pitch=True,
                 reference_tessitura_path='/home/leo/Recherche/DatasetManager/DatasetManager/arrangement/reference_tessitura.json',
                 simplify_instrumentation_path='/home/leo/Recherche/DatasetManager/DatasetManager/arrangement/simplify_instrumentation.json',
                 cache_dir=None,
                 compute_statistics_flag=None):
        """
        :param corpus_it_gen: calling this function returns an iterator
        over chorales (as music21 scores)
        :param name:
        :param metadatas: list[Metadata], the list of used metadatas
        :param subdivision: number of sixteenth notes per beat
        :param cache_dir: directory where tensor_dataset is stored
        """
        super(ArrangementFrameDataset, self).__init__(cache_dir=cache_dir)
        self.name = name
        self.corpus_it_gen = corpus_it_gen
        self.subdivision = subdivision  # We use only on beats notes so far

        # Reference tessitura for each instrument
        with open(reference_tessitura_path, 'r') as ff:
            tessitura = json.load(ff)
        self.reference_tessitura = {k: (music21.note.Note(v[0]), music21.note.Note(v[1])) for k, v in tessitura.items()}

        # Maps parts name found in mxml files to standard names
        with open(simplify_instrumentation_path, 'r') as ff:
            self.simplify_instrumentation = json.load(ff)

        #  Instrumentation used for learning
        self.instrumentation = get_instrumentation()

        #  Do we use written or sounding pitch
        self.transpose_to_sounding_pitch = transpose_to_sounding_pitch

        # Mapping between instruments and indices
        self.piano_tessitura = {}
        self.index2instrument = {}
        self.instrument2index = {}
        self.index2midi_pitch = {}
        self.midi_pitch2index = {}

        # Compute statistics slows down the construction of the dataset
        self.compute_statistics_flag = compute_statistics_flag
        return

    def __repr__(self):
        return f'ArrangementFrameDataset(' \
            f'{self.name},' \
            f'{self.subdivision})'

    def iterator_gen(self):
        return (self.sort_arrangement_pairs(arrangement_pair)
                for arrangement_pair in self.corpus_it_gen())

    def compute_index_dicts(self):
        #  Mapping midi_pitch to token for each instrument
        set_midiPitch_per_instrument ={'Piano': set()}
        # First pass over the database to create the mapping pitch <-> index for each instrument
        for arr_id, arr_pair in tqdm(enumerate(self.iterator_gen())):
            # Compute pianoroll representations of score (more efficient than manipulating the music21 streams)
            pianoroll_piano, _ = score_to_pianoroll(arr_pair['Piano'], self.subdivision, self.simplify_instrumentation,
                                                 self.transpose_to_sounding_pitch)
            pitch_set_this_track = set(np.where(np.sum(pianoroll_piano['Piano'], axis=0) > 0)[0])
            set_midiPitch_per_instrument['Piano'] = set_midiPitch_per_instrument['Piano'].union(pitch_set_this_track)

            pianoroll_orchestra, _ = score_to_pianoroll(arr_pair['Orchestra'], self.subdivision,
                                                     self.simplify_instrumentation, self.transpose_to_sounding_pitch)
            for instrument_name in pianoroll_orchestra:
                if instrument_name not in set_midiPitch_per_instrument.keys():
                    set_midiPitch_per_instrument[instrument_name] = set()
                pitch_set_this_track = set(np.where(np.sum(pianoroll_orchestra[instrument_name], axis=0) > 0)[0])
                set_midiPitch_per_instrument[instrument_name] = set_midiPitch_per_instrument[instrument_name].union(pitch_set_this_track)

        # Save this in a file
        if self.compute_statistics_flag:
            with open(f"{self.compute_statistics_flag}/pc_frequency_per_instrument", "w") as ff:
                for instrument_name, set_pitch_class in set_midiPitch_per_instrument.items():
                    ff.write(f"# {instrument_name}: \n")
                    for pc in set_pitch_class:
                        ff.write(f"   {pc}\n")

        # Local dicts used temporarily
        midi_pitch2index_per_instrument = {}
        index2midi_pitch_per_instrument = {}
        for instrument_name, set_midiPitch in  set_midiPitch_per_instrument.items():
            if instrument_name == "Piano":
                continue
            list_midiPitch = sorted(list(set_midiPitch))
            midi_pitch2index_per_instrument[instrument_name] = {}
            index2midi_pitch_per_instrument[instrument_name] = {}
            for index, midi_pitch in enumerate(list_midiPitch):
                midi_pitch2index_per_instrument[instrument_name][midi_pitch] = index
                index2midi_pitch_per_instrument[instrument_name][index] = midi_pitch
            # Silence
            midi_pitch2index_per_instrument[instrument_name][-1] = index+1
            index2midi_pitch_per_instrument[instrument_name][index+1] = -1

        # Mapping piano notes <-> indices
        self.piano_tessitura['lowest_pitch'] = min(list(set_midiPitch_per_instrument['Piano']))
        self.piano_tessitura['highest_pitch'] = max(list(set_midiPitch_per_instrument['Piano']))

        # Mapping instruments <-> indices
        self.instrument2index[START_SYMBOL] = [0]
        self.index2instrument[0] = START_SYMBOL

        index_counter = 1

        for instrument_name, number_instruments in self.instrumentation.items():
            # Check if instrument appears in the dataset
            if instrument_name not in midi_pitch2index_per_instrument.keys():
                continue
            self.instrument2index[instrument_name] = list(range(index_counter, index_counter + number_instruments))
            for ind in range(index_counter, index_counter + number_instruments):
                self.index2instrument[ind] = instrument_name
            index_counter += number_instruments
        self.instrument2index[END_SYMBOL] = [index_counter]
        self.index2instrument[index_counter] = END_SYMBOL
        index_counter += 1

        # Mapping pitch <-> index per voice (that's the one we'll use, easier to manipulate when training)
        for instrument_name, instrument_indices in self.instrument2index.items():
            for instrument_index in instrument_indices:
                if instrument_name in [START_SYMBOL, END_SYMBOL]:
                    # No padding useful here, we know our sequence length beforehand
                    # START and END symbols make it easier to fit standard transofrmers frameworks
                    self.midi_pitch2index[instrument_index] = {0: 0}
                    self.index2midi_pitch[instrument_index] = {0: 0}
                else:
                    self.midi_pitch2index[instrument_index] = midi_pitch2index_per_instrument[instrument_name]
                    self.index2midi_pitch[instrument_index] = index2midi_pitch_per_instrument[instrument_name]

        self.number_parts = len(self.midi_pitch2index)
        self.number_pitch_piano = self.piano_tessitura['highest_pitch'] - self.piano_tessitura['lowest_pitch'] + 1

        return

    def make_tensor_dataset(self):
        """
        Implementation of the make_tensor_dataset abstract base class
        """
        print('Making tensor dataset')

        self.compute_index_dicts()

        # one_tick = 1 / self.subdivision
        piano_tensor_dataset = []
        orchestra_tensor_dataset = []
        # metadata_tensor_dataset = []

        total_frames_counter = 0
        missed_frames_counter = 0

        # Variables for statistics
        scores = []
        num_frames_with_different_pitch_class = 0

        if self.compute_statistics_flag is not None:
            open(f"{self.compute_statistics_flag}/different_set_pc.txt", 'w').close()

        for arr_id, arr_pair in tqdm(enumerate(self.iterator_gen())):

            # Get pianorolls
            pianoroll_piano, num_frames_piano = score_to_pianoroll(arr_pair['Piano'], self.subdivision, self.simplify_instrumentation,
                                                 self.transpose_to_sounding_pitch)
            pianoroll_orchestra, num_frames_orchestra = score_to_pianoroll(arr_pair['Orchestra'], self.subdivision,
                                                     self.simplify_instrumentation, self.transpose_to_sounding_pitch)

            # Align
            # corresponding_offsets, this_scores = self.align_score(pianoroll_piano, num_frames_piano, pianoroll_orchestra, num_frames_orchestra)
            # Alignement
            corresponding_frames, this_scores = self.align_score(arr_pair['Piano'], arr_pair['Orchestra'])
            scores.extend(this_scores)

            # main loop
            for ((frame_piano, pc_piano), (frame_orchestra, pc_orchestra)) in corresponding_frames:
                #################################################
                # NO TRANSPOSITIONS ALLOWED FOR NOW
                # Only consider orchestra to compute the possible transpositions
                # BUT IF WE ALLOW THEM, WE SHOULD
                # 1/ COMPUTE ALL TRANSPOSISITONS IN -3 +3 RANGE
                #  2/ SELECT FRAMES WHERE IT'S POSSIBLE
                # transp_minus, transp_plus = self.possible_transposition(arr_pair['Orchestra'], offset=offset_orchestra)
                # for semi_tone in range(transp_minus, transp_plus + 1):
                #################################################

                total_frames_counter += 1

                #  Piano
                assert (pianoroll_piano.keys() != 'Piano'), 'More than one instrument in the piano score'
                piano_np = pianoroll_piano['Piano'][frame_piano]
                # Squeeze to observed tessitura
                piano_np_reduced = piano_np[
                                   self.piano_tessitura['lowest_pitch']: self.piano_tessitura['highest_pitch'] + 1]
                local_piano_tensor = torch.from_numpy(piano_np_reduced)

                # Orchestra
                local_orchestra_tensor = self.pianoroll_to_orchestral_tensor(pianoroll_orchestra,
                                                                        frame_orchestra,
                                                                        self.number_parts)

                if local_orchestra_tensor is None:
                    missed_frames_counter += 1
                    continue

                # Statistics: compare pitch class in orchestra and in piano
                if self.compute_statistics_flag:
                    if pc_piano != pc_orchestra:
                        num_frames_with_different_pitch_class += 1
                        with open(f"{self.compute_statistics_flag}/different_set_pc.txt", "a") as ff:
                            for this_pc in pc_piano:
                                ff.write(f"{this_pc} ")
                            ff.write("// ")
                            for this_pc in pc_orchestra:
                                ff.write(f"{this_pc} ")
                            ff.write("\n")

                # append and add batch dimension
                # cast to int
                piano_tensor_dataset.append(
                    local_piano_tensor[None, :].int())
                orchestra_tensor_dataset.append(
                    local_orchestra_tensor[None, :].int())

        piano_tensor_dataset = torch.cat(piano_tensor_dataset, 0)
        orchestra_tensor_dataset = torch.cat(orchestra_tensor_dataset, 0)

        #######################
        if self.compute_statistics_flag:
            # NW statistics
            mean_score = np.mean(scores)
            variance_score = np.var(scores)
            max_score = np.max(scores)
            min_score = np.min(scores)
            nw_statistics_folder = f"{self.compute_statistics_flag}/nw"
            if os.path.isdir(nw_statistics_folder):
                shutil.rmtree(nw_statistics_folder)
            os.makedirs(nw_statistics_folder)
            with open(f"{nw_statistics_folder}/scores.txt", "w") as ff:
                ff.write(f"Mean score: {mean_score}\n")
                ff.write(f"Variance score: {variance_score}\n")
                ff.write(f"Max score: {max_score}\n")
                ff.write(f"Min score: {min_score}\n")
                for elem in scores:
                    ff.write(f"{elem}\n")
            # Histogram
            n, bins, patches = plt.hist(scores, 50)
            plt.xlabel('Score')
            plt.ylabel('Frequency')
            plt.title('Histogram NW scores')
            plt.savefig(f'{nw_statistics_folder}/histogram_score.pdf')

            # Pitch class statistics
            pitch_class_statistics_folder = f"{self.compute_statistics_flag}/pitch_class"
            if os.path.isdir(pitch_class_statistics_folder):
                shutil.rmtree(pitch_class_statistics_folder)
            os.makedirs(pitch_class_statistics_folder)
            # Move different set pc
            shutil.move(f"{self.compute_statistics_flag}/different_set_pc.txt", f"{pitch_class_statistics_folder}/different_set_pc.txt")
            # Write the ratio of matching frames
            with open(f"{pitch_class_statistics_folder}/ratio_matching_pc_set.txt", "w") as ff:
                ff.write(f"Different PC sets: {num_frames_with_different_pitch_class}\n")
                ff.write(f"Total number frames: {total_frames_counter}\n")
                ff.write(f"Ratio: {num_frames_with_different_pitch_class / total_frames_counter}\n")
        #######################

        #######################
        # Create Tensor Dataset
        dataset = TensorDataset(piano_tensor_dataset,
                                orchestra_tensor_dataset)
        #######################

        print(
            # f'Sizes: \n Piano: {piano_tensor_dataset.size()}\n {orchestra_tensor_dataset.size()}\n {metadata_tensor_dataset.size()}\n')
            f'Sizes: \n Piano: {piano_tensor_dataset.size()}\n Orchestra: {orchestra_tensor_dataset.size()}\n')
        print(f'Missed frames:\n {missed_frames_counter} \n {missed_frames_counter / total_frames_counter}')
        return dataset

    def get_score_tensor(self, scores, offsets):
        return

    def transposed_score_and_metadata_tensors(self, score, semi_tone):
        return

    def get_metadata_tensor(self, score):
        return None

    def score_to_list_pc(self, score):
        # Need only the flatten orchestra for aligning
        if score.atSoundingPitch != 'unknown':
            sounding_pitch_score = score.toSoundingPitch()
        score_flat = sounding_pitch_score.flat
        notes_and_chords = score_flat.notes

        list_pc = []
        current_frame_index = 0
        current_set_pc = set()

        for elem in notes_and_chords:
            # Don't consider elements which are not on a subdivision of the beat
            note_start, note_end = quantize_and_filter_music21_element(elem, self.subdivision)
            if note_start is None:
                continue
            assert (note_start >= current_frame_index), "Elements are not sorted by increasing time ?"
            if note_start > current_frame_index:
                # Write in list_pc and move to next
                if len(current_set_pc) > 0:  # Check on length is only for the first iteration
                    list_pc.append((note_start, current_set_pc))
                current_set_pc = set()
                current_frame_index = note_start

            if elem.isNote:
                current_set_pc.add(elem.pitch.pitchClass)
            else:
                current_set_pc = current_set_pc.union(set(elem.pitchClasses))
        return list_pc

    def align_score(self, piano_score, orchestra_score):
        list_pc_piano = self.score_to_list_pc(piano_score)
        list_pc_orchestra = self.score_to_list_pc(orchestra_score)

        only_pc_piano = [e[1] for e in list_pc_piano]
        only_pc_orchestra = [e[1] for e in list_pc_orchestra]

        corresponding_indices, score_matrix = nw_align.nwalign(only_pc_piano, only_pc_orchestra, gapOpen=-3, gapExtend=-1)

        corresponding_frames = [(list_pc_piano[ind_piano], list_pc_orchestra[ind_orchestra])
                                 for ind_piano, ind_orchestra in corresponding_indices]

        return corresponding_frames, score_matrix

    ###################################
    # Small helpers for quickly determining which score is orchestra and which one is piano
    def sort_arrangement_pairs(self, arrangement_pair):
        # Find which score is piano and which is orchestral
        if len(self.list_instru_score(arrangement_pair[0])) > len(self.list_instru_score(arrangement_pair[1])):
            return {'Orchestra': arrangement_pair[0], 'Piano': arrangement_pair[1]}
        elif len(self.list_instru_score(arrangement_pair[0])) < len(self.list_instru_score(arrangement_pair[1])):
            return {'Piano': arrangement_pair[0], 'Orchestra': arrangement_pair[1]}
        else:
            raise Exception('The two scores have the same number of instruments')

    def list_instru_score(self, score):
        list_instru = []
        for part in score.parts:
            list_instru.append(part.partName)
        return list_instru
    ###################################

    def pianoroll_to_orchestral_tensor(self, pianoroll, frame_index, n_instruments):
        orchestra_np = np.zeros((n_instruments))
        for instrument_name, indices_instruments in self.instrument2index.items():
            if instrument_name in [START_SYMBOL, END_SYMBOL]:
                # Empty markers for Start and End
                continue
            number_of_parts = len(indices_instruments)
            if instrument_name not in pianoroll:
                notes_played = []
            else:
                notes_played = list(np.where(pianoroll[instrument_name][frame_index])[0])
            if len(notes_played) > number_of_parts:
                return None
            # Pad with -1 which means silence
            notes_played.extend([-1] * (number_of_parts - len(notes_played)))
            for this_note, this_instrument_index in zip(notes_played, indices_instruments):
                orchestra_np[this_instrument_index] = self.midi_pitch2index[this_instrument_index][this_note]

        orchestra_tensor = torch.from_numpy(orchestra_np)
        return orchestra_tensor

    def extract_score_tensor_with_padding(self, tensor_score):
        return None

    def extract_metadata_with_padding(self, tensor_metadata, start_tick, end_tick):
        return None

    def empty_score_tensor(self, score_length):

        return None

    def random_score_tensor(self, score_length):
        return None

    def get_offset_duration_from_frames_onsets(self, rhythm):
        rhythm = np.asarray(rhythm)
        next_rhythm = list(rhythm[1:])
        next_rhythm.append(rhythm[-1] + 1)
        durations = next_rhythm - rhythm
        offsets = [float(e / self.subdivision) for e in rhythm]
        durations = [float(e / self.subdivision) for e  in durations]
        return offsets, durations

    def piano_tensor_to_score(self, tensor_score, rhythm):

        assert len(tensor_score) == len(rhythm), "Score and rhythm must have the same length"
        piano_matrix= tensor_score.numpy()

        this_part = music21.stream.Part(id='Piano')
        music21_instrument = music21.instrument.fromString('Piano')
        this_part.insert(music21_instrument)
        offsets, durations = self.get_offset_duration_from_frames_onsets(rhythm)
        for frame_index, (offset, duration) in enumerate(zip(offsets, durations)):
            piano_vector = piano_matrix[frame_index]
            piano_notes = np.where(piano_vector > 0)
            note_list = []
            for piano_note in piano_notes[0]:
                pitch = self.piano_tessitura['lowest_pitch'] + piano_note
                velocity = piano_vector[piano_note]
                f = music21.note.Note(pitch)
                f.volume.velocity = velocity
                f.quarterLength = duration
                note_list.append(f)

            if len(note_list) == 0:
                this_part.insert(offset, music21.note.Rest())
            else:
                chord = music21.chord.Chord(note_list)
                this_part.insert(offset, chord)
        return this_part

    def orchestra_tensor_to_score(self, tensor_score, rhythm):
        # (batch, num_parts, notes_encoding)
        orchestra_matrix = tensor_score.numpy()

        # Batch is used as time in the score
        stream = music21.stream.Stream()

        offsets, durations = self.get_offset_duration_from_frames_onsets(rhythm)

        # First store every in a dict {instrus : [time [notes]]}
        score_dict = {}
        for instrument_index in range(self.number_parts):
            # Get instrument name
            instrument_name = self.index2instrument[instrument_index]
            if instrument_name in [START_SYMBOL, END_SYMBOL]:
                continue
            if instrument_name not in score_dict:
                score_dict[instrument_name] = {}
            for frame_index, (offset, duration) in enumerate(zip(offsets, durations)):
                midi_pitch = self.index2midi_pitch[instrument_index][orchestra_matrix[frame_index, instrument_index]]
                if midi_pitch == -1:
                    continue
                if offset not in score_dict[instrument_name]:
                    score_dict[instrument_name][offset] = (duration, set())
                score_dict[instrument_name][offset][1].add(midi_pitch)

        for instrument_name, list_notes in score_dict.items():
            this_part = music21.stream.Part(id=instrument_name)
            # re is for removing underscores in instrument names which raise errors in music21
            if instrument_name == "Cymbal":
                music21_instrument = music21.instrument.Cymbals()
            else:
                music21_instrument = music21.instrument.fromString(re.sub('_', ' ', instrument_name))
            this_part.insert(music21_instrument)

            for offset, note_informations in list_notes.items():
                note_duration, note_set = note_informations
                chord_list = []
                # Remove possible silence
                for midi_pitch in note_set:
                    this_note = music21.note.Note(midi_pitch)
                    this_note.quarterLength = note_duration
                    this_note.volume.velocity = 90
                    chord_list.append(this_note)
                this_elem = music21.chord.Chord(chord_list)
                this_part.insert(offset, this_elem)
            this_part.atSoundingPitch = self.transpose_to_sounding_pitch
            stream.append(this_part)

        return stream

    def tensor_to_score(self, tensor_score, score_type):
        if score_type == 'piano':
            return self.piano_tensor_to_score(tensor_score)
        elif score_type == 'orchestra':
            return self.orchestra_tensor_to_score(tensor_score)
        else:
            raise Exception(f"Expected score_type to be either piano or orchestra. Got {score_type} instead.")

    def visualise_batch(self, piano_infos, orchestra_infos, writing_dir=None, filepath=None):
        # data is a matrix (batch, ...)
        # Visualise a few examples
        if writing_dir is None:
            writing_dir = f"{os.getcwd()}/dump"

        piano_part = self.piano_tensor_to_score(piano_infos[0], piano_infos[1])
        orchestra_stream = self.orchestra_tensor_to_score(orchestra_infos[0], orchestra_infos[1])


        piano_part.write(fp=f"{writing_dir}/{filepath}_piano.xml", fmt='musicxml')
        orchestra_stream.write(fp=f"{writing_dir}/{filepath}_orchestra.xml", fmt='musicxml')
        # Both in the same score
        orchestra_stream.append(piano_part)
        orchestra_stream.write(fp=f"{writing_dir}/{filepath}_both.xml", fmt='musicxml')

if __name__ == '__main__':
    #  Read
    from DatasetManager.arrangement.arrangement_helper import ArrangementIteratorGenerator

    corpus_it_gen = ArrangementIteratorGenerator(
        arrangement_path='/home/leo/Recherche/databases/Orchestration/arrangement_mxml',
        subsets=[
            # 'bouliane',
            # 'imslp',
            # 'liszt_classical_archives',
            # 'hand_picked_Spotify',
            'debug'
        ],
        num_elements=None
    )

    kwargs = {}
    kwargs.update(
        {'name': "arrangement_frame_test",
         'corpus_it_gen': corpus_it_gen,
         'cache_dir': '/home/leo/Recherche/DatasetManager/DatasetManager/dataset_cache',
         'subdivision': 2,
         'transpose_to_sounding_pitch': True,
         'compute_statistics_flag': "/home/leo/Recherche/DatasetManager/DatasetManager/arrangement/statistics"
         })

    dataset = ArrangementFrameDataset(**kwargs)
    print(f'Creating {dataset.__repr__()}, '
          f'both tensor dataset and parameters')
    if os.path.exists(dataset.tensor_dataset_filepath):
        os.remove(dataset.tensor_dataset_filepath)
    tensor_dataset = dataset.tensor_dataset

    # Data loaders
    (train_dataloader,
     val_dataloader,
     test_dataloader) = dataset.data_loaders(
        batch_size=16,
        split=(0.85, 0.10),
        DEBUG_BOOL_SHUFFLE=False
    )

    # Visualise a few examples
    number_dump = 20
    writing_dir = f"{os.getcwd()}/dump"
    if os.path.isdir(writing_dir):
        shutil.rmtree(writing_dir)
    os.makedirs(writing_dir)
    for i_batch, sample_batched in enumerate(train_dataloader):
        piano_batch, orchestra_batch = sample_batched
        if i_batch > number_dump:
            break
        dataset.visualise_batch(piano_batch, orchestra_batch, writing_dir, filepath=f"{i_batch}")