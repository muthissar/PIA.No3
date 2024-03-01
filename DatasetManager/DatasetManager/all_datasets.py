import os

import music21

import DatasetManager
from DatasetManager.arrangement.arrangement_midiPiano_dataset import ArrangementMidipianoDataset
from DatasetManager.arrangement.arrangement_voice_dataset import ArrangementVoiceDataset
from DatasetManager.arrangement.arrangement_dataset import ArrangementDataset
from DatasetManager.arrangement.arrangement_frame_dataset import ArrangementFrameDataset
from DatasetManager.arrangement.arrangement_helper import ArrangementIteratorGenerator
from DatasetManager.chorale_dataset import ChoraleDataset, ChoraleBeatsDataset
from DatasetManager.helpers import ShortChoraleIteratorGen
from DatasetManager.lsdb.lsdb_data_helpers import LeadsheetIteratorGenerator
from DatasetManager.lsdb.lsdb_dataset import LsdbDataset
from DatasetManager.orchestration.orchestration_dataset import OrchestrationDataset
from DatasetManager.orchestration.orchestration_helper import OrchestraIteratorGenerator
from DatasetManager.piano.piano_helper import PianoIteratorGenerator
from DatasetManager.piano.piano_midi_dataset import PianoMidiDataset
from DatasetManager.the_session.folk_data_helpers import FolkIteratorGenerator
from DatasetManager.the_session.folk_dataset import FolkDataset, FolkMeasuresDataset, FolkMeasuresDatasetTranspose, FolkDatasetNBars


def get_all_datasets():

    database_path = f'{os.path.expanduser("~")}/Data/databases'

    annex_dataset = OrchestraIteratorGenerator(
        folder_path=f"{database_path}/Orchestration/orchestral",
        subsets=[
            "kunstderfuge"
        ],
        process_file=True,
    )

    return {
        'arrangement':
            {
                'dataset_class_name': ArrangementDataset,
                'corpus_it_gen':      ArrangementIteratorGenerator(
                    arrangement_path=f"{database_path}/Orchestration/arrangement",
                    subsets=[
                        'liszt_classical_archives',
                    ],
                    num_elements=None,
                ),
                'corpus_it_gen_instru_range': None
            },
        'arrangement_large':
            {
                'dataset_class_name': ArrangementDataset,
                'corpus_it_gen':      ArrangementIteratorGenerator(
                    arrangement_path=f"{database_path}/Orchestration/arrangement",
                    subsets=[
                        'imslp',
                        'liszt_classical_archives',
                        'bouliane',
                        'hand_picked_Spotify',
                        # 'debug'
                    ],
                    num_elements=None,
                ),
                'corpus_it_gen_instru_range': annex_dataset
            },
        'arrangement_test':
            {
                'dataset_class_name': ArrangementDataset,
                'corpus_it_gen':      ArrangementIteratorGenerator(
                    arrangement_path=f"{database_path}/Orchestration/arrangement",
                    subsets=[
                        'debug',
                    ],
                    num_elements=None,
                )
            },
        'arrangement_small':
            {
                'dataset_class_name': ArrangementDataset,
                'corpus_it_gen':      ArrangementIteratorGenerator(
                    arrangement_path=f"{database_path}/Orchestration/arrangement",
                    subsets=[
                        'small_liszt_beethov',
                    ],
                    num_elements=None,
                ),
                'corpus_it_gen_instru_range': None
            },
        'arrangement_frame_test':
            {
                'dataset_class_name': ArrangementFrameDataset,
                'corpus_it_gen':      ArrangementIteratorGenerator(
                    arrangement_path=f"{database_path}/Orchestration/arrangement",
                    subsets=[
                        'debug'
                    ],
                    num_elements=None,
                )
            },
        'arrangement_frame':
            {
                'dataset_class_name': ArrangementFrameDataset,
                'corpus_it_gen':      ArrangementIteratorGenerator(
                    arrangement_path=f"{database_path}/Orchestration/arrangement",
                    subsets=[
                        'liszt_classical_archives',
                    ],
                    num_elements=None,
                )
            },
        'arrangement_voice':
            {
                'dataset_class_name': ArrangementVoiceDataset,
                'corpus_it_gen':      ArrangementIteratorGenerator(
                    arrangement_path=f"{database_path}/Orchestration/arrangement",
                    subsets=[
                        'liszt_classical_archives',
                    ],
                    num_elements=None,
                ),
            },
        'arrangement_voice_small':
            {
                'dataset_class_name': ArrangementVoiceDataset,
                'corpus_it_gen': ArrangementIteratorGenerator(
                    arrangement_path=f"{database_path}/Orchestration/arrangement",
                    subsets=[
                        'small_liszt_beethov',
                    ],
                    num_elements=None,
                ),
            },
        'arrangement_midiPiano':
            {
                'dataset_class_name': ArrangementMidipianoDataset,
                'corpus_it_gen': ArrangementIteratorGenerator(
                    arrangement_path=f"{database_path}/Orchestration/arrangement",
                    subsets=[
                        'liszt_classical_archives',
                    ],
                    num_elements=None,
                ),
            },
        'arrangement_midiPiano_small':
            {
                'dataset_class_name': ArrangementMidipianoDataset,
                'corpus_it_gen': ArrangementIteratorGenerator(
                    arrangement_path=f"{database_path}/Orchestration/arrangement",
                    subsets=[
                        'small_liszt_beethov',
                    ],
                    num_elements=None,
                ),
            },
        'orchestration':
            {
                'dataset_class_name': OrchestrationDataset,
                'corpus_it_gen': OrchestraIteratorGenerator(
                    folder_path=f"{database_path}/Orchestration/orchestral",
                    subsets=[
                        # 'musicalion',
                        'kunstderfuge',
                        'mutopia',
                        'open_music_scores',
                        'bouliane',
                        'hand_picked_spotify'
                    ],
                    process_file=True,
                ),
            },
        'orchestration_debug':
            {
                'dataset_class_name': OrchestrationDataset,
                'corpus_it_gen': OrchestraIteratorGenerator(
                    folder_path=f"{database_path}/Orchestration/orchestral",
                    subsets=[
                        'debug'
                    ],
                    process_file=True,
                ),
            },
        'piano_midi':
            {
                'dataset_class_name': PianoMidiDataset,
                'corpus_it_gen': PianoIteratorGenerator(
                    subsets=[
                        'classic_piano_dataset',
                        'ecomp_piano_dataset'
                    ],
                    num_elements=None
                ),
            },
        'piano_midi_small':
            {'dataset_class_name': PianoMidiDataset,
                'corpus_it_gen': PianoIteratorGenerator(
                    subsets=[
                        'debug'
                    ],
                    num_elements=None
                ),
            },
        'bach_chorales':
            {
                'dataset_class_name': ChoraleDataset,
                'corpus_it_gen':      music21.corpus.chorales.Iterator
            },
        'bach_chorales_beats':
            {
                'dataset_class_name': ChoraleBeatsDataset,
                'corpus_it_gen':      music21.corpus.chorales.Iterator
            },
        'bach_chorales_beats_test':
            {
                'dataset_class_name': ChoraleBeatsDataset,
                'corpus_it_gen':      ShortChoraleIteratorGen()
            },
        'bach_chorales_test':
            {
                'dataset_class_name': ChoraleDataset,
                'corpus_it_gen':      ShortChoraleIteratorGen()
            },
        'lsdb_test':
            {
                'dataset_class_name': LsdbDataset,
                'corpus_it_gen':      LeadsheetIteratorGenerator(
                    num_elements=10
                )
            },
        'lsdb':
            {
                'dataset_class_name': LsdbDataset,
                'corpus_it_gen':      LeadsheetIteratorGenerator(
                    num_elements=None
                )
            },
        'folk':
            {
                'dataset_class_name': FolkDataset,
                'corpus_it_gen':      FolkIteratorGenerator(
                    num_elements=None,
                    has_chords=False,
                    time_sigs=[(3, 4), (4, 4)]
                )
            },
        'folk_test':
            {
                'dataset_class_name': FolkDataset,
                'corpus_it_gen':      FolkIteratorGenerator(
                    num_elements=10,
                    has_chords=False,
                    time_sigs=[(3, 4), (4, 4)]
                )
            },
        'folk_4by4_test':
            {
                'dataset_class_name': FolkDataset,
                'corpus_it_gen':      FolkIteratorGenerator(
                    num_elements=100,
                    has_chords=False,
                    time_sigs=[(4, 4)]
                )
            },
        'folk_4by4':
            {
                'dataset_class_name': FolkDataset,
                'corpus_it_gen':      FolkIteratorGenerator(
                    num_elements=None,
                    has_chords=False,
                    time_sigs=[(4, 4)]
                )
            },
        'folk_3by4_test':
            {
                'dataset_class_name': FolkDataset,
                'corpus_it_gen':      FolkIteratorGenerator(
                    num_elements=100,
                    has_chords=False,
                    time_sigs=[(3, 4)]
                )
            },
        'folk_3by4':
            {
                'dataset_class_name': FolkDataset,
                'corpus_it_gen':      FolkIteratorGenerator(
                    num_elements=None,
                    has_chords=False,
                    time_sigs=[(3, 4)]
                )
            },
        'folk_4by4chords':
            {
                'dataset_class_name': FolkDataset,
                'corpus_it_gen': FolkIteratorGenerator(
                    num_elements=None,
                    has_chords=True,
                    time_sigs=[(4, 4)]
                )
            },
        'folk_4by4measures_test':
            {
                'dataset_class_name': FolkMeasuresDataset,
                'corpus_it_gen': FolkIteratorGenerator(
                    num_elements=100,
                    has_chords=False,
                    time_sigs=[(4, 4)]
                )
            },
        'folk_4by4measures_test2':
            {
                'dataset_class_name': FolkMeasuresDataset,
                'corpus_it_gen': FolkIteratorGenerator(
                    num_elements=1,
                    has_chords=False,
                    time_sigs=[(4, 4)]
                )
            },
        'folk_4by4measures':
            {
                'dataset_class_name': FolkMeasuresDataset,
                'corpus_it_gen': FolkIteratorGenerator(
                    num_elements=None,
                    has_chords=False,
                    time_sigs=[(4, 4)]
                )
            },
        'folk_4by4measurestr_test':
            {
                'dataset_class_name': FolkMeasuresDatasetTranspose,
                'corpus_it_gen': FolkIteratorGenerator(
                    num_elements=1000,
                    has_chords=False,
                    time_sigs=[(4, 4)]
                )
            },
        'folk_4by4measurestr':
            {
                'dataset_class_name': FolkMeasuresDatasetTranspose,
                'corpus_it_gen': FolkIteratorGenerator(
                    num_elements=None,
                    has_chords=False,
                    time_sigs=[(4, 4)]
                )
            },
        'folk_4by4nbars_test':
            {
                'dataset_class_name': FolkDatasetNBars,
                'corpus_it_gen': FolkIteratorGenerator(
                    num_elements=100,
                    has_chords=False,
                    time_sigs=[(4, 4)]
                )
            },
        'folk_4by4nbars':
            {
                'dataset_class_name': FolkDatasetNBars,
                'corpus_it_gen': FolkIteratorGenerator(
                    num_elements=None,
                    has_chords=False,
                    time_sigs=[(4, 4)]
                )
            },
    }
