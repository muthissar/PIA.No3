import os
import shutil

from DatasetManager.arrangement.arrangement_dataset import ArrangementDataset
from DatasetManager.arrangement.arrangement_midiPiano_dataset import ArrangementMidipianoDataset
from DatasetManager.arrangement.arrangement_voice_dataset import ArrangementVoiceDataset
from DatasetManager.chorale_dataset import ChoraleBeatsDataset
from DatasetManager.dataset_manager import DatasetManager
from DatasetManager.lsdb.lsdb_dataset import LsdbDataset
from DatasetManager.metadata import FermataMetadata, TickMetadata, KeyMetadata, BeatMarkerMetadata
from DatasetManager.orchestration.orchestration_dataset import OrchestrationDataset
from DatasetManager.the_session.folk_dataset import FolkDataset


###########################################################
# Orchestration
def build_orchestration(dataset_manager, batch_size, subdivision, sequence_size, integrate_discretization,
                        max_transposition, number_dump, test_bool):
    name = 'orchestration'
    if test_bool:
        name += '_debug'

    arrangement_dataset: OrchestrationDataset = dataset_manager.get_dataset(
        name=name,
        transpose_to_sounding_pitch=True,
        subdivision=subdivision,
        sequence_size=sequence_size,
        integrate_discretization=integrate_discretization,
        max_transposition=max_transposition,
        velocity_quantization=2,
        compute_statistics_flag=False,
    )

    (train_dataloader,
     val_dataloader,
     test_dataloader) = arrangement_dataset.data_loaders(
        batch_size=batch_size,
        split=(0.85, 0.10),
        num_workers=0,
        DEBUG_BOOL_SHUFFLE=True
    )
    print('Num Train Batches: ', len(train_dataloader))
    print('Num Valid Batches: ', len(val_dataloader))
    print('Num Test Batches: ', len(test_dataloader))

    # Visualise a few examples
    writing_dir = f"{arrangement_dataset.dump_folder}/arrangement/writing"
    if os.path.isdir(writing_dir):
        shutil.rmtree(writing_dir)
    os.makedirs(writing_dir)
    for i_batch, sample_batched in enumerate(train_dataloader):
        piano_batch, orchestra_batch, instrumentation_batch = sample_batched
        if i_batch > number_dump:
            break
        arrangement_dataset.visualise_batch(piano_batch, orchestra_batch, None, writing_dir, filepath=f"{i_batch}",
                                            writing_tempo='adagio', subdivision=subdivision, only_orchestra=False)
    return


###########################################################
# Arrangement
def build_arrangement(dataset_manager, batch_size, subdivision, sequence_size, integrate_discretization,
                      max_transposition, number_dump, test_bool):
    name = 'arrangement'
    if test_bool:
        name += '_small'

    arrangement_dataset: ArrangementDataset = dataset_manager.get_dataset(
        name=name,
        transpose_to_sounding_pitch=True,
        subdivision=subdivision,
        sequence_size=sequence_size,
        integrate_discretization=integrate_discretization,
        max_transposition=max_transposition,
        alignement_type='complete',
        velocity_quantization=2,
        compute_statistics_flag=False,
    )

    (train_dataloader,
     val_dataloader,
     test_dataloader) = arrangement_dataset.data_loaders(
        batch_size=batch_size,
        split=(0.85, 0.10),
        num_workers=0,
        DEBUG_BOOL_SHUFFLE=True
    )
    print('Num Train Batches: ', len(train_dataloader))
    print('Num Valid Batches: ', len(val_dataloader))
    print('Num Test Batches: ', len(test_dataloader))

    # Visualise a few examples
    writing_dir = f"{arrangement_dataset.dump_folder}/arrangement/writing"
    if os.path.isdir(writing_dir):
        shutil.rmtree(writing_dir)
    os.makedirs(writing_dir)
    for i_batch, sample_batched in enumerate(train_dataloader):
        piano_batch, orchestra_batch, instrumentation_batch = sample_batched
        if i_batch > number_dump:
            break
        arrangement_dataset.visualise_batch(piano_batch, orchestra_batch, None, writing_dir, filepath=f"{i_batch}",
                                            writing_tempo='adagio', subdivision=subdivision, only_orchestra=False)
    return


###########################################################
# Arrangement voice piano
def build_arrangement_voice(dataset_manager, batch_size, subdivision, sequence_size, integrate_discretization,
                            max_transposition, number_dump, test_bool):
    name = 'arrangement_voice'
    if test_bool:
        name += '_small'
    arrangement_dataset: ArrangementVoiceDataset = dataset_manager.get_dataset(
        name=name,
        transpose_to_sounding_pitch=True,
        subdivision=subdivision,
        sequence_size=sequence_size,
        integrate_discretization=integrate_discretization,
        max_transposition=max_transposition,
        alignement_type='complete',
        compute_statistics_flag=False,
    )

    (train_dataloader,
     val_dataloader,
     test_dataloader) = arrangement_dataset.data_loaders(
        batch_size=batch_size,
        num_workers=0,
        split=(0.85, 0.10),
        DEBUG_BOOL_SHUFFLE=True
    )
    print('Num Train Batches: ', len(train_dataloader))
    print('Num Valid Batches: ', len(val_dataloader))
    print('Num Test Batches: ', len(test_dataloader))

    # Visualise a few examples
    writing_dir = f"{arrangement_dataset.dump_folder}/arrangement_voice/writing"
    if os.path.isdir(writing_dir):
        shutil.rmtree(writing_dir)
    os.makedirs(writing_dir)
    for i_batch, sample_batched in enumerate(train_dataloader):
        piano_batch, orchestra_batch, instrumentation_batch = sample_batched
        if i_batch > number_dump:
            break
        arrangement_dataset.visualise_batch(piano_batch, orchestra_batch, None, writing_dir,
                                            filepath=f"{i_batch}")


###########################################################
#  Arrangement Midi piano
def build_arrangement_midi(dataset_manager,
                           batch_size,
                           subdivision,
                           sequence_size,
                           integrate_discretization,
                           max_transposition,
                           number_dump,
                           test_bool):
    mean_number_messages_per_time_frame = 14
    name = 'arrangement_midiPiano'
    if test_bool:
        name += '_small'
    arrangement_dataset: ArrangementMidipianoDataset = dataset_manager.get_dataset(
        name=name,
        transpose_to_sounding_pitch=True,
        subdivision=subdivision,
        sequence_size=sequence_size,
        max_transposition=max_transposition,
        integrate_discretization=integrate_discretization,
        alignement_type='complete',
        compute_statistics_flag=False,
        mean_number_messages_per_time_frame=mean_number_messages_per_time_frame
    )

    (train_dataloader,
     val_dataloader,
     test_dataloader) = arrangement_dataset.data_loaders(
        batch_size=batch_size,
        cache_dir=dataset_manager.cache_dir,
        split=(0.85, 0.10),
        DEBUG_BOOL_SHUFFLE=True
    )
    print('Num Train Batches: ', len(train_dataloader))
    print('Num Valid Batches: ', len(val_dataloader))
    print('Num Test Batches: ', len(test_dataloader))

    # Visualise a few examples
    writing_dir = f"{arrangement_dataset.dump_folder}/arrangement_midi/writing"
    if os.path.isdir(writing_dir):
        shutil.rmtree(writing_dir)
    os.makedirs(writing_dir)
    for i_batch, sample_batched in enumerate(train_dataloader):
        piano_batch, orchestra_batch, instrumentation_batch = sample_batched
        if i_batch > number_dump:
            break
        arrangement_dataset.visualise_batch(piano_batch, orchestra_batch, None, writing_dir, filepath=f"{i_batch}")


###########################################################
# BACH
def build_bach_beat(dataset_manager, batch_size, subdivision, sequences_size, test_bool):
    metadatas = [
        TickMetadata(subdivision=subdivision),
        FermataMetadata(),
        KeyMetadata()
    ]
    name = 'bach_chorales_beats'
    if test_bool:
        name += '_test'
    bach_chorales_dataset: ChoraleBeatsDataset = dataset_manager.get_dataset(
        name=name,
        voice_ids=[0, 1, 2, 3],
        metadatas=metadatas,
        sequences_size=sequences_size,
        subdivision=subdivision
    )
    (train_dataloader,
     val_dataloader,
     test_dataloader) = bach_chorales_dataset.data_loaders(
        batch_size=batch_size,
        num_workers=0,
        split=(0.85, 0.10)
    )
    print('Num Train Batches: ', len(train_dataloader))
    print('Num Valid Batches: ', len(val_dataloader))
    print('Num Test Batches: ', len(test_dataloader))


# LSDB
def build_lsdb(dataset_manager, batch_size, sequences_size):
    lsdb_dataset: LsdbDataset = dataset_manager.get_dataset(
        name='lsdb_test',
        sequences_size=sequences_size,
    )
    (train_dataloader,
     val_dataloader,
     test_dataloader) = lsdb_dataset.data_loaders(
        batch_size=batch_size,
        split=(0.85, 0.10)
    )
    print('Num Train Batches: ', len(train_dataloader))
    print('Num Valid Batches: ', len(val_dataloader))
    print('Num Test Batches: ', len(test_dataloader))


# Folk Dataset
def build_folk(dataset_manager, batch_size, subdivision, sequences_size):
    metadatas = [
        BeatMarkerMetadata(subdivision=subdivision),
        TickMetadata(subdivision=subdivision)
    ]
    folk_dataset_kwargs = {
        'metadatas': metadatas,
        'sequences_size': sequences_size
    }
    folk_dataset: FolkDataset = dataset_manager.get_dataset(
        name='folk_4by4nbars',
        **folk_dataset_kwargs
    )
    (train_dataloader,
     val_dataloader,
     test_dataloader) = folk_dataset.data_loaders(
        batch_size=batch_size,
        num_workers=0,
        split=(0.7, 0.2)
    )
    print('Num Train Batches: ', len(train_dataloader))
    print('Num Valid Batches: ', len(val_dataloader))
    print('Num Test Batches: ', len(test_dataloader))


if __name__ == '__main__':
    number_dump = 10
    batch_size = 8
    subdivision = 16
    sequence_size = 7
    integrate_discretization = True
    max_transposition = 2
    test_bool = True
    dataset_manager = DatasetManager()

    # build_orchestration(
    #     dataset_manager=dataset_manager,
    #     batch_size=batch_size,
    #     subdivision=subdivision,
    #     sequence_size=sequence_size,
    #     integrate_discretization=integrate_discretization,
    #     max_transposition=max_transposition,
    #     number_dump=number_dump,
    #     test_bool=test_bool
    # )

    # build_arrangement(
    #     dataset_manager=dataset_manager,
    #     batch_size=batch_size,
    #     subdivision=subdivision,
    #     sequence_size=sequence_size,
    #     integrate_discretization=integrate_discretization,
    #     max_transposition=max_transposition,
    #     number_dump=number_dump,
    #     test_bool=test_bool
    # )

    # build_arrangement_midi(
    #     dataset_manager=dataset_manager,
    #     batch_size=batch_size,
    #     subdivision=subdivision,
    #     sequence_size=sequence_size,
    #     integrate_discretization=integrate_discretization,
    #     max_transposition=max_transposition,
    #     number_dump=number_dump,
    #     test_bool=test_bool
    # )

    # build_arrangement_voice(
    #     dataset_manager=dataset_manager,
    #     batch_size=batch_size,
    #     subdivision=subdivision,
    #     sequence_size=sequence_size,
    #     integrate_discretization=integrate_discretization,
    #     max_transposition=max_transposition,
    #     number_dump=number_dump,
    #     test_bool=test_bool
    # )

    # build_bach_beat(
    #     dataset_manager=dataset_manager,
    #     batch_size=batch_size,
    #     subdivision=subdivision,
    #     sequences_size=sequence_size,
    #     test_bool=test_bool
    # )
