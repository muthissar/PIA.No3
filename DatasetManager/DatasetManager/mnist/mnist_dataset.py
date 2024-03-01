import matplotlib as mpl
import torchvision

mpl.use('Agg')

from DatasetManager.music_dataset import MusicDataset


class MnistDataset(MusicDataset):
    """
    Class for all arrangement dataset
    It is highly recommended to run arrangement_statistics before building the database
    """

    def __init__(self,
                 name,
                 cache_dir=None):
        """
        :param corpus_it_gen: calling this function returns an iterator
        over chorales (as music21 scores)
        :param name:
        :param metadatas: list[Metadata], the list of used metadatas
        :param subdivision: number of sixteenth notes per beat
        :param cache_dir: directory where tensor_dataset is stored
        """
        super(MnistDataset, self).__init__(cache_dir=cache_dir)
        self.name = name
        return

    def __repr__(self):
        return f'ArrangementDataset-' \
            f'{self.name}'

    def iterator_gen(self):
        return (arrangement_pair for arrangement_pair in self.corpus_it_gen())

    def iterator_gen_complementary(self):
        return (score for score in self.corpus_it_gen_instru_range())

    def make_tensor_dataset(self, frame_orchestra=None):
        dataset = torchvision.datasets.MNIST(root='.', train=True, download=True,
                                             transform=torchvision.transforms.Compose([
                                                 torchvision.transforms.ToTensor(),
                                                 torchvision.transforms.Normalize((0.1307,), (0.3081,))
                                             ]))
        return dataset

    def transposed_score_and_metadata_tensors(self, score, semi_tone):
        return

    def get_metadata_tensor(self, score):
        return None

    def extract_score_tensor_with_padding(self, tensor_score):
        return None

    def extract_metadata_with_padding(self, tensor_metadata, start_tick, end_tick):
        return None

    def empty_score_tensor(self, score_length):
        return None

    def random_score_tensor(self, score_length):
        return None

    def visualise_batch(self, piano_pianoroll, orchestra_pianoroll, durations_piano=None, writing_dir=None,
                        filepath=None, writing_tempo='adagio', subdivision=None):
        pass
