import os
import random
from itertools import islice
import matplotlib
import matplotlib.image
import numpy as np
import torch
from torch.utils import data


class DSpriteMovieDataset(data.IterableDataset):
    """
    Class for all arrangement dataset
    It is highly recommended to run arrangement_statistics before building the database
    """

    def __init__(self, width, height, duration, latent_features, random_position, random_colour):
        """
        :param corpus_it_gen: calling this function returns an iterator
        over chorales (as music21 scores)
        :param name:
        :param metadatas: list[Metadata], the list of used metadatas
        :param subdivision: number of sixteenth notes per beat
        """
        super().__init__()

        # image dimensions
        self.width = width  #  width is x. 0 is left
        self.height = height  #  height is y. 0 is top
        self.channel = 3
        self.duration = duration

        # latent factors
        # self.shapes = ['triangle', 'square', 'circle']
        self.shapes = ['square']
        self.colours = ['red', 'green', 'blue', 'yellow', 'cyan', 'magenta']
        self.initial_sizes = range(2, 8, 2)
        self.max_delta_position = 4
        self.size_growths = [-2, -1, 0, 1, 2]
        self.background_colours = ['black', 'white']
        # list of factors actually varying and modeled in the dataset
        self.latent_features = latent_features
        # {
        #     'shape': True,
        #     'colour': True,
        #     'background_colour': True,
        #     'size': True,
        #     'size_growth': False,
        # }

        self.random_position = random_position
        self.random_colour = random_colour


        #  mapping symbolic to value
        self.colour_map = {
            'red': torch.tensor([255, 0, 0]).long(),
            'green': torch.tensor([0, 255, 0]).long(),
            'blue': torch.tensor([0, 0, 255]).long(),
            'yellow': torch.tensor([255, 255, 0]).long(),
            'magenta': torch.tensor([255, 0, 255]).long(),
            'cyan': torch.tensor([0, 255, 255]).long(),
            'black': torch.tensor([0, 0, 0]).long(),
            'white': torch.tensor([255, 255, 255]).long()
        }
        return

    def __repr__(self):
        name = f'DSpriteMovie'
        return name

    def __str__(self):
        return f'DSpriteMovie'

    def __iter__(self):
        return self

    def __next__(self):
        """
        Generates one sample of data
        """
        # movie
        movie = torch.zeros((self.height, self.width, self.channel, self.duration))
        positions_x = torch.zeros((self.duration))
        positions_y = torch.zeros((self.duration))
        dx = torch.zeros((self.duration))
        dy = torch.zeros((self.duration))

        # randomly draw trajectories for latent factors
        if self.latent_features['colour']:
            colour = self.colour_map[random.sample(self.colours, 1)[0]]
        else:
            colour = self.colour_map['red']

        if self.latent_features['background_colour']:
            background_colour = self.colour_map[random.sample(self.background_colours, 1)[0]]
        else:
            background_colour = self.colour_map['white']

        if self.latent_features['shape']:
            shape = random.sample(self.shapes, 1)
        else:
            shape = 'square'

        if self.latent_features['size']:
            initial_size = random.sample(self.initial_sizes, 1)[0]
        else:
            initial_size = 4

        if self.latent_features['size_growth']:
            size_growth = random.sample(self.size_growths)
        else:
            size_growth = 0

        # init position
        initial_delta_x = random.randint(-self.max_delta_position, self.max_delta_position)
        initial_delta_y = random.randint(-self.max_delta_position, self.max_delta_position)
        initial_direction = (initial_delta_x, initial_delta_y)

        # draw background
        movie[:, :, 0, :] = background_colour[0]
        movie[:, :, 1, :] = background_colour[1]
        movie[:, :, 2, :] = background_colour[2]

        # initial position
        x_init = random.randint(initial_size // 2, self.width - initial_size // 2)
        y_init = random.randint(initial_size // 2, self.width - initial_size // 2)

        x_t = x_init
        y_t = y_init
        size_t = initial_size
        half_size_t = size_t // 2
        direction_t = initial_direction

        for t in range(self.duration):
            ########################
            #  Save latent informations
            positions_x[t] = x_t
            positions_y[t] = y_t
            dx[t] = direction_t[0]
            dy[t] = direction_t[1]

            ########################
            #  Draw frame
            if shape == 'square':
                movie[x_t - half_size_t:x_t + half_size_t, y_t - half_size_t:y_t + half_size_t, 0, t] = colour[0]
                movie[x_t - half_size_t:x_t + half_size_t, y_t - half_size_t:y_t + half_size_t, 1, t] = colour[1]
                movie[x_t - half_size_t:x_t + half_size_t, y_t - half_size_t:y_t + half_size_t, 2, t] = colour[2]

            ########################
            #  Get new parameters at t+1
            if self.random_colour:
                colour = self.colour_map[random.sample(self.colours, 1)[0]]

            ########################
            #  Get new direction at t+1
            if not self.random_position:
                left_tp1 = x_t + direction_t[0] - half_size_t
                right_tp1 = x_t + direction_t[0] + half_size_t
                up_tp1 = y_t + direction_t[1] - half_size_t
                down_tp1 = y_t + direction_t[1] + half_size_t
                if left_tp1 <= 0:
                    direction_x = -direction_t[0]
                    x_tp1 = -left_tp1 + half_size_t
                elif right_tp1 >= self.width:
                    direction_x = -direction_t[0]
                    x_tp1 = self.width - (right_tp1 - self.width) - half_size_t
                else:
                    direction_x = direction_t[0]
                    x_tp1 = x_t + direction_x

                if up_tp1 <= 0:
                    direction_y = -direction_t[1]
                    y_tp1 = - up_tp1 + half_size_t
                elif down_tp1 >= self.height:
                    direction_y = -direction_t[1]
                    y_tp1 = self.height - (down_tp1 - self.height) - half_size_t
                else:
                    direction_y = direction_t[1]
                    y_tp1 = y_t + direction_y

                size_t = size_t
                half_size_t = size_t // 2
                x_t = x_tp1
                y_t = y_tp1
                direction_t = (direction_x, direction_y)
            else:
                x_t = random.randint(initial_size // 2, self.width - initial_size // 2)
                y_t = random.randint(initial_size // 2, self.width - initial_size // 2)
                direction_t = (0, 0)

        movie_norm = self.normalise(movie)
        return {
            'movie': movie_norm,
            'shape': shape,
            'colour': colour,
            'background_color': background_colour,
            'initial_size': initial_size,
            'positions_x': positions_x,
            'positions_y': positions_y,
            'dx': dx,
            'dy': dy
        }

    @staticmethod
    def normalise(movie):
        movie_norm = (movie / 255.) * 2 - 1
        return movie_norm

    @staticmethod
    def unnormalise(movie):
        movie_unnorm = (movie + 1) / 2
        return movie_unnorm

    def data_loaders(self, batch_size, num_workers):
        train_dl = data.DataLoader(
            self,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

        val_dl = data.DataLoader(
            self,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

        eval_dl = data.DataLoader(
            self,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
        return train_dl, val_dl, eval_dl

    def visualise_batch(self, movies, writing_dir):
        batch_dim, _, _, _, duration = movies.shape

        movies = self.unnormalise(movies)

        if not os.path.isdir(writing_dir):
            os.makedirs(writing_dir)

        for batch_ind in range(batch_dim):
            for time in range(duration):
                filepath = f'{writing_dir}/{batch_ind}_{time}.png'
                movie_t = movies[batch_ind, :, :, :, time].numpy()
                scale = int(320 // self.width)
                matplotlib.image.imsave(filepath, self.scaler(movie_t, scale=scale))

    @staticmethod
    def scaler(data, scale):
        # increases the size of an image bfore plotting
        new_data = np.zeros((data.shape[0] * scale, data.shape[1] * scale, 3))
        for j in range(data.shape[0]):
            for k in range(data.shape[1]):
                new_data[j * scale: (j + 1) * scale, k * scale: (k + 1) * scale] = data[j, k]
        return new_data


if __name__ == '__main__':
    # width = 32
    # height = 32
    # duration = 10
    # latent_features = {
    #     'shape': False,
    #     'colour': True,
    #     'background_colour': True,
    #     'size': False,hierarchy_levels
    #     'size_growth': False,
    # }
    # dataset = DSpriteMovieDataset(height=height,
    #                               width=width,
    #                               duration=duration,
    #                               latent_features=latent_features
    #                               )
    # movies = []
    # for batch_ind in range(100):
    #     movies.append(dataset.__getitem__())
    # movies = torch.stack(movies)
    # dataset.visualise_batch(movies, '/home/leo/Recherche/Code/DatasetManager/DatasetManager/dump/dSprite_movie')

    width = 32
    height = 32
    duration = 1
    latent_features = {
        'shape': False,
        'colour': False,
        'background_colour': False,
        'size': False,
        'size_growth': False,
    }
    dataset = DSpriteMovieDataset(height=height,
                                  width=width,
                                  duration=duration,
                                  latent_features=latent_features,
                                  random_position=False,
                                  random_colour=False,
                                  )
    dl, _, _ = dataset.data_loaders(batch_size=4, num_workers=0)
    movies = []
    for batch in islice(dl, 20):
        movie = batch['movie']
        movies.append(movie)
    movies = torch.cat(movies, 0)
    dataset.visualise_batch(movies, '/home/gaetan/Downloads/tmp')
