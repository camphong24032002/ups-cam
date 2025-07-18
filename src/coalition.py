from abc import ABC, abstractmethod
import numpy as np
import torch


def spaced_elements(array, num_elems=4):
    return [x[len(x)//2] for x in np.array_split(np.array(array), num_elems)]


class AbstractPlayerIterator(ABC):

    def __init__(self, input, list_indices=[], random=False):
        self._assert_input_compatibility(input)
        self.input_shape = input.shape
        self.random = random
        self.n_players = self._get_number_of_players_from_shape()
        self.permutation = torch.Tensor(list(set(range(self.n_players)) - set(list_indices))).to(torch.int32)
        # print(list_indices)
        # print(self.permutation)
        self.n_players -=  len(list_indices)
        mask_input = torch.zeros(self.input_shape, dtype=torch.int32)
        mask = torch.zeros(self._input_shape_merged(), dtype=torch.float32)
        self.mask_input = mask_input
        self.mask = mask
        if random:
            self.permutation = self.permutation[torch.randperm(self.permutation.size(0))]

        self.i = 0
        self.kn = self.n_players
        self.ks = spaced_elements(range(self.n_players), self.kn)

    def set_n_steps(self, steps):
        self.kn = steps
        self.ks = spaced_elements(range(self.n_players), self.kn)

    def get_number_of_players(self):
        return self.n_players

    def get_explanation_shape(self):
        return self.input_shape

    def get_coalition_size(self):
        return 1

    def get_steps_list(self):
        return self.ks

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i == self.n_players:
            raise StopIteration
        m = self._get_masks_for_index(self.i)
        self.i = self.i + 1
        return m

    @abstractmethod
    def _assert_input_compatibility(self, input):
        pass

    @abstractmethod
    def _get_masks_for_index(self, i):
        pass

    @abstractmethod
    def _get_number_of_players_from_shape(self):
        pass


class ImagePlayerIterator(AbstractPlayerIterator):
    def __init__(self, input_tensor, list_indices=[], random=False, window_shape=(1, 1, 1)):
        self.window_shape = window_shape
        assert self.window_shape is not None, "window_shape cannot be None"
        assert len(self.window_shape) == 3, "window_shape must contain 3 elements"

        # In PyTorch, input tensors have shape (batch_size, channels, height, width)
        # We exclude the batch dimension for input shape
        self.input_shape = input_tensor.shape

        # Adjusted assertions for PyTorch tensor shapes
        assert 1 <= self.window_shape[0] <= self.input_shape[0], \
            "last dimension of window_shape must be in range 1..n_input_channels"

        assert self.window_shape[0] == self.input_shape[0] or self.window_shape[0] == 1, \
            "last element of window_shape must be 1 or equal to the number of channels of the input"

        assert self.input_shape[1] % self.window_shape[1] == 0 and self.input_shape[2] % self.window_shape[2] == 0, \
            "input dimensions must be multiples of window_shape dimensions"

        super(ImagePlayerIterator, self).__init__(input_tensor, list_indices, random)

    def _input_shape_merged(self):
        shape = list(self.input_shape)
        if self.window_shape[0] > 1:
            # Merge the channels dimension if window_shape[-1] > 1
            shape[0] = 1  # Channels dimension
        return shape

    def _assert_input_compatibility(self, input_tensor):
        assert input_tensor.dim() == 3, 'ImagePlayerIterator requires an input with 3 dimensions'

    def _get_number_of_players_from_shape(self):
        shape = self._input_shape_merged()
        if self.window_shape[1] > 1:
            shape[1] = shape[1] // self.window_shape[1]  # Height dimension
        if self.window_shape[2] > 1:
            shape[2] = shape[2] // self.window_shape[2]  # Width dimension

        nplayers = int(np.prod(shape))
        return nplayers

    def _get_masks_for_index(self, i):
        mask_input = self.mask_input.clone()
        mask = self.mask.clone()
        i = self.permutation[i]

        # Compute number of rows and columns
        nrows = self.input_shape[1] // self.window_shape[1]
        ncols = self.input_shape[2] // self.window_shape[2]
        row_step = self.window_shape[1]
        col_step = self.window_shape[2]
        coalition_size = row_step * col_step

        # Compute row and column indices
        row = i // nrows
        col = i % ncols

        # Set mask values
        row_start = int(row * row_step)
        row_end = int((row + 1) * row_step)
        col_start = int(col * col_step)
        col_end = int((col + 1) * col_step)

        mask_input[:, row_start:row_end, col_start:col_end] = 1
        mask[:, row_start:row_end, col_start:col_end] = 1.0 / coalition_size

        return mask_input, mask

    def get_explanation_shape(self):
        return self._input_shape_merged()
