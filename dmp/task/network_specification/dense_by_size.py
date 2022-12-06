from dataclasses import dataclass
from typing import Any
from dmp.task.dataset import Dataset

from dmp.task.network_specification.network_specification import NetworkSpecification
from dmp.task.training_experiment.training_experiment_utils import find_best_layout_for_budget_and_depth
from dmp.layer import *


@dataclass
class DenseBySize(NetworkSpecification):
    shape: str  # (migrate)
    size: int  # (migrate)
    depth: int  # (migrate)
    # input_layer : dict
    #input_layer.activation (migrate from input_activation)
    layer: dict
    '''
        + activation (migrate)
        + kernel_regularizer (migrate)
        + bias_regularizer (migrate)
        + activity_regularizer (migrate)
        + batch_norm (new?)
    '''
    output_layer: dict
    residual : str # (migrate from shape)
    

    # activation (migrate from runtime compute of output_activation)

    def make_network(self, dataset: Dataset):
        # TODO: make it so we don't need this hack
        shape = self.shape
        # residual_mode = 'none'
        # residual_suffix = '_residual'
        # if shape.endswith(residual_suffix):
        #     residual_mode = 'full'
        #     shape = shape[0:-len(residual_suffix)]

        # delta, widths, network_structure, num_free_parameters, layer_shapes = \
        #     find_best_layout_for_budget_and_depth(
        #         dataset.input_shape,
        #         residual_mode,
        #         # task.input_activation,
        #         self.output_layer,
        #         target_size,
        #         widths_factory(shape)(dataset.output_shape[1], task.depth),
        #         layer_args,
        #     )

        widths_factory = widths_factory(self.shape)
        delta, widths, network_structure, num_free_parameters, layer_shapes = \
            find_best_layout_for_budget_and_depth(
                self.size,
                lambda search_parameter : self.make_network_from_widths(
                    widths_factory(shape)(dataset.output_shape[1], self.depth))
            )

    # TODO: wrap make_network_from_widths up with shape to width function and have it call find_best_layout_for_budget_and_depth above

    def make_network_from_widths(
        self,
        widths: List[int],
    ) -> Layer:
        parent = Input({'shape': self.input_shape}, [])
        # Loop over depths, creating layer from "current" to "layer", and iteratively adding more
        for depth, width in enumerate(widths):
            if depth == len(widths) - 1:
                layer_config = self.output_layer
            else:
                layer_config = self.layer
            layer_config = layer_config.copy()

            # Fully connected layer
            layer_config['units'] = width
            current_layer = Dense(layer_config, parent)

            # Skip connections for residual modes
            residual_mode = self.residual
            if residual_mode == 'none':
                pass
            elif residual_mode == 'full':
                # If this isn't the first or last layer, and the previous layer is
                # of the same width insert a residual sum between layers
                # NB: Only works for rectangle
                if depth > 0 and depth < len(widths) - 1 and \
                    width == widths[depth - 1]:
                    current_layer = Add({}, [current_layer, parent])
            else:
                raise NotImplementedError(
                    f'Unknown residual mode "{residual_mode}".')
                    
            parent = current_layer
        return parent

    def widths_factory(self, shape:str) -> Callable[[float], List[int]]:

        if shape == 'rectangle':
            return get_rectangular_widths
        elif shape == 'trapezoid':
            return get_trapezoidal_widths
        elif shape == 'exponential':
            return get_exponential_widths
        elif shape == 'wide_first_2x':
            return get_wide_first_2x
        elif shape == 'wide_first_4x':
            return get_wide_first_4x
        elif shape == 'wide_first_5x':
            return get_wide_first_5x
        elif shape == 'wide_first_8x':
            return get_wide_first_8x
        elif shape in {'wide_first', 'wide_first_10x'}:
            return get_wide_first_layer_rectangular_other_layers_widths
        elif shape == 'wide_first_16x':
            return get_wide_first_16x
        elif shape == 'wide_first_20x':
            return get_wide_first_20x
        else:
            assert False, 'Shape "{}" not recognized.'.format(shape)

def get_wide_first_layer_rectangular_other_layers_widths(
    num_outputs: int,
    depth: int,
    first_layer_width_multiplier: float = 10,
) -> Callable[[float], List[int]]:

    def make_layout(search_parameter):
        layout = []
        if depth > 1:
            layout.append(search_parameter)
        if depth > 2:
            inner_width = max(1, int(round(search_parameter / first_layer_width_multiplier)))
            layout.extend((inner_width for k in range(0, depth - 2)))
        layout.append(num_outputs)
        return layout

    return make_layout


def get_wide_first_2x(num_outputs: int,
                      depth: int) -> Callable[[float], List[int]]:
    return get_wide_first_layer_rectangular_other_layers_widths(
        num_outputs, depth, 2)


def get_wide_first_4x(num_outputs: int,
                      depth: int) -> Callable[[float], List[int]]:
    return get_wide_first_layer_rectangular_other_layers_widths(
        num_outputs, depth, 4)


def get_wide_first_8x(num_outputs: int,
                      depth: int) -> Callable[[float], List[int]]:
    return get_wide_first_layer_rectangular_other_layers_widths(
        num_outputs, depth, 8)


def get_wide_first_16x(num_outputs: int,
                       depth: int) -> Callable[[float], List[int]]:
    return get_wide_first_layer_rectangular_other_layers_widths(
        num_outputs, depth, 16)


def get_wide_first_5x(num_outputs: int,
                      depth: int) -> Callable[[float], List[int]]:
    return get_wide_first_layer_rectangular_other_layers_widths(
        num_outputs, depth, 5)


def get_wide_first_20x(num_outputs: int,
                       depth: int) -> Callable[[float], List[int]]:
    return get_wide_first_layer_rectangular_other_layers_widths(
        num_outputs, depth, 20)