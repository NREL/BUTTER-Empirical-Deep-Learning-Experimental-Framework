from functools import singledispatchmethod
from typing import Any, Dict, Iterable, Set

import numpy
from dmp.structure.n_add import NAdd
from dmp.structure.n_input import NInput
from dmp.structure.n_dense import NDense
from dmp.structure.network_module import NetworkModule
import tensorflow.keras.layers as layers


class GrowNetworkVisitor:
    """
    Visitor that fills one network with the values from another. 
    If the destination network is larger, this will 'grow' the source into
    the destination.

    Old to old weights are retained.
    Old to new weights are left as-is
    New to new weights are left as-is
    New to old weights are scaled by the 'scale' parameter
    """

    def __init__(
        self,
        source: NetworkModule,
        source_node_layer_map: Dict[NetworkModule, layers.Layer],
        dest: NetworkModule,
        dest_node_layer_map: Dict[NetworkModule, layers.Layer],
        old_to_old_scale: float = 1.0,
        new_to_new_scale: float = 1.0,
        old_to_new_scale: float = 1.0,
        new_to_old_scale: float = 0.0,
        old_vs_new_blending: float = 0.0,
    ) -> None:
        self._source: NetworkModule = source
        self._source_node_layer_map: Dict[NetworkModule,
                                          layers.Layer] = source_node_layer_map
        self._dest: NetworkModule = dest
        self._dest_node_layer_map: Dict[NetworkModule,
                                        layers.Layer] = dest_node_layer_map

        self.old_to_old_scale: float = old_to_old_scale
        self.new_to_new_scale: float = new_to_new_scale
        self.old_to_new_scale: float = old_to_new_scale
        self.new_to_old_scale: float = new_to_old_scale
        self.old_vs_new_blending : float = old_vs_new_blending

        self._visited: Set[NetworkModule] = set()

        self._visit(source, dest)

    # def __call__(self) -> Tuple[list, Any]:
    #     return self._inputs, self._outputs

    def _visit(
        self,
        source_node: NetworkModule,
        dest_node: NetworkModule,
    ) -> None:
        if source_node in self._visited or dest_node in self._visited:
            return
        self._visited.add(source_node)
        self._visited.add(dest_node)

        self._do_visit(source_node, dest_node)

        for s, d in zip(source_node.inputs, dest_node.inputs):
            self._visit(s, d)

    @singledispatchmethod
    def _do_visit(
        self,
        source_node: NetworkModule,
        dest_node: NetworkModule,
    ) -> None:
        raise NotImplementedError('Unsupported module of type "{}".'.format(
            type(source_node)))

    @_do_visit.register
    def _(
        self,
        source_node: NInput,
        dest_node: NetworkModule,
    ) -> None:
        return

    @_do_visit.register
    def _(
        self,
        source_node: NDense,
        dest_node: NetworkModule,
    ) -> None:
        self._check_same_type_node(source_node, dest_node)

        source_layer: layers.Dense = \
            self._source_node_layer_map[source_node]  # type: ignore
        dest_layer: layers.Dense = \
            self._dest_node_layer_map[dest_node]  # type: ignore

        source_weights, source_biases = source_layer.get_weights(
        )  # type: ignore
        dest_weights, dest_biases = dest_layer.get_weights()  # type: ignore

        sw_shape = source_weights.shape
        sb_shape = source_biases.shape

        src_in_idx = sw_shape[0]
        src_out_idx = sw_shape[1]

        # scale old to new nodes
        dest_weights[:src_in_idx, src_out_idx:] = \
            self.scale_block(
            source_weights,
            dest_weights[:src_in_idx, src_out_idx:],
            self.old_to_new_scale,
            dest_weights[:src_in_idx, src_out_idx:])

        # scale new to old nodes
        dest_weights[src_in_idx:, :src_out_idx] = \
            self.scale_block(
            source_weights,
            dest_weights[src_in_idx:, :src_out_idx],
            self.new_to_old_scale,
            dest_weights[src_in_idx:, :src_out_idx])

        # scale new nodes
        dest_weights[src_in_idx:, src_out_idx:] = \
            self.scale_block(
            source_weights,
            dest_weights[src_in_idx:, src_out_idx:],
            self.new_to_new_scale,
            dest_weights[src_in_idx:, src_out_idx:])
        
        dest_biases[sb_shape[0]:] = \
            self.scale_block(
                source_biases,
                dest_biases[sb_shape[0]:],
                self.new_to_new_scale,
                dest_biases[sb_shape[0]:])
        
        # scale old nodes
        dest_weights[src_in_idx:, src_out_idx:] = \
            self.scale_block(
            source_weights,
            dest_weights[src_in_idx:, src_out_idx:],
            self.new_to_new_scale,
            dest_weights[src_in_idx:, src_out_idx:])
            
        blend = self.old_vs_new_blending
        dest_weights[:src_in_idx, :src_out_idx] = \
            source_weights * (1-blend) + \
            blend * dest_weights[:src_in_idx, :src_out_idx]
        dest_biases[:sb_shape[0]] = \
            source_biases * (1-blend) + \
            blend * dest_biases[:sb_shape[0]]

        

        # set old-to-old to be the same
        new_weights[0][:old_weights[0].shape[0], :old_weights[0].
                       shape[1], ] = old_weights[0]
        new_weights[1][:old_weights[1].shape[0]] = old_weights[1]

        # scale new-to-old weights
        new_weights[0] = self._scale * new_weights[0]
        new_weights[1] = self._scale * new_weights[1]

        dest_layer.set_weights(new_weights)

    @_do_visit.register
    def _(
        self,
        source_node: NAdd,
        dest_node: NetworkModule,
    ) -> None:
        return

    def _check_same_type_node(
        self,
        source_node: NetworkModule,
        dest_node: NetworkModule,
    ) -> None:
        if not isinstance(dest_node, NDense):
            raise TypeError(
                f'Source and destination NetworkModule are not the same type {type(source_node)} != {type(dest_node)}.'
            )

    def flatten(self, items):
        """Yield items from any nested iterable; see Reference."""
        for x in items:
            if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
                yield from self.flatten(x)
            else:
                yield x

    def scale_block(
        self,
        source_population,
        dest_population,
        scale_setting,
        target_block,
    )->None:
        if scale_setting is None \
                or target_block.size <= 0 \
            or source_population.size <= 0 \
                or dest_population.size <= 0:
            return target_block

        source_std = numpy.std(
            numpy.fromiter(self.flatten(source_population), dtype=float))
        new_std = numpy.std(
            numpy.fromiter(self.flatten(dest_population), dtype=float))
        # if numpy .isnan(new_std) or numpy .isnan(source_std):
        #     print(f'Numerical error in setting adjustment {source_std} {new_std} {source_population.size} {dest_population.size}')
        #     return target_block
        # print(f'Numerical error in setting adjustment {source_std} {new_std} {scale_setting} {source_population.size} {dest_population.size}')
        adjustment = (scale_setting * source_std) / new_std
        return target_block * adjustment
