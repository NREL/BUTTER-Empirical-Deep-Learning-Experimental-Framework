from typing import Type, Iterable, Mapping, Union

from lmarshal.marshal import MarshallingInfo


class Marshaler:
    """
    {
        "__ref_id" : 1
        "__ref_ix": [0, 1, 1, 2, 3, 4] # which node numbers should be interpreted as references
    }
    """

    def __init__(self, marshaler_map: {Type: MarshallingInfo}) -> None:
        self._marshaler_map = marshaler_map
        self._vertex_index = {}  # if use_references else None
        self._vertex_visited_set = set()

    def marshal(self, target: any, referencing: bool) -> any:
        # marshal target into a plain object
        target_type = type(target)
        if target_type not in self._marshaler_map:
            raise ValueError(f'Type has undefined marshaling protocol: "{target_type}"')
        return self._marshaler_map[target_type](self, target, referencing)

    def marshal_object(self, target: any, referencing: bool, handler) -> any:
        target_id = id(target)

        # if use_references:
        if target_id not in self._vertex_index:
            self._vertex_index[target_id] = len(self._vertex_index)  # index target
        elif referencing:
            return self._vertex_index[target_id]  # if target is already indexed, return its reference index
        elif target_id in self._vertex_visited_set:  # detect cycles not elided by referencing
            raise ValueError('Circular reference detected')

        self._vertex_visited_set.add(target_id)  # add to cycle detection set
        marshaled = handler(self, target, referencing)  # marshal target ...
        self._vertex_visited_set.remove(target_id)  # remove from cycle detection set
        return marshaled

    @staticmethod
    def marshal_list(marshaler: 'Marshaler', target: Iterable, referencing: bool) -> list:
        return marshaler.marshal_object(
            target, referencing, lambda marshalled: [marshaler.marshal(e, referencing) for e in target])

    @staticmethod
    def marshal_dict(marshaler: 'Marshaler', target: Mapping, referencing: bool) -> Union[dict, int]:
        return marshaler.marshal_object(
            target,
            referencing,
            lambda marshalled: {
                k: marshaler.marshal(v, referencing)
                for k, v in sorted(marshalled.items())})

    @staticmethod
    def marshal_string(
            marshaler: 'Marshaler',
            target: str,
            referencing: bool,
    ) -> Union[str, int]:
        return marshaler.marshal_object(target, referencing, lambda marshalled: marshalled)

    @staticmethod
    def marshal_passthrough(marshaler: 'Marshaler', target: any, referencing: bool) -> any:
        return target

    @staticmethod
    def marshal_integer(marshaler: 'Marshaler', target: int, referencing: bool) -> int:
        if referencing:
            raise ValueError('Integer encountered while marshaling a referencing target')
        return target

    @staticmethod
    def marshal_dataclass():
        pass
