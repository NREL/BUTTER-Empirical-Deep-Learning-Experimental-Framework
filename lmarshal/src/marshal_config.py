from typing import Iterable


class MarshalConfig:
    __slots__ = [
        'type_key',
        'label_key',
        'reference_prefix',
        'escape_prefix',
        'flat_dict_key',
        'tuple_type_code',
        'set_type_code',
        'label_all',
        'label_referenced',
        'circular_references_only',
        'reference_strings',
        'control_key_set',
        'control_prefix_set',
        'enum_value_key',
    ]

    def __init__(
        self,
        type_key: str = '',
        label_key: str = '&',
        reference_prefix: str = '*',
        escape_prefix: str = '!',
        flat_dict_key: str = ':',
        tuple_type_code: str = 't',
        set_type_code: str = 's',
        reserved_prefixes_and_keys: Iterable[str] = ('!', '@', '#', '$', '%',
                                                     '^', '&', '*', '=', '|',
                                                     '<', '>', '?', ';', ':'),
        label_all: bool = False,
        label_referenced: bool = True,
        circular_references_only: bool = False,
        reference_strings: bool = False,
        enum_value_key: str = 'v',
    ):

        self.type_key: str = type_key
        self.label_key: str = label_key
        self.reference_prefix: str = reference_prefix
        self.escape_prefix: str = escape_prefix
        self.flat_dict_key: str = flat_dict_key
        self.tuple_type_code: str = tuple_type_code
        self.set_type_code: str = set_type_code

        self.label_all: bool = label_all
        self.label_referenced: bool = label_referenced
        self.circular_references_only: bool = circular_references_only
        self.reference_strings: bool = reference_strings
        self.enum_value_key: str = enum_value_key

        if len(reference_prefix) <= 0 or len(escape_prefix) <= 0:
            raise ValueError('Prefixes are zero length.')

        self.control_key_set = {type_key, label_key, flat_dict_key}
        if len(self.control_key_set
               ) != 3 or escape_prefix in self.control_key_set:
            raise ValueError('Control and escape keys are not distinct.')
        self.control_key_set.update(reserved_prefixes_and_keys)
        if escape_prefix in self.control_key_set:
            self.control_key_set.remove(escape_prefix)

        self.control_prefix_set = {reference_prefix, escape_prefix}
        if len(self.control_prefix_set) != 2:
            raise ValueError('Control prefixes are not distinct.')
        self.control_prefix_set.update(reserved_prefixes_and_keys)
