from typing import Iterable, Mapping, Type, Callable, Iterator, Optional

from lmarshal.marshal_config import MarshalConfig
from lmarshal.marshaler import Marshaler
from lmarshal.types import PostDemarshalReferenceSetter, PostDemarshalListener, PostDemarshalSetter


class Demarshaler:
    """
    + construct empty elements (empty constructor)
    + set elements / references (setter function)
    + call post demarshal initializers (post demarshal hook)


    """

    def __init__(self, config: MarshalConfig) -> None:
        self._config: MarshalConfig = config
        self._reference_index: {str: any} = {}
        self._setters: [(PostDemarshalSetter, str)] = []
        self._initializers: [PostDemarshalListener] = []

    def demarshal(self, target: any) -> any:

        # recursively demarshal target
        result, is_reference = self._demarshal(target)
        if is_reference:
            raise ValueError('Demarshaling a bare reference.')

        # dereference deferred references
        reference_index = self._reference_index
        for setter, label in self._setters:
            if label not in reference_index:
                raise ValueError(f'Undefined reference {label}.')
            setter(reference_index[label])

        # initialize elements
        for initializer in self._initializers:
            initializer()

        return result

    def _demarshal(self, target: any) -> (any, bool):
        target_type = type(target)
        demarshaler_type_map = self._config.demarshaler_type_map
        if target_type not in demarshaler_type_map:
            raise ValueError(f'Type has undefined demarshaling protocol: "{target_type}".')
        return demarshaler_type_map[target_type](self, target)

    def demarshal_passthrough(self, target: any) -> any:
        return target

    def demarshal_string(self, target: str) -> str:
        if target.startswith(self._config.escape_prefix):
            return self.unescape_string(target)
        return target

    def unescape_string(self, target: str) -> str:
        return target[1:]

    def demarshal_list(self, target: Iterable) -> []:
        return [self._demarshal(e) for e in target]

    def demarshal_dict_or_typed_object(self, target: Mapping) -> any:
        if self._config.type_key not in target:
            return self.demarshal_dict(target)  # demarshal normal dict
        # demarshal typed objects
        type_code = target[self._config.type_key]
        type_code_map = self._config.demarshaler_type_code_map
        if type_code not in type_code_map:
            raise ValueError(f'Unknown type code "{type_code}".')
        return type_code_map[type_code](target)

    # def demarshal_dict(self, target: Mapping):
    #     if self._config.flat_dict_key in target:
    #         if len(target) > 1:
    #             raise ValueError('Demarshaling a flat dict with more than one key.')
    #         return self.demarshal_flat_dict(target[self._config.flat_dict_key])  # handle nonconformant keys
    #     # TODO: optionally allow referenced strings as keys?
    #     return {self.demarshal_key(k): self._demarshal(v) for k, v in sorted(target.items())}

    def demarshal_key(self, target: str) -> str:
        if target.startswith(self._config.escape_prefix):
            return self.unescape_string(target)
        return target

    # def demarshal_flat_dict(self, target: list):
    #     demarshaled = self._demarshal(target)
    #     if not isinstance(demarshaled, Iterable):
    #         raise ValueError(f'Found a non-Iterable {type(demarshaled)} while demarshaling a flat dict.')
    #     result = {}
    #     for kvp in demarshaled:
    #         if not isinstance(kvp, list) or len(kvp) != 2:
    #             raise ValueError('Expected a Key-Value pair list of length 2, but found something else.')
    #         result[kvp[0]] = kvp[1]
    #     return result

    def demarshal_referencable(self, target: any) -> (any, bool):
        reference_index = self._reference_index
        reference_prefix = self._config.reference_prefix
        if isinstance(target, str) and target.startswith(reference_prefix):  # if target is a reference:
            label = self.demarshal_reference(target)
            if label in reference_index:
                return reference_index[label], False  # if reference is demarshaled, just return it
            return label, True  # if result is not demarshaled yet, return the label for deferred dereferencing

        # if target is not a reference, demarshal and index it
        label_key = self._config.label_key
        label = target[label_key] if isinstance(target, Mapping) and label_key in target else \
            Marshaler.make_implicit_label(len(reference_index))
        result = self._demarshal(target)
        reference_index[label] = result
        return result, False

    def demarshal_reference(self, target: str) -> any:
        return target[len(self._config.reference_prefix):]

    def dereference(self, label: str) -> any:
        reference_index = self._reference_index
        if label not in reference_index:
            raise ValueError(f'Undefined reference {label}.')
        return reference_index[label]

    def demarshal_dict_items(self, target: Mapping) -> Iterator[(any, any, bool)]:
        item_iterator = None
        flat_dict_key = self._config.flat_dict_key
        if flat_dict_key in target:
            if len(target) > 1:
                raise ValueError('Demarshaling a flattened dict with more than one key.')
            item_iterator = ((kvp[0], kvp[1]) for kvp in target[flat_dict_key])
        else:
            item_iterator = sorted(target.items())
        return ((self.demarshal_key(k), *self._demarshal(v)) for k, v in item_iterator)

    def demarshal_flat_dict_items(self, target: any) -> Iterator[(any, any, bool)]:
        demarshaled, is_reference = self._demarshal(target)
        if is_reference or not isinstance(demarshaled, Iterator):
            raise ValueError(f'Found a non-Iterator {type(demarshaled)} while demarshaling a flat dict.')
        for kvp, is_reference in demarshaled:
            if not isinstance(kvp, list) or len(kvp) != 2:
                raise ValueError(
                    'Expected a list of length 2 while demarshaling a flat dict, but found something else.')
            key, key_is_reference = self.demarshal_key()
            yield kvp[0], *kvp[1]

    def demarshal_recursively(
            self,
            target: any,
            factory: Callable[[], any],
            element_demarshaler: Callable[[any], Iterator[(any, any, bool)]],
            element_setter: Callable[[any, any, any], any],
            initializer: Optional[Callable[[], any]],
    ) -> (any, bool):
        result = factory()
        for key, value, is_reference in element_demarshaler(target):
            if is_reference:
                self._setters.append((element_setter, value))
            else:
                element_setter(result, key, value)

        if initializer is not None:
            self._initializers.append(initializer)
        return result, False

    def demarshal_list(self, target: Iterable, result_type: Type = list) -> any:

        def setter(result, key, value):
            while len(result) < key:
                result.append(None)
            result.insert(key, value)

        return self.demarshal_recursively(
            target,
            result_type.__call__,
            lambda t: ((k, *self._demarshal(v)) for k, v in enumerate(t)),
            setter,
            lambda result: None,
        )

    def demarshal_dict(self, target: Mapping, result_type: Type = dict) -> any:
        def setter(result, key, value):
            result[key] = value

        return self.demarshal_recursively(
            target,
            result_type.__call__,
            lambda t: ((self.demarshal_key(k), *self._demarshal(v)) for k, v in self.demarshal_dict_items(t)),
            setter,
            lambda result: None,
        )

    def demarshal_object(self, target: Mapping, result_type: Type) -> any:
        def setter(result, key, value):
            result[key] = value

        return self.demarshal_recursively(
            target,
            result_type.__new__,
            lambda t: ((self.demarshal_key(k), *self._demarshal(v)) for k, v in self.demarshal_dict_items(t)),
            setter,
            lambda result: None,
        )

        result = result_type.__new__(result_type)
        for key, value in self.demarshal_dict(target):
            self.enqueue_delayed_setter(key, lambda dereferenced: result.__setattr__(key, dereferenced))
        return result

    def enqueue_delayed_setter(self, label: str, setter: PostDemarshalReferenceSetter) -> None:
        self._post_demarshal_reference_setters.append((label, setter))
