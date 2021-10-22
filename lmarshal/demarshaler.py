from typing import Iterable, Mapping, Hashable, MutableMapping, MutableSequence, Union

from lmarshal.marshal_config import MarshalConfig
from lmarshal.marshaler import Marshaler
from lmarshal.types import PostDemarshalReferenceSetter, PostDemarshalListener


class Demarshaler:
    """
    + construct empty elements (empty constructor)
    + set elements / references (setter function)
    + call post demarshal initializers (post demarshal hook)


    """
    def __init__(self, config: MarshalConfig) -> None:
        self._config: MarshalConfig = config
        self._reference_index: {str: any} = {}
        self._post_demarshal_reference_setters: [(str, PostDemarshalReferenceSetter)] = []
        self._post_demarshal_listeners: [PostDemarshalListener] = []

    def demarshal(self, target: any) -> any:
        result = self._demarshal(target)  # demarshal target
        for label, setter in self._post_demarshal_reference_setters:  # set delayed references
            setter(self.dereference(label))
        for listener in self._post_demarshal_listeners:  # notify post-demarshal listeners
            listener()
        return result

    def _demarshal(self, target: any) -> any:
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

    def demarshal_dict(self, target: Mapping):
        if self._config.flat_dict_key in target:
            if len(target) > 1:
                raise ValueError('Demarshaling a flat dict with more than one key.')
            return self.demarshal_flat_dict(target[self._config.flat_dict_key])  # handle nonconformant keys
        # TODO: optionally allow referenced strings as keys?
        return {self.demarshal_key(k): self._demarshal(v) for k, v in sorted(target.items())}

    def demarshal_key(self, target: str) -> str:
        if target.startswith(self._config.escape_prefix):
            return self.unescape_string(target)
        return target

    def demarshal_flat_dict(self, target: list):
        demarshaled = self._demarshal(target)
        if not isinstance(demarshaled, Iterable):
            raise ValueError(f'Found a non-Iterable {type(demarshaled)} while demarshaling a flat dict.')
        result = {}
        for kvp in demarshaled:
            if not isinstance(kvp, list) or len(kvp) != 2:
                raise ValueError('Expected a Key-Value pair list of length 2, but found something else.')
            result[kvp[0]] = kvp[1]
        return result

    def demarshal_referencable(self, target: any) -> any:
        reference_prefix = self._config.reference_prefix
        if isinstance(target, str) and target.startswith(reference_prefix):  # if target is a reference:
            label = self.demarshal_reference(target)
            result = self.try_dereference(label)
            if result is not None:  # if reference is demarshaled, return it
                return result
            # value is not yet demarshaled, so use delayed initialization
            return None

        # if target is not a reference, demarshal and index it
        label_key = self._config.label_key
        label = target[label_key] if isinstance(target, Mapping) and label_key in target else \
            Marshaler.make_implicit_label(len(self._reference_index))
        result = self._demarshal(target)
        self._reference_index[label] = result
        return result

    def demarshal_reference(self, target: str) -> any:
        return target[len(self._config.reference_prefix):]

    def try_dereference(self, label: str) -> any:
        return self._reference_index.get(label, None)

    def dereference(self, label: str) -> any:
        reference_index = self._reference_index
        if label not in reference_index:
            raise ValueError(f'Undefined reference {label}.')
        return reference_index[label]

    def post_demarshal_map(self, parent: Union[MutableMapping, MutableSequence], key: Hashable, value: any) -> None:
        parent[key] = value

    def post_demarshal_object(self, parent: object, key: str, value: any) -> None:
        parent.__setattr__(key, value)

    def listen(self, listener: PostDemarshalListener) -> None:
        self._post_demarshal_listeners.append(listener)
