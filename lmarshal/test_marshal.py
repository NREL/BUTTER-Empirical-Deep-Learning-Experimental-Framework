import json
import pickle

import pytest

from dmp.experiment.structure.n_add import NAdd
from dmp.experiment.structure.n_dense import NDense
from dmp.experiment.structure.n_input import NInput
from lmarshal.marshal import Marshal
from lmarshal.marshal_config import MarshalConfig


@pytest.mark.parametrize("type_key", ['', '%', 'type code'])
@pytest.mark.parametrize("label_key", ['&', '*', 'label'])
@pytest.mark.parametrize("reference_prefix", ['*', '&', 'ref-', ])
@pytest.mark.parametrize("escape_prefix", ['!', '__', '\\'])
@pytest.mark.parametrize("flat_dict_key", [':', 'flat', ' '])
@pytest.mark.parametrize("label_all", [False, True])
@pytest.mark.parametrize("label_referenced", [False, True])
@pytest.mark.parametrize("circular_references_only", [False, True])
@pytest.mark.parametrize("reference_strings", [False, True])
def test_network_json_serializer(
        type_key,
        label_key,
        reference_prefix,
        escape_prefix,
        flat_dict_key,
        label_all,
        label_referenced,
        circular_references_only,
        reference_strings,
):
    marshal = Marshal(MarshalConfig(
        type_key=type_key,
        label_key=label_key,
        reference_prefix=reference_prefix,
        escape_prefix=escape_prefix,
        flat_dict_key=flat_dict_key,
        label_all=label_all,
        label_referenced=label_referenced,
        circular_references_only=circular_references_only,
        reference_strings=reference_strings))

    layers = [NInput(0), NDense(1), NAdd(2), NDense(3), NAdd(4)]
    layers[1].inputs = [layers[0]]
    layers[2].inputs = [layers[0], layers[1]]
    layers[3].inputs = [layers[2]]
    layers[4].inputs = [layers[2], layers[3]]
    output = layers[-1]

    marshal.register_type(NInput)
    marshal.register_type(NDense)
    marshal.register_type(NAdd)

    check_first = not circular_references_only
    layers.append(layers)
    for e in layers:
        check_marshaling(marshal, e, check_first, True, False, False)

    if not circular_references_only:
        l = []
        bar = []
        e = {}
        s = set()
        x = [e, s, e, s]
        x.append(x)
        foo = [0, 1, 2, bar]
        bar.extend([foo, l, bar, l])

        t = (
            None, True, False, 0, 1, -1.2, {}, [],
            ['a', 'b', 'c'], {'a', 'b', 1}, {'a': 0, 'b': 'b', 'c': 'cee'},
            {0: 'a', 2: 'two', None: 'three', 'four': None},
            l, bar, e, s, x, foo)
        l.extend([foo, t])

        elements = [l, bar, e, s, x, foo, t]
        elements.append(elements)
        for e in elements:
            check_marshaling(marshal, e, check_first, False, False, True)


def check_marshaling(marshal, target, check_first, check_strings, check_equality, check_pickle):
    marshaled = marshal.marshal(target)
    demarshaled = marshal.demarshal(marshaled)
    remarshaled = marshal.marshal(demarshaled)
    demarshaled_again = marshal.demarshal(remarshaled)
    marshaled_again = marshal.marshal(demarshaled_again)

    if check_equality:
        if check_first:
            assert target == demarshaled
        assert demarshaled_again == demarshaled

    if check_strings:
        second = json.dumps(remarshaled, sort_keys=True, separators=(',', ':'))
        if check_first:
            first = json.dumps(marshaled, sort_keys=True, separators=(',', ':'))
            # print(f'first  {first}')
            assert first == second
        # print(f'second {second}')
        third = json.dumps(marshaled_again, sort_keys=True, separators=(',', ':'))
        # print(f'third  {third}')
        assert second == third

    if check_pickle:
        if check_first:
            assert pickle.dumps(target) == pickle.dumps(demarshaled)
        assert pickle.dumps(demarshaled_again) == pickle.dumps(demarshaled)
