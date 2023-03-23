import sys
sys.path.insert(0, './')

import json
import pickle

import pytest

import dmp.layer.layer as layer
from lmarshal.src.marshal import Marshal
from lmarshal.src.marshal_config import MarshalConfig


@pytest.mark.parametrize("type_key", ['', '%', 'type code', 'type'])
@pytest.mark.parametrize("label_key", ['&', '*', 'label'])
@pytest.mark.parametrize("reference_prefix", [
    '*',
    '&',
    'ref-',
])
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
    marshal = Marshal(
        MarshalConfig(type_key=type_key,
                      label_key=label_key,
                      reference_prefix=reference_prefix,
                      escape_prefix=escape_prefix,
                      flat_dict_key=flat_dict_key,
                      label_all=label_all,
                      label_referenced=label_referenced,
                      circular_references_only=circular_references_only,
                      reference_strings=reference_strings))

    layers = [
        layer.Input({}, []),
        layer.Dense({}, []),
        layer.Add({}, []),
        layer.Dense({}, []),
        layer.Add({}, []),
    ]
    layers[1].inputs = [layers[0]]
    layers[2].inputs = [layers[0], layers[1]]
    layers[3].inputs = [layers[2]]
    layers[4].inputs = [layers[2], layers[3]]

    marshal.register_type(layer.Input)
    marshal.register_type(layer.Dense)
    marshal.register_type(layer.Add)

    check_first = not circular_references_only
    layers.append(layers)
    for e in layers:
        check_marshaling(marshal, e, check_first, True, False, False)

    if check_first:
        l = []
        bar = []
        e = {}
        s = set()
        x = [e, s, e, s]
        x.append(x)
        foo = [0, 1, 2, bar]
        bar.extend([foo, l, bar, l])

        t = (None, True, False, 0, 1, -1.2, {}, [], ['a', 'b', 'c'],
             {'a', 'b', 1}, {
                 'a': 0,
                 'b': 'b',
                 'c': 'cee'
             }, ['', '!', '%', '&', '*', ':'], {
                 0: 'a',
                 2: 'two',
                 None: 'three',
                 'four': None
             }, l, bar, e, s, x, foo)
        l.extend([foo, t])

        elements = [l, bar, e, s, x, foo, t]
        elements.append(elements)
        for e in elements:
            check_marshaling(marshal, e, check_first, False, False, True)

    t = {'': '', '!': '!', '%': '%', '&': '&', '*': '*', ':': ':'}
    t['self'] = t
    check_marshaling(marshal, t, check_first, True, False, False)


def check_marshaling(marshal, target, check_first, check_strings,
                     check_equality, check_pickle):
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
            first = json.dumps(marshaled,
                               sort_keys=True,
                               separators=(',', ':'))
            assert first == second
        third = json.dumps(marshaled_again,
                           sort_keys=True,
                           separators=(',', ':'))
        assert second == third
    # print(f"first  {json.dumps(marshaled, sort_keys=True, separators=(',', ':'))}")
    # print(f"second {json.dumps(remarshaled, sort_keys=True, separators=(',', ':'))}")
    # print(f"third  {json.dumps(marshaled_again, sort_keys=True, separators=(',', ':'))}")

    if check_pickle:
        if check_first:
            assert pickle.dumps(target) == pickle.dumps(demarshaled)
        assert pickle.dumps(demarshaled_again) == pickle.dumps(demarshaled)
