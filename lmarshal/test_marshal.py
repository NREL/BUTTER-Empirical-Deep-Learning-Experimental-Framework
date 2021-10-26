import json

import pytest

from dmp.experiment.structure.n_add import NAdd
from dmp.experiment.structure.n_dense import NDense
from dmp.experiment.structure.n_input import NInput
from lmarshal.marshal import Marshal
from lmarshal.marshal_config import MarshalConfig


@pytest.mark.parametrize("type_key", ['%', 't', 'type code'])
@pytest.mark.parametrize("label_key", ['&', '*', 'label'])
@pytest.mark.parametrize("reference_prefix", ['*', '&', 'ref-',])
@pytest.mark.parametrize("escape_prefix", ['!', '\\', 'esc', '_', '__', '\n'])
@pytest.mark.parametrize("flat_dict_key", [':', 'flat', ''])
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

    check_first = not (label_all and circular_references_only)

    check_marshaling(marshal, output, check_first)
    check_marshaling(marshal, layers, check_first)
    for e in layers:
        check_marshaling(marshal, e, check_first)


def check_marshaling(marshal, target, check_first):
    marshaled = marshal.marshal(target)
    first = json.dumps(marshaled, sort_keys=True, separators=(',', ':'))
    demarshaled = marshal.demarshal(marshaled)
    remarshaled = marshal.marshal(demarshaled)
    second = json.dumps(remarshaled, sort_keys=True, separators=(',', ':'))
    if check_first:
        assert first == second
    demarshaled = marshal.demarshal(remarshaled)
    marshaled = marshal.marshal(demarshaled)
    third = json.dumps(marshaled, sort_keys=True, separators=(',', ':'))
    assert second == third
