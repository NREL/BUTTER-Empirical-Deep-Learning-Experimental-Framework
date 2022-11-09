

from dmp.structure.network_module import NetworkModule


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

    def __init__(self, target: NetworkModule) -> None:
        self._inputs: list = []
        self._nodes: dict = {}
        self._outputs: list = []

        output = self._visit(target)
        self._outputs = [output]

    def __call__(self) -> Tuple[list, Any]:
        return self._inputs, self._outputs

    def _visit(self, target: NetworkModule) -> Any:
        if target not in self._nodes:
            keras_inputs = [self._visit(i) for i in target.inputs]
            self._nodes[target] = self._visit_raw(target, keras_inputs)
        return self._nodes[target]

    @singledispatchmethod
    def _visit_raw(self, target, keras_inputs) -> Any:
        raise NotImplementedError(
            'Unsupported module of type "{}".'.format(type(target)))

    @_visit_raw.register
    def _(self, target: NInput, keras_inputs) -> Any:
        result = Input(shape=target.shape)
        self._inputs.append(result)
        return result

    @_visit_raw.register
    def _(self, target: NDense, keras_inputs) -> Any:

        kernel_regularizer = make_regularizer(target.kernel_regularizer)
        bias_regularizer = make_regularizer(target.bias_regularizer)
        activity_regularizer = make_regularizer(
            target.activity_regularizer)

        return Dense(
            target.shape[0],
            activation=target.activation,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_initializer=target.kernel_initializer,
        )(*keras_inputs)

    @_visit_raw.register
    def _(self, target: NAdd, keras_inputs) -> Any:
        return tensorflow.keras.layers.add(keras_inputs)

