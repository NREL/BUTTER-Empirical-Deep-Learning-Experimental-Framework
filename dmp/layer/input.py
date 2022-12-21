from dmp.layer.layer import Layer, register_layer_type

class Input(Layer):
    def marshal(self) -> dict:
        result = self.config.copy()
        shape = result.get('shape', None)
        if shape is not None:
            result['shape'] = list(shape)
        return result

    def demarshal(self, flat: dict) -> None:
        shape = flat.get('shape', None)
        if shape is not None:
            flat['shape'] = tuple(shape)
        self.__init__(flat)


register_layer_type(Input)