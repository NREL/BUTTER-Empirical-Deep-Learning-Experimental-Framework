from dmp.layer.layer import Layer
import dmp.keras_interface.keras_keys as keras_keys

class Input(Layer):
    def marshal(self) -> dict:
        result = self.config.copy()
        shape = result.get(keras_keys.shape, None)
        if shape is not None:
            result[keras_keys.shape] = list(shape)
        return result

    def demarshal(self, flat: dict) -> None:
        shape = flat.get(keras_keys.shape, None)
        if shape is not None:
            flat[keras_keys.shape] = tuple(shape)
        self.__init__(flat)

