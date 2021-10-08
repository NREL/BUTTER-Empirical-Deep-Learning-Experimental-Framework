from typing import Callable, Optional, Hashable, Type

TypeCode = Hashable
ObjectFactory = Callable[['ObjectTranslator', {}], any]
ObjectConverter = Optional[Callable[['ObjectTranslator', any], {}]]


class ObjectTranslator:
    """
    internal map of object type to deserializer
        and object to serializer

    serializer:
        + switch on type
        + vars = vars(target).copy() (default)
        + vars[type] = type (always)
        + special function to handle graph fields
    default deserializer:
        + delete vars[type]
        + switch on type
        + type(**vars)
        + special function to handle graph fields
    """

    def __init__(
            self,
            type_field: str = 'type',
    ) -> None:
        self._type_field: str = type_field
        self._factory_map: {TypeCode: ObjectFactory} = {}
        self._converter_map: {TypeCode: ObjectConverter} = {}

    def register_type_code(
            self,
            type_code: TypeCode,
            factory: ObjectFactory = None,
            converter: ObjectConverter = None,
    ) -> None:
        self._factory_map[type_code] = factory
        self._converter_map[type_code] = converter

    def register_class(
            self,
            type: Type,
            type_code: Optional[TypeCode] = None,
            factory: Optional[ObjectFactory] = None,
            converter: Optional[ObjectConverter] = None,
    ) -> None:
        type_code = type.__name__ if type_code is None else type_code
        factory = self.make_default_factory(type) if factory is None else factory
        converter = self.make_default_converter(type) if converter is None else converter
        return self.register_type_code(type_code, factory, converter)

    @staticmethod
    def make_default_factory(type: Type) -> ObjectFactory:
        return lambda translator, source: type(**source)

    @staticmethod
    def make_default_converter(type: Type) -> ObjectConverter:
        return lambda translator, source: vars(source)


    def dict_to_object(self, source: {}) -> any:
        type_code = source[self._type_field]
        del source[self._type_field]
        handler = self._factory_map[type_code]
        return handler(self, source)


    def object_to_dict(self, source: any) -> {}:

        pass

    def graph_to_list(self, roots: [any]) -> [any]:
        """
        :param roots:
        :return:

        {
            roots: 3
                - if omitted, the first node is the root (backwards compatible)
            vertices : [...]
                - vertices in bottom-up / reverse DFS order? (only works for DAG)

            what fields are graph fields?
                - maybe all sub-objects?
                - maybe serializer/deserializer knows when to do this
        }

        """
        pass

    def list_to_graph(self, source: [any]) -> [any]:
        pass
