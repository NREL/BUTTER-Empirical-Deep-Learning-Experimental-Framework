from dataclasses import dataclass
from typing import Any, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple, Type, Union, get_args
from enum import Enum
import simplejson
import psycopg
from psycopg import sql
from dmp.marshaling import marshal
from pprint import pprint

# psycopg.extras.register_uuid()
# psycopg.extras.register_default_json(loads=simplejson.loads,
#                                       globally=True)  # type: ignore
# psycopg.extras.register_default_jsonb(loads=simplejson.loads,
#                                        globally=True)  # type: ignore
# psycopg.extensions.register_adapter(dict,
#                                      psycopg.extras.Json)  # type: ignore


class AttributeValueType(Enum):
    Null = (0, type(None), None, 'NULL')
    Bool = (1, bool, 'value_bool', 'boolean')
    Integer = (2, int, 'value_int', 'bigint')
    Float = (3, float, 'value_float', 'double precision')
    String = (4, str, 'value_str', 'text')
    JSON = (5, list, 'value_json', 'jsonb')

    def __init__(
        self,
        type_code: int,
        python_type: Type,
        sql_column: Optional[str],
        sql_type: str,
    ) -> None:
        self.type_code: int = type_code
        self.python_type: Type = python_type
        self.sql_column: Optional[str] = sql_column
        self.sql_type: str = sql_type


_attribute_type_code_map = tuple(pvt for pvt in AttributeValueType)


def get_attribute_value_type_for_type_code(
        type_code: int) -> AttributeValueType:
    return _attribute_type_code_map[type_code]


_attribute_type_map = {pvt.python_type: pvt for pvt in AttributeValueType}


def get_attribute_value_type_for_type(type_: Type) -> AttributeValueType:
    return _attribute_type_map.get(type_, AttributeValueType.JSON)


def get_attribute_value_type_for_value(value: Any) -> AttributeValueType:
    return get_attribute_value_type_for_type(type(value))


ComparableValue = Union[None, bool, int, float, str]

@dataclass
class Attribute():
    id_: int
    kind: str
    value_type: AttributeValueType
    comparable_value: ComparableValue
    actual_value: Any

def _make_column_identifier_seq(
    source: List[Tuple[str, str]], ) -> List[Tuple[str, sql.Identifier, str]]:
    return [(column_name, sql.Identifier(column_name), column_type)
            for column_name, column_type in source]

class PostgresAttributeMap:

    _attribute_table: sql.Identifier
    _select_attribute: sql.Composed
    _kind_type_value_map: Dict[str, Dict[AttributeValueType, Dict[
        ComparableValue, Attribute]]]  # {kind:{type:{comparable_value:param}}}
    _kind_value_map: Dict[str,
                          Dict[ComparableValue,
                               Attribute]]  # {kind:{comparable_value:param}}
    _id_map: Dict[int, Attribute]  # {id:param}

    _id_columns: List[Tuple[str, sql.Identifier,
                            str]] = _make_column_identifier_seq([
                                ('attribute_id', 'integer'),
                            ])

    _index_columns: List[Tuple[str, sql.Identifier,
                               str]] = _make_column_identifier_seq([
                                   ('value_type', 'smallint'),
                                   ('kind', 'text'),
                               ])

    _value_columns: List[Tuple[str, sql.Identifier,
                               str]] = _make_column_identifier_seq([
                                   (
                                       attribute_type.sql_column,
                                       attribute_type.sql_type,
                                   ) for attribute_type in AttributeValueType
                                   if attribute_type.sql_column is not None
                               ])

    # _column_names: List[str]

    _id_column: int = 0
    _type_column: int = 1
    _kind_column: int = 2
    _column_map: Dict[Optional[str], int]
    _type_to_column_map: Dict[AttributeValueType, int]

    _value_types: Sequence[Type] = get_args(ComparableValue) + (dict, )

    _matching_clause: sql.Composable
    _casting_clause: sql.Composable
    _key_columns: sql.Composable
    _query_values_key_columns: sql.Composable

    # _column_map : Dict[str, int]
    # _value_type_to_column_map : Dict[AttributeValueType, int]

    def __init__(
        self,
        cursor,
        attribute_table='attr',
    ) -> None:
        super().__init__()

        self._kind_type_value_map = {}
        self._kind_value_map = {}
        self._id_map = {}

        self._attribute_table = sql.Identifier(attribute_table)

        index_and_value_columns = self._index_columns + self._value_columns
        all_columns = self._id_columns + index_and_value_columns

        self._column_map = {
            column_name: i
            for i, (column_name, column_id,
                    column_type) in enumerate(all_columns)
        }

        self._type_to_column_map = {  # type: ignore
            attribute_type: self._column_map.get(attribute_type.sql_column,
                                                 None)
            for attribute_type in AttributeValueType
        }

        self._matching_clause = sql.SQL(' and ').join((sql.SQL(
            "{attribute_table}.{column_id} IS NOT DISTINCT FROM t.{column_id}"
        ).format(
            attribute_table=self._attribute_table,
            column_id=column_id,
        ) for column_name, column_id, column_type in index_and_value_columns))

        self._casting_clause = sql.SQL(', ').join(
            (sql.SQL("{}::{}").format(column_id, sql.SQL(column_type))
             for column_name, column_id, column_type in index_and_value_columns
             ))

        self._key_columns = sql.SQL(',').join(
            (column_id
             for column_name, column_id, column_type in index_and_value_columns
             ))

        self._query_values_key_columns = sql.SQL(', ').join(
            (sql.SQL('query_values.{column_id}').format(column_id=column_id, )
             for column_name, column_id, column_type in index_and_value_columns
             ))

        cursor.execute(
            sql.SQL("SELECT {columns} FROM {attribute_table}").format(
                columns=sql.SQL(',').join(
                    (column_id
                     for column_name, column_id, column_type in all_columns)),
                attribute_table=self._attribute_table,
            ))

        for row in cursor.fetchall():
            value_type = get_attribute_value_type_for_type_code(
                row[self._type_column])
            value_column = self._type_to_column_map[value_type]
            database_value = None if value_column is None else row[value_column]
            value = self._recover_value_from_database(
                value_type,
                database_value,
            )
            comparable_value = self._make_comparable_value(value_type, value)

            self._register_attribute(
                Attribute(
                    row[self._id_column],
                    row[self._kind_column],
                    value_type,
                    comparable_value,
                    value,
                ))

    def to_attribute_id(self, kind: str, value: Any, cursor=None) -> int:
        value_type = get_attribute_value_type_for_value(value)
        comparable_value = self._make_comparable_value(value_type, value)

        try:
            return self._kind_type_value_map[kind][value_type][
                comparable_value].id_
        except KeyError:
            if cursor is None:
                raise KeyError(
                    f'Unable to translate attribute {kind} : {value}.')

            typed_values = [None] * len(_attribute_type_code_map)
            typed_values[value_type.type_code] = self._make_database_value(
                value_type,
                value,
            )

            query = sql.SQL("""
WITH query_values as (
    SELECT 
        (   SELECT attribute_id
            FROM {attribute_table}
            WHERE {matching_clause}
            LIMIT 1
        ) attribute_id,
        *
    FROM
    (   SELECT {casting_clause}
        FROM (VALUES ( {values} ) ) AS t ({key_columns})
    ) t
),
inserted as (
    INSERT INTO {attribute_table} AS p ({key_columns})
    SELECT {key_columns}
    FROM query_values
    WHERE attribute_id IS NULL
    ON CONFLICT DO NOTHING
    RETURNING
        attribute_id, 
        {key_columns}
)
SELECT attribute_id, {key_columns} from query_values WHERE attribute_id IS NOT NULL
UNION ALL
SELECT * from inserted
;""").format(attribute_table=self._attribute_table,
             matching_clause=self._matching_clause,
             casting_clause=self._casting_clause,
             key_columns=self._key_columns,
             query_values_key_columns=self._query_values_key_columns,
             values=sql.SQL(' ,').join((sql.Literal(v)
                                        for v in (value_type.type_code, kind,
                                                  *(typed_values[1:]))), ))

            # print(cursor.mogrify(query))
            cursor.execute(query)

            results = cursor.fetchall()
            if len(results) >= 1:
                result = results[0]
                if result is not None and result[0] is not None:
                    self._register_attribute(
                        Attribute(
                            int(result[0]),
                            kind,
                            value_type,
                            comparable_value,
                            value,
                        ))
            return self.to_attribute_id(kind, value, None)  # retry

    def get_all_kinds(self) -> Sequence[str]:
        return tuple(self._kind_type_value_map.keys())

    # def get_all_attributes_for_kind(self, kind) -> Sequence[Attribute]:
    #     return tuple(self._kind_value_map[kind].values())

    # def get_all_ids_for_kind(self, kind) -> Sequence[int]:
    #     return tuple(self._attribute_to_id_map[kind].values())

    def to_attribute_ids(
        self,
        kvl: Union[Dict[str, Any], Iterable[Tuple[str, Any]]],
        cursor=None,
    ) -> Sequence[int]:
        if isinstance(kvl, dict):
            kvl = kvl.items()  # type: ignore
        return [
            self.to_attribute_id(kind, value, cursor) for kind, value in kvl
        ]

    def to_sorted_attribute_ids(
        self,
        kvl: Union[Dict[str, Any], Iterable[Tuple[str, Any]]],
        cursor=None,
    ) -> List[int]:
        return sorted(self.to_attribute_ids(kvl, cursor=cursor))

    def attribute_from_id(
        self,
        id_: int,
        cursor=None,
    ) -> Attribute:
        return self._id_map[id_]

    def attribute_from_ids(
        self,
        ids: Iterable[int],
        cursor=None,
    ) -> List[Attribute]:
        return [self.attribute_from_id(e, cursor) for e in i]

    def _register_attribute(
        self,
        attribute: Attribute,
    ) -> None:
        self._kind_type_value_map.setdefault(attribute.kind, {}).setdefault(
            attribute.value_type, {})[attribute.comparable_value] = attribute
        self._kind_value_map.setdefault(
            attribute.kind, {})[attribute.comparable_value] = attribute
        self._id_map[attribute.id_] = attribute

    def _make_comparable_value(
        self,
        value_type: AttributeValueType,
        value: Any,
    ) -> Any:
        if value_type == AttributeValueType.JSON:
            return simplejson.dumps(marshal.marshal(value))
        return value

    def _make_database_value(
        self,
        value_type: AttributeValueType,
        value: Any,
    ) -> Any:
        if value_type == AttributeValueType.JSON:
            return psycopg.types.json.Jsonb(marshal.marshal(value))
        return value

    def _recover_value_from_database(
        self,
        value_type: AttributeValueType,
        database_value: Any,
    ) -> Any:
        if value_type == AttributeValueType.JSON:
            return marshal.demarshal(database_value)
        return database_value