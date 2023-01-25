from typing import Any, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple, Type, Union, get_args

import uuid
from jobqueue.cursor_manager import CursorManager
from dmp.marshaling import marshal
from dmp.postgres_interface.attr import Attr, ComparableValue
from dmp.postgres_interface.attribute_value_type import AttributeValueType, get_attribute_value_type_for_type_code, get_attribute_value_type_for_value

from dmp.postgres_interface.column_group import ColumnGroup
from dmp.postgres_interface.postgres_schema import PostgresSchema

from psycopg.sql import Composable, Identifier, SQL, Composed, Literal
from psycopg.types.json import Jsonb


def _make_column_identifier_seq(
    source: List[Tuple[str, str]], ) -> List[Tuple[str, Identifier, str]]:
    return [(column_name, Identifier(column_name), column_type)
            for column_name, column_type in source]


class PostgresAttrMap:
    _schema: PostgresSchema

    _kind_type_value_map: Dict[str, Dict[AttributeValueType, Dict[
        ComparableValue, Attr]]]  # {kind:{type:{comparable_value:param}}}
    _kind_value_map: Dict[str, Dict[ComparableValue,
                                    Attr]]  # {kind:{comparable_value:param}}
    _id_map: Dict[int, Attr]  # {id:param}

    _get_or_create_attr_query: Composed

    def __init__(
        self,
        schema: PostgresSchema,
    ) -> None:
        super().__init__()

        self._schema = schema

        self._kind_type_value_map = {}
        self._kind_value_map = {}
        self._id_map = {}

        attr_table = self._schema.attr
        id_group = attr_table['id']
        index_group = attr_table['index']
        value_group = attr_table['value']
        json_group = attr_table['json']

        all_but_id_group = ColumnGroup.concatenate((
            index_group,
            value_group,
            json_group,
        ))

        matching_clause = SQL(' and ').join(
            (SQL("{attr_table}.{column_id} IS NOT DISTINCT FROM t.{column_id}"
                 ).format(
                     attr_table=attr_table.identifier,
                     column_id=column_id,
                 ) for column_id in value_group.identifiers))

        input_table = Identifier('_input')
        inserted_table = Identifier('_inserted')

        self._get_or_create_attr_query = SQL("""
WITH {input_table} as (
    SELECT 
        (   SELECT {attr_id}
            FROM {attr_table}
            WHERE {matching_clause}
            LIMIT 1
        ) {attr_id},
        *
    FROM
    (   SELECT {casting_clause}
        FROM (VALUES ( {values} ) ) AS t ({key_columns})
    ) t
),
{inserted_table} as (
    INSERT INTO {attr_table} AS p ({key_columns})
    SELECT {key_columns}
    FROM {input_table}
    WHERE {attr_id} IS NULL
    ON CONFLICT DO NOTHING
    RETURNING
        {attr_id}, 
        {key_columns}
)
SELECT {attr_id}, {key_columns} from {input_table} WHERE {attr_id} IS NOT NULL
UNION ALL
SELECT * from {inserted_table}
;""").format(
            input_table=input_table,
            attr_id=id_group.columns_sql,
            attr_table=attr_table.identifier,
            matching_clause=matching_clause,
            casting_clause=all_but_id_group.casting_sql,
            values=all_but_id_group.placeholders,
            key_columns=all_but_id_group.columns_sql,
            inserted_table=inserted_table,
            query_values_key_columns=all_but_id_group.of(
                input_table),
        )

        self._load_all_attributes()

    def to_attr_id(self, kind: str, value: Any, c=None) -> int:
        value_type = get_attribute_value_type_for_value(value)
        database_value = self._make_database_value(value_type, value)
        comparable_value = self._make_comparable_value(value_type,
                                                       database_value)

        try:
            return self._kind_type_value_map[kind][value_type][
                comparable_value].id_

        except KeyError:
            typed_values = [None] * len(AttributeValueType)
            typed_values[value_type.type_code] = comparable_value

            json_value = None
            if value_type == AttributeValueType.JSON:
                json_value = Jsonb(database_value)

            with CursorManager(
                    self._schema.credentials,
                    binary=True,
            ) as cursor:

                cursor.execute(
                    self._get_or_create_attr_query,
                    (
                        value_type.type_code,
                        kind,
                        *(typed_values[1:]),
                        json_value,
                    ),
                    binary=True,
                )

                results = cursor.fetchall()
                if len(results) >= 1:
                    result = results[0]
                    if result is not None and result[0] is not None:
                        self._register_attribute(
                            Attr(
                                int(result[0]),
                                kind,
                                value_type,
                                comparable_value,
                                # value,
                            ))
            return self.to_attr_id(
                kind,
                value,
            )  # retry

    def get_all_kinds(self) -> Sequence[str]:
        return tuple(self._kind_type_value_map.keys())

    # def get_all_attributes_for_kind(self, kind) -> Sequence[Attribute]:
    #     return tuple(self._kind_value_map[kind].values())

    # def get_all_ids_for_kind(self, kind) -> Sequence[int]:
    #     return tuple(self._attribute_to_id_map[kind].values())

    def to_attr_ids(
        self,
        kvl: Union[Dict[str, Any], Iterable[Tuple[str, Any]]],
    ) -> Sequence[int]:
        if isinstance(kvl, dict):
            kvl = kvl.items()  # type: ignore
        return [self.to_attr_id(kind, value) for kind, value in kvl]

    def to_sorted_attr_ids(
        self,
        kvl: Union[Dict[str, Any], Iterable[Tuple[str, Any]]],
    ) -> List[int]:
        return sorted(self.to_attr_ids(kvl))

    def attribute_from_id(
        self,
        id_: int,
    ) -> Attr:
        return self._id_map[id_]

    def attribute_map_from_ids(
        self,
        ids: Iterable[int],
    ) -> Dict[str, Any]:
        return {
            attr.kind: attr.comparable_value
            for attr in (self.attribute_from_id(e) for e in ids)
        }

    def _load_all_attributes(self) -> None:
        attr_table = self._schema.attr

        columns = ColumnGroup.concatenate((
            attr_table['id'],
            attr_table['index'],
            attr_table['value'],
            # attr_table['json'],
        ))

        id_column_index = columns['id']
        kind_column_index = columns['kind']
        type_column_index = columns['value_type']

        type_to_column_map = {  # type: ignore
            attribute_type: columns[attribute_type.sql_column]
            for attribute_type in AttributeValueType
            if attribute_type.sql_column is not None
        }

        with CursorManager(self._schema.credentials, binary=True) as cursor:
            cursor.execute(
                SQL("SELECT {columns} FROM {attr_table}").format(
                    columns=columns.columns_sql,
                    attr_table=attr_table.identifier,
                ))

            for row in cursor.fetchall():
                value_type = get_attribute_value_type_for_type_code(
                    row[type_column_index])

                comparable_value = None
                if value_type is not AttributeValueType.Null:
                    value_column = type_to_column_map[value_type]
                    comparable_value = row[value_column]

                self._register_attribute(
                    Attr(
                        row[id_column_index],
                        row[kind_column_index],
                        value_type,
                        comparable_value,
                    ))

    def _register_attribute(
        self,
        attribute: Attr,
    ) -> None:
        self._kind_type_value_map.setdefault(attribute.kind, {}).setdefault(
            attribute.value_type, {})[attribute.comparable_value] = attribute
        self._kind_value_map.setdefault(
            attribute.kind, {})[attribute.comparable_value] = attribute
        self._id_map[attribute.id_] = attribute

    def _make_comparable_value(
        self,
        value_type: AttributeValueType,
        database_value: Any,
    ) -> Any:
        if value_type == AttributeValueType.JSON:
            return self._make_json_digest(database_value)
        return database_value

    def _make_json_digest(self, value: Any) -> uuid.UUID:
        return self._schema.json_to_uuid(value)

    def _make_database_value(
        self,
        value_type: AttributeValueType,
        value: Any,
    ) -> Any:
        if value_type == AttributeValueType.JSON:
            return marshal.marshal(value)
        return value

    def _recover_value_from_database(
        self,
        value_type: AttributeValueType,
        database_value: Any,
    ) -> Any:
        if value_type == AttributeValueType.JSON:
            return marshal.demarshal(database_value)
        return database_value
