from typing import Any
from uuid import UUID
from hashlib import md5
from dmp.postgres_interface.postgres_interface_common import json_dump_function


def str_to_uuid(target: str) -> UUID:
    return UUID(md5(target.encode("utf-8")).hexdigest())


def json_to_uuid(target: Any) -> UUID:
    return str_to_uuid(json_dump_function(target))


def object_to_uuid(target: Any) -> UUID:
    from dmp.marshaling import marshal
    
    return json_to_uuid(marshal.marshal(target))
