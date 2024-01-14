from dmp.postgres_interface.element.column import Column
from dmp.postgres_interface.element.table import Table


class RunTable(Table):
    queue = Column("queue", "smallint")
    status = Column("status", "smallint")
    priority = Column("priority", "integer")
    id = Column("id", "uuid")
    start_time = Column("start_time", "timestamp")
    update_time = Column("update_time", "timestamp")
    worker_id = Column("worker_id", "uuid")
    parent_id = Column("parent_id", "uuid")
    experiment_id = Column("experiment_id", "uuid")
    command = Column("command", "jsonb")
    history: Column = Column("history", "bytea")
    extended_history: Column = Column("extended_history", "bytea")
    error_message = Column("error_message", "text")

    def __init__(self) -> None:
        super().__init__("run")
