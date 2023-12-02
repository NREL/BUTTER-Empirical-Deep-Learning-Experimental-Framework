from dmp.postgres_interface.element.column import Column
from dmp.postgres_interface.element.table import Table


class RunStatusTable(Table):
    queue = Column("queue", "smallint")
    status = Column("status", "smallint")
    priority = Column("priority", "integer")
    id = Column("id", "uuid")
    start_time = Column("start_time", "timestamp")
    update_time = Column("update_time", "timestamp")
    worker = Column("worker", "uuid")
    error_count = Column("error_count", "smallint")
    error = Column("error", "text")
    parent = Column("parent", "uuid")
    experiment_id = Column("experiment_id", "uuid")
    summarized = Column("summarized", "smallint")

    def __init__(self) -> None:
        super().__init__("run_status")
