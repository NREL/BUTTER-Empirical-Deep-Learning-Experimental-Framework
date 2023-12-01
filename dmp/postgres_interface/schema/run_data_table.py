from dmp.postgres_interface.element.column import Column
from dmp.postgres_interface.element.table import Table


class RunDataTable(Table):
    id = Column("id", "uuid")
    command = Column("command", "jsonb")

    def __init__(self) -> None:
        super().__init__("run_data")
