from dataclasses import dataclass
from dmp.postgres_interface.element.a_column_group import AColumnGroup
from dmp.postgres_interface.element.column import Column
from dmp.postgres_interface.element.column_group import ColumnGroup
from dmp.postgres_interface.element.table import Table


class HistoryTable(Table):
    id: Column = Column("id", "uuid")
    experiment_id: Column = Column("experiment_id", "uuid")
    run_history: Column = Column("run_history", "bytea")
    extended_history: Column = Column("run_extended_history", "bytea")

    def __init__(self) -> None:
        super().__init__("history")
