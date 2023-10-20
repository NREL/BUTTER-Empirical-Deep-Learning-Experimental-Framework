from dataclasses import dataclass
from dmp.postgres_interface.element.a_column_group import AColumnGroup
from dmp.postgres_interface.element.column import Column
from dmp.postgres_interface.element.column_group import ColumnGroup
from dmp.postgres_interface.element.table import Table


class CheckpointTable(Table):
    run_id: Column = Column("run_id", "uuid")
    fit_number: Column = Column("fit_number", "integer")
    fit_epoch: Column = Column("fit_epoch", "integer")
    epoch: Column = Column("epoch", "integer")

    def __init__(self) -> None:
        super().__init__("checkpoint")
