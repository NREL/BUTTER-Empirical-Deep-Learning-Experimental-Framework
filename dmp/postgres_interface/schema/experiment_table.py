from dataclasses import dataclass
from dmp.postgres_interface.element.a_column_group import AColumnGroup
from dmp.postgres_interface.element.column import Column
from dmp.postgres_interface.element.column_group import ColumnGroup
from dmp.postgres_interface.element.table import Table


class ExperimentTable(Table):
    experiment_id: Column = Column("experiment_id", "uuid")
    experiment: Column = Column("experiment", "jsonb")
    most_recent_run: Column = Column("most_recent_run", "timestamptz")
    num_runs: Column = Column("num_runs", "integer")

    old_experiment_id: Column = Column("old_experiment_id", "integer")

    by_epoch: Column = Column("by_epoch", "bytea")
    by_loss: Column = Column("by_loss", "bytea")
    by_progress: Column = Column("by_progress", "bytea")
    epoch_subset: Column = Column("epoch_subset", "bytea")

    def __init__(self) -> None:
        super().__init__("experiment2")
