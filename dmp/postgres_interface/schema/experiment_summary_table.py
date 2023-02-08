from dataclasses import dataclass
from dmp.postgres_interface.element.a_column_group import AColumnGroup
from dmp.postgres_interface.element.column import Column
from dmp.postgres_interface.element.column_group import ColumnGroup
from dmp.postgres_interface.element.table import Table


@dataclass(frozen=True)
class ExperimentSummaryTable(Table):
    name: str = 'experiment_summary'
    experiment_id: Column = Column('experiment_id', 'uuid')
    last_run_timestamp: Column = Column('last_run_timestamp', 'timestamp')
    run_update_limit: Column = Column('run_update_limit', 'timestamp')
    by_epoch: Column = Column('by_epoch', 'bytea')
    by_loss: Column = Column('by_loss', 'bytea')
    by_progress: Column = Column('by_progress', 'bytea')
    epoch_subset: Column = Column('epoch_subset', 'bytea')



