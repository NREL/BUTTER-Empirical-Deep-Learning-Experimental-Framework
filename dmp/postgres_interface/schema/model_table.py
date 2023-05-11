from dataclasses import dataclass
from dmp.postgres_interface.element.a_column_group import AColumnGroup
from dmp.postgres_interface.element.column import Column
from dmp.postgres_interface.element.column_group import ColumnGroup
from dmp.postgres_interface.element.table import Table


@dataclass(frozen=True)
class ModelTable(Table):
    name: str = 'model'
    run_id: Column = Column('run_id', 'uuid')
    epoch: Column = Column('epoch', 'integer')
    model_number: Column = Column('model_number', 'integer')
    model_epoch: Column = Column('model_epoch', 'integer')

    @property
    def columns(self) -> AColumnGroup:
        return ColumnGroup(
            self.run_id,
            self.model_number,
            self.model_epoch,
            self.epoch,
        )