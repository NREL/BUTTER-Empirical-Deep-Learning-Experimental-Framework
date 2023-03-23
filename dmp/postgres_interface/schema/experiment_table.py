from dataclasses import dataclass
from dmp.postgres_interface.element.a_column_group import AColumnGroup
from dmp.postgres_interface.element.column import Column
from dmp.postgres_interface.element.column_group import ColumnGroup
from dmp.postgres_interface.element.table import Table


@dataclass(frozen=True)
class ExperimentTable(Table):
    name: str = 'experiment'
    experiment_id: Column = Column('experiment_id', 'uuid')
    experiment_attrs: Column = Column('experiment_attrs', 'integer[]')
    experiment_tags: Column = Column('experiment_tags', 'integer[]')
    old_experiment_id: Column = Column('old_experiment_id', 'integer')

    @property
    def values(self) -> AColumnGroup:
        return self.old_experiment_id

    @property
    def all(self) -> AColumnGroup:
        return ColumnGroup(
            self.experiment_id,
            self.experiment_attrs,
            self.experiment_tags,
            self.values,
        )

