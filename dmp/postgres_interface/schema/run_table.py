from dataclasses import dataclass
from dmp.postgres_interface.element.a_column_group import AColumnGroup
from dmp.postgres_interface.element.column import Column
from dmp.postgres_interface.element.column_group import ColumnGroup
from dmp.postgres_interface.element.table import Table


@dataclass(frozen=True)
class RunTable(Table):
    name: str = 'run'
    experiment_id: Column = Column('experiment_id', 'uuid')
    run_timestamp: Column = Column('run_timestamp', 'timestamp')
    run_id: Column = Column('run_id', 'uuid')
    job_id: Column = Column('job_id', 'uuid')
    seed: Column = Column('seed', 'bigint')
    slurm_job_id: Column = Column('slurm_job_id', 'bigint')
    task_version: Column = Column('task_version', 'smallint')
    num_nodes: Column = Column('num_nodes', 'smallint')
    num_cpus: Column = Column('num_cpus', 'smallint')
    num_gpus: Column = Column('num_gpus', 'smallint')
    gpu_memory: Column = Column('gpu_memory', 'integer')
    host_name: Column = Column('host_name', 'text')
    batch: Column = Column('batch', 'text')
    run_data: Column = Column('run_data', 'jsonb')
    run_history: Column = Column('run_history', 'bytea')
    extended_history: Column = Column('run_extended_history', 'bytea')

    @property
    def values(self) -> AColumnGroup:
        return ColumnGroup(
            self.run_id,
            self.job_id,
            self.seed,
            self.slurm_job_id,
            self.task_version,
            self.num_nodes,
            self.num_cpus,
            self.num_gpus,
            self.gpu_memory,
            self.host_name,
            self.batch,
        )

    @property
    def insertion_columns(self) -> AColumnGroup:
        return ColumnGroup(
            self.values,
            self.run_data,
            self.run_history,
            self.extended_history,
        )
