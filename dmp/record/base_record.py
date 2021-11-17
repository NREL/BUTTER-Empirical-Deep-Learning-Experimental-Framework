import math
from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID

import sqlalchemy.dialects.postgresql
from sqlalchemy import Column, Integer, REAL, BigInteger, SmallInteger, TIMESTAMP, DateTime
from dmp.sql_alchemy_globals import mapper_registry


@mapper_registry.mapped
@dataclass
class BaseRecord:
    __tablename__ = 'materialized_experiments_3_base'
    __sa_dataclass_metadata_key__ = ''

    id: int = field(init=False, metadata={'': Column(BigInteger, primary_key=True, nullable=False)})
    job: Optional[UUID] = field(default=None,
                                metadata={'': Column(sqlalchemy.dialects.postgresql.UUID, nullable=False)})
    timestamp: Optional[DateTime] = field(default=None, metadata={'': Column(TIMESTAMP, nullable=False)})
    budget: int = field(default=-1, metadata={'': Column(Integer, nullable=False)})
    depth: int = field(default=-1, metadata={'': Column(SmallInteger, nullable=False)})
    learning_rate: float = field(default=math.nan, metadata={'': Column(REAL, nullable=False)})
    epochs: int = field(default=-1, metadata={'': Column(Integer, nullable=False)})
    batch_size: int = field(default=-1, metadata={'': Column(Integer, nullable=False)})
    validation_split: float = field(default=math.nan, metadata={'': Column(REAL, nullable=False)})
    label_noise: float = field(default=math.nan, metadata={'': Column(REAL, nullable=False)})
    group: int = field(default=-1, metadata={'': Column(SmallInteger, nullable=False)})
    dataset: int = field(default=-1, metadata={'': Column(SmallInteger, nullable=False)})
    topology: int = field(default=-1, metadata={'': Column(SmallInteger, nullable=False)})
    residual_mode: int = field(default=-1, metadata={'': Column(SmallInteger, nullable=False)})
    optimizer: int = field(default=-1, metadata={'': Column(SmallInteger, nullable=False)})
    activation: int = field(default=-1, metadata={'': Column(SmallInteger, nullable=False)})
    runtime: int = field(default=-1, metadata={'': Column(BigInteger)})
