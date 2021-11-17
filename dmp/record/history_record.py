from dataclasses import dataclass, field

from sqlalchemy import Column, BigInteger
from sqlalchemy.dialects import postgresql

from dmp.sql_alchemy_globals import mapper_registry


@mapper_registry.mapped
@dataclass
class HistoryRecord:
    __tablename__ = 'materialized_experiments_3_history'
    __sa_dataclass_metadata_key__ = ''

    id: int = field(init=False, metadata={'': Column(BigInteger, primary_key=True)})

    hinge: [float] = field(default=None, metadata={'': Column(postgresql.ARRAY(postgresql.REAL))})
    accuracy: [float] = field(default=None, metadata={'': Column(postgresql.ARRAY(postgresql.REAL))})
    val_hinge: [float] = field(default=None, metadata={'': Column(postgresql.ARRAY(postgresql.REAL))})
    val_accuracy: [float] = field(default=None, metadata={'': Column(postgresql.ARRAY(postgresql.REAL))})
    squared_hinge: [float] = field(default=None, metadata={'': Column(postgresql.ARRAY(postgresql.REAL))})
    cosine_similarity: [float] = field(default=None, metadata={'': Column(postgresql.ARRAY(postgresql.REAL))})
    val_squared_hinge: [float] = field(default=None, metadata={'': Column(postgresql.ARRAY(postgresql.REAL))})
    mean_squared_error: [float] = field(default=None, metadata={'': Column(postgresql.ARRAY(postgresql.REAL))})
    mean_absolute_error: [float] = field(default=None, metadata={'': Column(postgresql.ARRAY(postgresql.REAL))})
    val_cosine_similarity: [float] = field(default=None, metadata={'': Column(postgresql.ARRAY(postgresql.REAL))})
    val_mean_squared_error: [float] = field(default=None, metadata={'': Column(postgresql.ARRAY(postgresql.REAL))})
    root_mean_squared_error: [float] = field(default=None, metadata={'': Column(postgresql.ARRAY(postgresql.REAL))})
    val_mean_absolute_error: [float] = field(default=None, metadata={'': Column(postgresql.ARRAY(postgresql.REAL))})
    kullback_leibler_divergence: [float] = \
        field(default=None, metadata={'': Column(postgresql.ARRAY(postgresql.REAL))})
    val_root_mean_squared_error: [float] = \
        field(default=None, metadata={'': Column(postgresql.ARRAY(postgresql.REAL))})
    mean_squared_logarithmic_error: [float] = \
        field(default=None, metadata={'': Column(postgresql.ARRAY(postgresql.REAL))})
    val_kullback_leibler_divergence: [float] = \
        field(default=None, metadata={'': Column(postgresql.ARRAY(postgresql.REAL))})
    val_mean_squared_logarithmic_error: [float] = \
        field(default=None, metadata={'': Column(postgresql.ARRAY(postgresql.REAL))})
    loss: [float] = field(default=None, metadata={'': Column(postgresql.ARRAY(postgresql.REAL))})
