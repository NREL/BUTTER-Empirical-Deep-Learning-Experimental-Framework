from dataclasses import dataclass, field

from sqlalchemy import Column, BigInteger
from sqlalchemy.dialects import postgresql

# # define two Table objects
# log_base_table = Table(
#     'materialized_experiments_3_base', metadata_obj,
#     Column('id', BigInteger, primary_key=True),
#     Column('job', sqlalchemy.dialects.postgresql.UUID),
#     # = field(default=lambda : UUID())Column('timestamp', TIMESTAMP),
#     Column('budget', Integer),
#     Column('depth', SmallInteger),
#     Column('learning_rate', REAL),
#     Column('epochs', Integer),
#     Column('batch_size', Integer),
#     Column('validation_split', REAL),
#     Column('label_noise', REAL),
#     Column('groupname', SmallInteger),
#     Column('dataset', SmallInteger),
#     Column('topology', SmallInteger),
#     Column('residual_mode', SmallInteger),
#     Column('optimizer', SmallInteger),
#     Column('activation', SmallInteger),
#     Column('runtime', Integer),
# )
#
# log_loss_table = Table('address', metadata_obj,
#                       Column('id', Integer, primary_key=True),
#                       Column('user_id', Integer, ForeignKey('user.id')),
#                       Column('email_address', String)
#                       )
from dmp.sql_alchemy_globals import mapper_registry


@mapper_registry.mapped
@dataclass
class ValLossRecord:
    __tablename__ = 'materialized_experiments_3_val_loss'
    __sa_dataclass_metadata_key__ = ''

    id: int = field(init=False, metadata={'': Column(BigInteger, primary_key=True)})
    val_loss: [float] = field(default=None, metadata={'': Column(postgresql.ARRAY(postgresql.REAL))})
