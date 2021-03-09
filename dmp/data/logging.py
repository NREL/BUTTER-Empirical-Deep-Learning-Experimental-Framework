"""
logging.py

Functions to write experimental log files to a database for later analysis

Use write_log elsewhere in the DMP codebase.

write_log(log_data, config):
    log_data: Dictionary or other JSON serializeable object
    config: Dictionary or String - config["log"] object from aspect_test type config

Configurations:
- for JSON file logging, set "log":"./path/to/logdir"
- for Postgres logging, set "log":"postgres"

Note, we detect this by checking the prefix of the "log" string. So if you want to save JSON files into a folder called postgres for some reason, please use "./postgres"

}


TODO:
- Allow user to configure postgres connection string
- Separate file based and postgres based logging into separate modules / classes so sqlalchemy becomes an optional installation

"""


from command_line_tools import (
    command_line_config,
    run_tools,
)
import os
import json
import numpy

class NpEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, numpy.integer):
            return int(obj)
        elif isinstance(obj, numpy.floating):
            return float(obj)
        elif isinstance(obj, numpy.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

from sqlalchemy.ext.declarative import declarative_base  
from sqlalchemy.orm import sessionmaker
from sqlalchemy.schema import FetchedValue
from sqlalchemy.sql import func
from sqlalchemy import Column, Integer, Text, TIMESTAMP, String
from sqlalchemy.dialects.postgresql import UUID, JSON, JSONB
import sqlalchemy


Base = declarative_base()
class _log(Base):
    __tablename__ = 'log'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    timestamp = Column(TIMESTAMP, server_default=func.now())
    doc = Column(JSON)

def _connect():
    connection_string = 'postgresql://jperrsau:@localhost:5432/dmp'
    db = sqlalchemy.create_engine(connection_string)  
    engine = db.connect()
    Base.metadata.create_all(engine)  
    session = sessionmaker(engine)()
    return engine, session

def _close(engine, session):
    session.close()
    engine.close()


### Postgres logger
def write_postgres(run_name, log_data, config):
    engine, session = _connect()
    log_data = json.loads(NpEncoder().encode(log_data))
    newlog = _log(name=run_name, doc=log_data)
    print('log postgres: committing {}'.format(run_name))
    session.add(newlog) 
    session.commit()
    _close(engine, session)

### File logger
def write_file(run_name, log_data, log_path="./log"):
    run_tools.makedir_if_not_exists(log_path)
    log_file = os.path.join(log_path, '{}.json'.format(run_name))
    print('log file: {}'.format(log_file))
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2, sort_keys=True, cls=NpEncoder)

### Generic logger
def write_log(log_data, config):
    if config[:8] == 'postgres':
        write_postgres(log_data['run_name'], log_data, config)
    else:
        write_file(log_data['run_name'], log_data, config)
