"""
logging.py

Functions to write experimental log files to a database for later analysis

Use write_log elsewhere in the DMP codebase.

write_log(log_data, config):
    log_data: Dictionary or other JSON serializeable object
    config: Dictionary or String - config["log"] object from aspect_test type config

Configurations:

For JSON File based logging, you have two options

{
    "log": "./path
}

{
    "log": {
        "backend":"file",
        "path":"./path"
    }
}


For Postgres logging

{
    "log": {
        "backend":"postgres"
    }
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

### Section: postgres logger

from sqlalchemy.ext.declarative import declarative_base  
from sqlalchemy.orm import sessionmaker
from sqlalchemy.schema import FetchedValue
from sqlalchemy.sql import func
from sqlalchemy import Column, Integer, Text, TIMESTAMP, String
from sqlalchemy.dialects.postgresql import UUID, JSON, JSONB
import sqlalchemy

engine = None
session = None

Base = declarative_base()
class _log(Base):
    __tablename__ = 'log'
    id = Column(Integer, primary_key=True)
    name = Column(String)
    timestamp = Column(TIMESTAMP, server_default=func.now())
    doc = Column(JSON)

def _connect():
    global engine
    global session
    connection_string = 'postgresql://jperrsau:@localhost:5432/dmp'
    db = sqlalchemy.create_engine(connection_string)  
    engine = db.connect()
    Base.metadata.create_all(engine)  
    session = sessionmaker(engine)()

def _close():
    global engine
    global session
    session.close()
    engine.close()
    session = None
    engine = None

def write_postgres(run_name, log_data, config):
    if session is None:
        print('log postgres: connecting to server...')
        _connect()
    log_data = json.loads(NpEncoder().encode(log_data))
    newlog = _log(name=run_name, doc=log_data)
    print('log postgres: committing {}'.format(run_name))
    session.add(newlog) 
    session.commit()



### Section: file logger

def write_file(run_name, log_data, log_path="./log"):
    run_tools.makedir_if_not_exists(log_path)
    log_file = os.path.join(log_path, '{}.json'.format(run_name))
    print('log file: {}'.format(log_file))
    with open(log_file, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2, sort_keys=True, cls=NpEncoder)



### Generic logging function
def write_log(log_data, config):
    if config[:8] == 'postgres':
        write_postgres(log_data['run_name'], log_data, config)
    else:
        write_file(log_data['run_name'], log_data, config)
