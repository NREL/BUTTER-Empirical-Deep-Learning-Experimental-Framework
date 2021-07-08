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
import numpy as np

from command_line_tools import (
    command_line_config,
    run_tools,
)
import os
import json
import numpy

from dmp.data.safe_json_encoder import SafeJSONEncoder


class NpEncoder(SafeJSONEncoder):

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
from sqlalchemy import Column, Integer, Text, TIMESTAMP, String, Index
from sqlalchemy.dialects.postgresql import UUID, JSON, JSONB
import sqlalchemy

_credentials = None
_database = "dmp"

Base = declarative_base()


class _log(Base):
    __tablename__ = 'log'
    __table_args__ = (Index('groupname_timestamp', "groupname", "timestamp"),)
    id = Column(Integer, primary_key=True)
    job = Column(UUID(as_uuid=True), index=True)
    groupname = Column(String)
    name = Column(String)
    timestamp = Column(TIMESTAMP, server_default=func.now(), index=True)
    doc = Column(JSONB)


def _connect():
    global _credentials
    if _credentials is None:
        try:
            filename = os.path.join(os.environ['HOME'], ".jobqueue.json")
            _data = json.loads(open(filename).read())
            _credentials = _data[_database]
            user = _credentials["user"]
        except KeyError as e:
            raise Exception("No credetials for {} found in {}".format(_database, filename))
    connection_string = 'postgresql://{user}:{password}@{host}:5432/{database}'.format(**_credentials)
    db = sqlalchemy.create_engine(connection_string)
    engine = db.connect()
    Base.metadata.create_all(engine)
    session = sessionmaker(engine)()
    return engine, session


def _close(engine, session):
    session.close()
    engine.close()


### Postgres logger
def write_postgres(run_name, log_data, job=None):
    engine, session = _connect()
    log_data = json.loads(NpEncoder().encode(log_data))
    newlog = _log(name=run_name, doc=log_data, job=job)
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


def read_file(log_file):
    with open(log_file, 'r', encoding='utf-8') as f:
        return json.load(f)


### Generic logger
def write_log(log_data, path="./log", log_environment=True, name=None, job=None):
    _log_data = log_data.copy()
    if name is None:
        name = log_data["run_name"]
    if log_environment:
        _log_data.setdefault("environment", {}).update(get_environment())
    if path[:8] == 'postgres':
        write_postgres(name, _log_data, job=job)
    else:
        write_file(name, _log_data, path)


import subprocess, os, platform, datetime


def get_environment():
    env = {}

    # Git hash of current version of codebase
    try:
        file_dir = os.path.dirname(__file__)
        env["git_hash"] = subprocess.check_output(["git", "describe", "--always"], cwd=file_dir).strip().decode()
    except Exception as e:
        print("Caught exception while retrieving git hash: " + str(e))

    # Platform
    env["hostname"] = platform.node()
    env["platform"] = platform.platform()
    env["python_version"] = platform.python_version()

    # Environment variables
    # env["DMP_TYPE"] = os.getenv('DMP_TYPE')
    # env["DMP_RANK"] = os.getenv('DMP_RANK')
    # env["DMP_NUM_CPU_WORKERS"] = os.getenv('DMP_NUM_CPU_WORKERS')
    # env["DMP_NUM_GPU_WORKERS"] = os.getenv('DMP_NUM_GPU_WORKERS')
    env["SLURM_JOB_ID"] = os.getenv("SLURM_JOB_ID")

    return env
