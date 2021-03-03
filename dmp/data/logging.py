"""
dmplog.py

Functions to write experimental log files to a database for later analysis
"""

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

def write(name, doc):
    if session is None:
        _connect()
    newlog = _log(name=name, doc=doc)
    session.add(newlog) 
    session.commit()
