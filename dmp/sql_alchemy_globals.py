from sqlalchemy import MetaData
from sqlalchemy.orm import registry

class_registry = {}
metadata_obj = MetaData()
mapper_registry = registry(metadata=metadata_obj, class_registry=class_registry)


def _get_sql_engine():
    global _credentials
    if _credentials is None:
        try:
            filename = os.path.join(os.environ['HOME'], ".jobqueue.json")
            _data = json.loads(open(filename).read())
            _credentials = _data[_database]
            _credentials["user"]
        except KeyError:
            raise Exception("No credetials for {} found in {}".format(_database, filename))
    connection_string = 'postgresql://{user}:{password}@{host}:5432/{database}'.format(**_credentials)
    return sqlalchemy.create_engine(connection_string)

# TODO: finish this guy for sqlalchemy
def connect(credentials : {}):
    max_wait_time = 60 * 60
    max_attempts = 10000
    attempts = 0
    while attempts < max_attempts:
        wait_time = 60.0

        try:
            psycopg2.connect(**credentials)
        except psycopg2.OperationalError as e:
            print(f'OperationalError while connecting to database: {e}', flush=True)

            if attempts >= max_attempts or wait_time >= max_wait_time:
                raise e

            sleep_time = random.uniform(0.0, wait_time)
            wait_time += sleep_time
            attempts += 1
            time.sleep(sleep_time)
            continue
        break


def _connect():
    db = _get_sql_engine()
    engine = db.connect()
    Base.metadata.create_all(engine)
    session = sessionmaker(engine)()
    return engine, session