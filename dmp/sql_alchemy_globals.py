from sqlalchemy import MetaData
from sqlalchemy.orm import registry

class_registry = {}
metadata_obj = MetaData()
mapper_registry = registry(metadata=metadata_obj, class_registry=class_registry)

# TODO: finish this guy for sqlalchemy
def connect(credentials : {}):
    max_wait_time = 60 * 60
    max_attempts = 10000
    attempts = 0
    while attempts < max_attempts:
        wait_time = 60.0

        try:
            connection = psycopg2.connect(**credentials)
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