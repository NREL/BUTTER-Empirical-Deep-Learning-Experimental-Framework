from jobqueue.cursor_manager import CursorManager
from jobqueue.job import Job
from jobqueue.job_queue import JobQueue
from jobqueue.job_status import JobStatus
from psycopg import sql

class DMPJobQueue(JobQueue):


    def complete(self, job: Job) -> None:
        """When a job is finished, this function will mark the status as done."""
        with CursorManager(self._credentials) as cursor:
            cursor.execute(
                sql.SQL("""
WITH 
UPDATE {status_table}
SET status = {complete_status},
    update_time = NOW()
WHERE id = %s;""").format(
                    status_table=sql.Identifier(self._status_table),
                    complete_status=sql.Literal(JobStatus.Complete.value),
                ),
                [job.id],
            )