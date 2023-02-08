import random
import subprocess
import sys
import time

from dmp import common

if __name__ == "__main__":
    subprocess_args = sys.argv[1:]
    print(f'Starting Worker Manager...')

    
    print(f'...')
    print(f'... {subprocess_args}')
    while True:
        git_hash = common.get_git_hash()

        print(f'Start subprocess loop...')

        print(f'Launching subprocess command "{" ".join(subprocess_args)}"...',
              flush=True)
        worker = subprocess.Popen(subprocess_args,
                                  bufsize=1,
                                  universal_newlines=True,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT,
                                  close_fds=True)
        while True:
            outstream = worker.stdout
            if outstream is None:
                break
            output = outstream.readline()
            if len(output) == 0 and worker.poll() is not None:
                break
            if output:
                sys.stdout.write(output)
        returncode = worker.poll()

        if returncode == 0:
            if common.get_git_hash() == git_hash:
                break
            print(f'Restarting worker due to git hash change.', flush=True)
            continue
        print(f'Subprocess failed with returncode {returncode}.', flush=True)
        time.sleep(random.uniform(5, 90))
    print(
        f'Subprocess completed with returncode {returncode}, exiting Worker Manager...',
        flush=True)
