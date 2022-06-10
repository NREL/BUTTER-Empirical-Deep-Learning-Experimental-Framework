import random
import subprocess
import sys
import time

if __name__ == "__main__":
    subprocess_args = sys.argv[1:]
    print(f'Starting Worker Manager...')

    print(f'...')
    print(f'... {subprocess_args}')
    while True:
        print(f'Start subprocess loop...')

        print(f'Launching subprocess command "{" ".join(subprocess_args)}"...', flush=True)
        worker = subprocess.Popen(subprocess_args, bufsize=1, universal_newlines=True, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, close_fds=True)
        while True:
            output = worker.stdout.readline()
            if len(output) == 0 and worker.poll() is not None:
                break
            if output:
                sys.stdout.write(output)
        returncode = worker.poll()

        if returncode == 0:
            break
        print(f'Subprocess failed with returncode {returncode}.', flush=True)
        time.sleep(random.uniform(5, 90))
    print(f'Subprocess completed with returncode {returncode}, exiting Worker Manager...', flush=True)
