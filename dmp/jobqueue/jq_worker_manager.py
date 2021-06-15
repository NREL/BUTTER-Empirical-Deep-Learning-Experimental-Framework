import subprocess
import sys

if __name__ == "__main__":
    subprocess_args = sys.argv[1:]
    print('Starting Worker Manager...')
    while True:
        print('Launching subprocess command "{}"...'.format(' '.join(subprocess_args)))
        completed_process = subprocess.run(subprocess_args)
        if completed_process.returncode == 0:
            break
        print('Subprocess failed with returncode {}.'.format(completed_process.returncode))
    print('Subprocess completed successfully, exiting Worker Manager...')
