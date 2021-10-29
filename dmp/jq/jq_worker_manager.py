import platform
import subprocess
import sys
import time

if __name__ == "__main__":
    subprocess_args = sys.argv[1:]
    print(f'Starting Worker Manager...')
    while True:
        print(f'Launching subprocess command "{" ".join(subprocess_args)}"...')
        # completed_process = subprocess.run(subprocess_args,
        #                                    # capture_output=False,
        #                                    bufsize=0,
        #                                    # universal_newlines=True,
        #                                    stdout=subprocess.STDOUT,
        #                                    stderr=subprocess.STDOUT)
        worker = subprocess.run(subprocess_args, bufsize=1, universal_newlines=True, stdout=subprocess.STDOUT)
        if worker.returncode != 1:
            break
        print(f'Subprocess failed with returncode {worker.returncode}.')
        time.sleep(5)
    print(f'Subprocess completed with returncode {worker.returncode}, exiting Worker Manager...')
