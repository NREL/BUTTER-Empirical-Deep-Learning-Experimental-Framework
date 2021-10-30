import subprocess
import sys
import time

print('worker manager!\n')

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
        worker = subprocess.Popen(subprocess_args, bufsize=1, universal_newlines=True, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, close_fds=True)
        while True:
            output = worker.stdout.readline()
            if len(output) == 0 and worker.poll() is not None:
                break
            if output:
                sys.stdout.write(output)
                # sys.stdout.flush()
                # print(output.strip(), flush=True)
        returncode = worker.poll()

        if returncode != 1:
            break
        print(f'Subprocess failed with returncode {returncode}.')
        time.sleep(60)
    print(f'Subprocess completed with returncode {returncode}, exiting Worker Manager...')
