import gc
import multiprocessing

bind = "0.0.0.0:23456"
# workers = multiprocessing.cpu_count()
workers = 1
preload_app = True
timeout = 120

# disable GC in master as early as possible
gc.disable()

def when_ready(server):
    # freeze objects after preloading app
    gc.freeze()
    print("Objects frozen in perm gen: ", gc.get_freeze_count())

def post_fork(server, worker):
    # reenable GC on worker
    gc.enable()