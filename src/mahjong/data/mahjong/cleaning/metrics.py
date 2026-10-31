"""Process-local observations; never a source of recovery or business facts."""
from contextlib import contextmanager
import os
import resource
import time
import threading

_values = {}
_values_lock=threading.Lock()
_cpu_start = 0
_io_start = {}


def _io():
    try:
        with open('/proc/self/io') as stream:
            return {k:int(v) for k,v in (line.strip().split(':') for line in stream)}
    except OSError:
        return {}


def reset():
    global _cpu_start,_io_start
    _values.clear()
    usage=resource.getrusage(resource.RUSAGE_SELF)
    _cpu_start=usage.ru_utime+usage.ru_stime
    _io_start=_io()


def add(name,value=1):
    with _values_lock:
        _values[name]=_values.get(name,0)+value


@contextmanager
def timed(name):
    start=time.perf_counter()
    try:
        yield
    finally:
        add(name,time.perf_counter()-start)


def fsync(fd, *, category='fsync'):
    add(category+'_calls')
    with timed(category+'_seconds'):
        os.fsync(fd)


def snapshot():
    usage=resource.getrusage(resource.RUSAGE_SELF)
    peak=usage.ru_maxrss*1024
    # Linux getrusage can retain a launcher high-water mark across exec.
    # VmHWM belongs to the current executable's address space.
    try:
        with open('/proc/self/status') as stream:
            for line in stream:
                if line.startswith('VmHWM:'):
                    peak=int(line.split()[1])*1024
                    break
    except OSError:
        pass
    result=dict(_values,pid=os.getpid(),process_cpu_seconds=usage.ru_utime+usage.ru_stime-_cpu_start,
                process_peak_rss_bytes=peak)
    current=_io()
    result['process_io']={k:v-_io_start.get(k,0) for k,v in current.items()} if current else None
    return result
