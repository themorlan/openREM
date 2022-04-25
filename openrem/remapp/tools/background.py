"""
Implements helpers to run and manage sub processes
"""

from multiprocessing import Process
from django import db
from django.db.models import Q

from remapp.models import BackgroundTask
import os
import traceback
import time
import signal
import uuid as uuidgen

def _run(fun, task_type, uuid, *args, **kwargs):
    """
    This helper manages the background process. (Create BackgroundTask object,
    actually call the function, handle Exceptions)
    """

    b = BackgroundTask.objects.create(
        pid = os.getpid(),
        task_type=task_type,
        uuid=uuid)
    b.save()
    try:
        fun(*args, **kwargs)
    except Exception: # Literally anything could happen here
        err_msg =  traceback.format_exc()
        BackgroundTask.objects.filter(pid__exact=os.getpid()).update(
            complete=True, completed_successfull=False, status=err_msg)
        return

    BackgroundTask.objects.filter(pid__exact=os.getpid()).update(
        complete=True, completed_successfull=True)

def run_in_background(fun, task_type, *args, **kwargs):
    """
    Runs fun as background Process.

    This method will create a BackgroundTask object, the primary key of which
    corresponds to the process id (pid). Inside the background process this Id may
    be read via os.getpid(). Inside the calling process it can be obtained via
    the pid argument of return value.
    The function will not return until the BackgroundTask object exists. Note that 
    BackgroundTask objects will not be deleted onto completion - instead there
    complete Flag will be set to True.
    This function cannot be used with Django Tests, unless they use TransactionTestCase
    instead of TestCase (which is far slower, so use with caution).

    :param fun: The function to run
    :param task_type: One of the strings declared in BackgroundTask.task_type. Indicates which
        kind of background process this is supposed to be. (E.g. move, query, ...)
    :param args: Positional arguments. Passed to fun. 
    :param kwargs:  Keywords arguments. Passed to fun.
    :returns: The Process object. Via pid attribute the process id can be obtained.
    """
    # On linux connection gets copied which leads to problems.
    # Close them so a new one is created for each process
    db.connections.close_all()

    uuid = str(uuidgen.uuid4())
    p = Process(target=_run,
    args=(fun, uuid, task_type, *args),
    kwargs=kwargs)

    p.start()
    while True: # Wait until the Task object exists or process returns
        if (p.exitcode is None 
            and BackgroundTask.objects.filter(
                Q(pid__exact=p.pid) &
                Q(complete__exact=False)
            ).count() < 1):
            time.sleep(0.2)
        else:
            break
    return _get_task_via_uuid(uuid)

def terminate_background(task: BackgroundTask):
    """
    Terminate a background task by force. Sets complete=True on the task object
    """
    if os.name == 'nt':
        # On windows this signal is not implemented. The api will just use TerminateProcess instead
        # The if/else here is only to make absolutely clear that we are not doing the same on windows
        # versus linux and potentially those commands will have to differ at some point in the future.
        os.kill(task.pid, signal.SIGTERM)
    else:
        os.kill(task.pid, signal.SIGTERM)
    task.completed_successfull = False
    task.complete=True
    task.save()

def _get_task_via_uuid(uuid):
    return BackgroundTask.objects.filter(uuid__exact=uuid).first()

def _get_task_via_pid(pid):
    return BackgroundTask.objects.filter(Q(pid__exact=pid) 
        & Q(complete__exact=False)).first()

def get_current_task():
    """
    Call inside a background process to get the associated BackgroundTask object.
    If this is not executed in a background Task None will be returned.
    """
    process_id = os.getpid()
    return _get_task_via_pid(process_id)