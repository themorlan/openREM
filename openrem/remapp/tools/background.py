"""
Implements helpers to run and manage sub processes
"""

import os
import traceback
import time
import signal
import sys
import uuid
from multiprocessing import Process
import datetime

import django
from django import db
from django.db.models import Q

# Setup django. This is required on windows, because process is created via spawn and 
# django will not be initialized anymore then (On Linux this will only be executed once)
basepath = os.path.dirname(__file__) 
projectpath = os.path.abspath(os.path.join(basepath, "..", ".."))
if projectpath not in sys.path:
    sys.path.insert(1, projectpath)
os.environ["DJANGO_SETTINGS_MODULE"] = "openremproject.settings"
django.setup()

from remapp.models import BackgroundTask

def _run(fun, task_type, taskuuid, *args, **kwargs):
    """
    This helper manages the background process. (Create BackgroundTask object,
    actually call the function, handle Exceptions)
    """

    b = BackgroundTask.objects.create(
        uuid=taskuuid,
        pid = os.getpid(),
        task_type=task_type,
        started_at=datetime.datetime.now()
    )
    b.save()
    try:
        fun(*args, **kwargs)
    except: # Literally anything could happen here
        err_msg =  traceback.format_exc()
        BackgroundTask.objects.filter(pid__exact=os.getpid()).update(
            complete=True, completed_successfull=False, status=err_msg)
        return

    BackgroundTask.objects.filter(pid__exact=os.getpid()).update(
        complete=True, completed_successfull=True)

def run_in_background(fun, task_type, *args, **kwargs):
    """
    Runs fun as background Process.

    This method will create a BackgroundTask object.
    Inside the calling process it can be obtained via get_current_task().
    This function will not return until the BackgroundTask object exists. Note that 
    BackgroundTask objects will not be deleted onto completion - instead there
    complete Flag will be set to True.
    This function cannot be used with Django Tests, unless they use TransactionTestCase
    instead of TestCase (which is far slower, so use with caution).

    :param fun: The function to run. Note that you should set the status of the task yourself
        an mark as completed when exiting directly e.g. via sys.exit(). Otherwise the function
        is expected to return normally on success or to return with an exception on error.
    :param task_type: One of the strings declared in BackgroundTask.task_type. Indicates which
        kind of background process this is supposed to be. (E.g. move, query, ...)
    :param args: Positional arguments. Passed to fun. 
    :param kwargs:  Keywords arguments. Passed to fun.
    :returns: The Process object. Via pid attribute the process id can be obtained.
    """
    # On linux connection gets copied which leads to problems.
    # Close them so a new one is created for each process
    db.connections.close_all()
    
    taskuuid = str(uuid.uuid4())
    p = Process(
        target=_run,
        args=(fun, task_type, taskuuid, *args),
        kwargs=kwargs
    )

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
    return _get_task_via_uuid(taskuuid)

def terminate_background(task: BackgroundTask):
    """
    Terminate a background task by force. Sets complete=True on the task object.
    """
    if os.name == 'nt':
        # On windows this signal is not implemented. The api will just use TerminateProcess instead
        # The if/else here is only to make absolutely clear that we are not doing the same on windows
        # versus linux and potentially those commands will have to differ at some point in the future.
        os.kill(task.pid, signal.SIGTERM)
    else:
        os.kill(task.pid, signal.SIGTERM)
    os.waitpid(task.pid, 0)
    task.completed_successfull = False
    task.complete=True
    task.status = "Forcefully aborted"
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