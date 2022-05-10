#    OpenREM - Radiation Exposure Monitoring tools for the physicist
#    Copyright (C) 2022 The Royal Marsden NHS Foundation Trust
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    Additional permission under section 7 of GPLv3:
#    You shall not make any use of the name of The Royal Marsden NHS
#    Foundation trust in connection with this Program in any press or
#    other public announcement without the prior written consent of
#    The Royal Marsden NHS Foundation Trust.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""
..  module:: background
    :synopsis: Module with tools to run functions on a different process
        and to record their execution with tasks

..  moduleauthor:: Jannis Widmer

"""

import os
import traceback
import time
import signal
import sys
import uuid
from multiprocessing import Process

import django
from django import db
from django.db.models import Q
from django.db import transaction
from django.utils import timezone

# Setup django. This is required on windows, because process is created via spawn and
# django will not be initialized anymore then (On Linux this will only be executed once)
basepath = os.path.dirname(__file__)
projectpath = os.path.abspath(os.path.join(basepath, "..", ".."))
if projectpath not in sys.path:
    sys.path.insert(1, projectpath)
os.environ["DJANGO_SETTINGS_MODULE"] = "openremproject.settings"
django.setup()

from remapp.models import BackgroundTask, DicomQuery


def run_as_task(func, task_type, num_proc, num_of_task_type, taskuuid, *args, **kwargs):
    """
    Runs func as a task. (Which means it runs normally, but a BackgroundTask
    object is created and hence the execution as well as occurred errors are
    documented and visible to the user).

    As a note: This is used as a helper for run_in_background. However, in
    principle it could also be used to run any function in sequential
    for which we would like to document that it was executed. (Has some not so nice
    parts, especially that user can terminate a bunch of sequentially running tasks)

    Note that waiting here if it is ok to start is actually quite ugly, since running a lot
    of processes with conditions will use a lot of RAM without any use. This however is the
    simplest possible fix for making the run_in_background_with_limits non blocking, which
    is a requirement for the docker import. A nice update would be to either have 1 process
    managing the execution of the other processes, or at least do this based on signaling
    from the exiting processes instead of polling.

    :param func: The function to run
    :param task_type: A string documenting what kind of task this is
    :param num_proc: The maximum number of processes that should be executing
    :param num_of_task_type: The maximum number of processes for multiple tasks type which are allowed to
        run at the same time
    :param taskuuid: An uuid which will be used as uuid of the BackgroundTask object. If None will generate one itself
    :args: Args to func
    :kwargs: Args to func
    :return: The created BackgroundTask object
    """
    if taskuuid is None:
        taskuuid = str(uuid.uuid4())

    b = BackgroundTask.objects.create(
        uuid=taskuuid,
        proc_id=os.getpid(),
        task_type=task_type,
        started_at=None,
    )
    b.save()

    import time

    if num_proc > 0 or len(num_of_task_type) > 0:
        while True:
            with transaction.atomic():
                qs = BackgroundTask.objects.filter(
                    Q(complete__exact=False) & Q(started_at__isnull=False)
                )
                a = qs.count()
                if num_proc == 0 or a < num_proc:
                    ok = True
                    for k, v in num_of_task_type.items():
                        a = qs.filter(
                            Q(task_type__exact=k) & Q(started_at__isnull=False)
                        ).count()
                        ok = ok and a < v
                        if not ok:
                            break
                    if ok:
                        b.started_at = timezone.now()
                        b.save()
                        break
            time.sleep(0.3)
    else:
        b.started_at = timezone.now()
        b.save()

    try:
        func(*args, **kwargs)
    except Exception:  # Literally anything could happen here
        b = _get_task_via_uuid(taskuuid)
        b.complete = True
        b.completed_successfully = False
        b.error = traceback.format_exc()
        b.save()
        return b

    b = _get_task_via_uuid(taskuuid)
    if not b.complete:
        b.complete = True
        b.completed_successfully = True
        b.save()
    return b


def run_in_background_with_limits(
    func, task_type, num_proc, num_of_task_type, *args, **kwargs
):
    """
    Runs fun as background Process.

    This method will create a BackgroundTask object, which can be obtained
    via get_current_task() inside the calling process.
    This function will not return until the BackgroundTask object exists in the database.
    Potentially it may only return after the process has already exited.
    Note that BackgroundTask objects will not be deleted onto completion - instead the
    complete flag will be set to True.
    This function cannot be used with Django Tests, unless they use TransactionTestCase
    instead of TestCase (which is far slower, so use with caution).
    num_proc and num_of_task_type can be used to give conditions on the start.

    :param func: The function to run. Note that you should set the status of the task yourself
        and mark as completed when exiting yourself e.g. via sys.exit(). Assuming the function
        returns normally on success or returns with an exception on error, the status of
        the BackgroundTask object will be set correctly.
    :param task_type: One of the strings declared in BackgroundTask.task_type. Indicates which
        kind of background process this is supposed to be. (E.g. move, query, ...)
    :param num_proc: Will wait with execution until there are less than num_proc active tasks.
        If num_proc == 0 will run directly
    :param num_of_task_type: A dictionary from str to int, where key should be some used
        task_type. Will wait until there are less active tasks of task_type than
        specified in the value of the dict for that key.
    :param args: Positional arguments. Passed to func.
    :param kwargs:  Keywords arguments. Passed to func.
    :returns: The BackgroundTask object.
    """
    taskuuid = str(uuid.uuid4())

    # On linux connection gets copied which leads to problems.
    # Close them so a new one is created for each process
    db.connections.close_all()

    p = Process(
        target=run_as_task,
        args=(func, task_type, num_proc, num_of_task_type, taskuuid, *args),
        kwargs=kwargs,
    )

    p.start()
    while True:  # Wait until the Task object exists or process returns
        if (
            p.exitcode is None
            and BackgroundTask.objects.filter(
                Q(proc_id__exact=p.pid) & Q(complete__exact=False)
            ).count()
            < 1
        ):
            time.sleep(0.2)
        else:
            break
    return _get_task_via_uuid(taskuuid)


def run_in_background(func, task_type, *args, **kwargs):
    """
    Syntactic sugar around run_in_background_with_limits.

    Arguments correspond to run_in_background_with_limits. Will always run the
    passed function as fast as possible. This function should always
    be used for functions triggered from the webinterface.
    """
    return run_in_background_with_limits(func, task_type, 0, {}, *args, **kwargs)


def terminate_background(task: BackgroundTask):
    """
    Terminate a background task by force. Sets complete=True on the task object.
    """
    task.completed_successfully = False
    task.complete = True
    task.error = "Forcefully aborted"
    task.save()

    if task.proc_id > 0:
        try:
            if os.name == "nt":
                # On windows this signal is not implemented. The api will just use TerminateProcess instead.
                os.kill(task.proc_id, signal.SIGTERM)
            else:
                os.kill(task.proc_id, signal.SIGTERM)
                # Wait until the process has returned (potentially it already has when we call wait)
                # On  Windows the equivalent does not work, but seems to be blocking there anyway
                os.waitpid(task.proc_id, 0)

        except (ProcessLookupError, OSError):
            pass


def wait_task(task: BackgroundTask):
    """
    Wait until the task has completed
    """
    while True:
        task.refresh_from_db()
        if task.complete:
            return
        time.sleep(0.3)


def _get_task_via_uuid(task_uuid):
    return BackgroundTask.objects.filter(uuid__exact=task_uuid).first()


def _get_task_via_pid(process_id):
    return BackgroundTask.objects.filter(
        Q(proc_id__exact=process_id) & Q(complete__exact=False)
    ).first()


def get_current_task():
    """
    :return: The associated BackgroundTask object when called in a task.
        If this is not executed in a background Task None will be returned.
    """
    process_id = os.getpid()
    return _get_task_via_pid(process_id)


def get_or_generate_task_uuid():
    """
    :return: If called from within a task the task id, else a generated uuid
    """
    task = get_current_task()
    if task is None:
        return str(uuid.uuid4())
    else:
        return task.uuid


def record_task_info(info_msg):
    """
    Small helper that checks if we are in a task and
    assuming we are records info_msg as info.
    """
    b = get_current_task()
    if b is not None:
        b.info = info_msg
        b.save()


def record_task_error_exit(error_msg):
    """
    Small helper that checks if we are in a task and
    assuming we are records error_msg as well as setting
    the completed_successfully to false and completed to
    True. Note that get_current_task will return None
    after a call to this.
    """
    b = get_current_task()
    if b is not None:
        b.complete = True
        b.completed_successfully = False
        b.error = error_msg
        b.save()


def record_task_related_query(study_instance_uid):
    """
    Tries to find the related DicomQRRspStudy object
    given a study instance uid and if this is running in
    a task will record it to the query object. This
    is used to later find the import tasks that were run as
    part of a query.
    Since this actually just takes the latest query if the user
    runs imports manually via script it may in principle wrongly
    associate.
    """
    b = get_current_task()
    if b is not None:
        if DicomQuery.objects.exists():
            queries = DicomQuery.objects.filter(
                started_at__isnull=False
            ).latest(
                "started_at"
            ).dicomqrrspstudy_set.filter(
                study_instance_uid=study_instance_uid
            ).all()

            for query in queries:
                query.related_imports.add(b)
