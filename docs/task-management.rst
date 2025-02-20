Task management
***************

.. contents::

Viewing task and service statuses
=================================

.. figure:: img/TasksMenu.png
   :figwidth: 30%
   :align: right
   :alt: Config options

   Figure 1: The ``Tasks`` menu

Users who are logged in with admin rights can use the **Tasks** menu and choose **All tasks** to see the following:

* A list of the tasks currently being executed
* A list of previous tasks and their final status. If any errors occurred they will be displayed here.

.. figure:: img/tasks3waiting4inprogress.png
   :figwidth: 100%
   :align: center
   :alt: Task and service status

   Figure 2: The task administration page

Terminating running tasks
=========================

It is possible to terminate any active tasks by clicking the red button. **There is no confirmation step**.
Note that this immediately interrupts everything this process was doing so far, leading to things like partially
imported studies. In general this should not be an issue (in case of aborted imports they
should be completed when you start importing them again), but note that there is a certain risk in killing tasks
and use this only as a last resort.

A note on move: executing a move will create a task which then produces import tasks for all the studies it should
import. This means if you intend to abort a move you should abort the task with Task type "move" and not the import
tasks started by that process!

Configuring the size of task history
====================================

The status of 2000 active, recent and older tasks are stored in the OpenREM database. This limit can be
altered by users who are logged in with admin rights by clicking on **Task settings** in the **Config** menu and
changing the current value. If this limit is set to a very high value it can cause the web browser to run out of
memory when trying to view the **Task** page due to the large number of rows in the tables.
