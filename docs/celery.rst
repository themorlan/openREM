##########################
Celery and Flower - Legacy
##########################

**Document not ready for translation**

Celery and RabbitMQ were used by OpenREM to run tasks in the background, like exports and DICOM queries.
Since release 1.0 OpenREM uses standard python multiprocessing - hence none of those tools is required anymore.
RabbitMQ handles the message queue and Celery provides the 'workers'  that perform the tasks. 
