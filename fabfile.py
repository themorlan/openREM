# This Python file uses the following encoding: utf-8
# Original code from: Test-Driven Development with Python by Harry Percival (O’Reilly).
# Copyright 2014 Harry Percival, 978-1-449-36482-3.”


from fabric import task
from fabric.connection import Connection
from patchwork import files
import random

REPO_URL = 'https://bitbucket.org/openrem/openrem.git'


@task
def _create_directory_structure_if_necessary(c, site_folder):
    for subfolder in ('database', 'static', 'virtualenv', 'source'):
        c.run('mkdir -p {0}/{1}'.format(site_folder, subfolder))


@task
def _get_latest_source(c, source_folder):
    if files.exists(c, source_folder + '/.git'):
        c.run('cd {0} && git fetch'.format(source_folder))
    else:
        c.run('git clone {0} {1}'.format(REPO_URL, source_folder))
    # current_commit = local("git log -n 1 --format=%H", capture=True)
    current_commit = c.local("echo $BITBUCKET_COMMIT")
    c.run('cd {0} && git reset --hard {1}'.format(source_folder, current_commit))


# def _update_settings(c, source_folder, site_name):
#     settings_path = source_folder + '/openrem/openremproject/local_settings.py'
#     sed(settings_path, "DEBUG = True", "DEBUG = False")
#     sed(settings_path,
#         'ALLOWED_HOSTS =.+$',
#         'ALLOWED_HOSTS = ["{0}"]'.format(site_name)
#     )
#     secret_key_file = source_folder + '/openrem/openremproject/secret_key.py'
#     if not exists(secret_key_file):
#         chars = 'abcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*(-_=+)'
#         key = ''.join(random.SystemRandom().choice(chars) for _ in range(50))
#         append(secret_key_file, 'SECRET_KEY = "{0}"'.format(key))
#     append(settings_path, '\nfrom .secret_key import SECRET_KEY')


@task
def _update_virtualenv(c, source_folder):
    virtualenv_folder = source_folder + '/../virtualenv'
    # if not exists(virtualenv_folder + '/bin/pip'):
    #     run('virtualenv {0}'.format(virtualenv_folder))
    # run('{0}/bin/pip install -r {1}/requirements.txt'.format(virtualenv_folder, source_folder))
    c.run('{0}/bin/pip install -e {1}/'.format(virtualenv_folder, source_folder))


@task
def _update_static_files(c, source_folder):
    c.run(
        'cd {0}'
        ' && ../virtualenv/bin/python openrem/manage.py collectstatic --noinput'.format(source_folder)
    )


@task
def _update_database(c, source_folder):
    c.run(
        'cd {0}'
        ' && ../virtualenv/bin/python openrem/manage.py makemigrations remapp --noinput'.format(source_folder)
    )
    c.run(
        'cd {0}'
        ' && ../virtualenv/bin/python openrem/manage.py migrate --noinput'.format(source_folder)
    )


@task
def _restart_gunicorn(c):
    c.run(
        'sudo /usr/sbin/service gunicorn-{0} restart'.format(Connection.host)
    )


@task
def deploy(c):
    site_folder = '/home/{0}/sites/{1}'.format(c.user, c.host)
    source_folder = site_folder + '/source'
    _create_directory_structure_if_necessary(c, site_folder)
    _get_latest_source(c, source_folder)
    # _update_settings(c, source_folder, env.host)
    _update_virtualenv(c, source_folder)
    _update_static_files(c, source_folder)
    _update_database(c, source_folder)
    _restart_gunicorn(c)
