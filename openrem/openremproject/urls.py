from django.conf import settings
from django.conf.urls import include, url
from django.contrib import auth, admin
from .settings import VIRTUAL_DIRECTORY
import remapp.views

urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^{0}'.format(VIRTUAL_DIRECTORY), include('remapp.urls')),
    url(r'^{0}openrem/'.format(VIRTUAL_DIRECTORY), include('remapp.urls')),
    url(r'^{0}admin/'.format(VIRTUAL_DIRECTORY), admin.site.urls),
    # Login / logout.
    url(r'^{0}login/$'.format(VIRTUAL_DIRECTORY), auth.views.login, name='login'),
    url(r'^{0}logout/$'.format(VIRTUAL_DIRECTORY), remapp.views.logout_page, name='logout'),
]

if settings.DEBUG:
    try:
        import debug_toolbar
        urlpatterns = [
            url(r'^{0}__debug__/'.format(VIRTUAL_DIRECTORY), include(debug_toolbar.urls)),
        ] + urlpatterns
    except ImportError:
        pass
