from django.conf import settings
from django.urls import include, path
from django.contrib import auth, admin
from .settings import VIRTUAL_DIRECTORY
import remapp.views

urlpatterns = [
    path('{0}'.format(VIRTUAL_DIRECTORY), include('remapp.urls')),
    path('{0}openrem/'.format(VIRTUAL_DIRECTORY), include('remapp.urls')),  # is this actually necessary?
    path('{0}admin/'.format(VIRTUAL_DIRECTORY), admin.site.urls),
    # Login / logout.
    path('{0}login/'.format(VIRTUAL_DIRECTORY), auth.views.LoginView, name='login'),
    path('logout/'.format(VIRTUAL_DIRECTORY), remapp.views.logout_page, name='logout'),
]

if settings.DEBUG:
    try:
        import debug_toolbar
        urlpatterns = [
            path('{0}__debug__/'.format(VIRTUAL_DIRECTORY), include(debug_toolbar.urls)),
        ] + urlpatterns
    except ImportError:
        pass
