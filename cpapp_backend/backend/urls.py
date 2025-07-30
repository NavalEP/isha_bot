"""
URL configuration for backend project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include, re_path
from django.conf import settings
from django.conf.urls.static import static
from django.views.generic import TemplateView
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from django.views.static import serve
import os
from cpapp.api.chat.views import ShortlinkRedirectView

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/v1/agent/', include('cpapp.urls')),
    # Serve images directly from static directory
    path('images/<path:path>', serve, {'document_root': os.path.join(settings.STATICFILES_DIRS[0], 'images')}),
    path('favicon.svg', serve, {'document_root': settings.STATICFILES_DIRS[0], 'path': 'favicon.svg'}),
    # Serve static assets with proper MIME types
    path('static/<path:path>', serve, {'document_root': settings.STATICFILES_DIRS[0]}),
]

# Serve static files in development (MUST be before catch-all route)
if settings.DEBUG:
    urlpatterns += staticfiles_urlpatterns()
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATICFILES_DIRS[0])

# Catch all routes and serve React app (must be last)
urlpatterns += [
    re_path(r'^.*$', TemplateView.as_view(template_name='index.html'), name='home'),
]
