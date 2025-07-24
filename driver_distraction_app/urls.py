from django.urls import path
from django.views.decorators.csrf import csrf_exempt

from . import views
from .auth_views import (  # Import directly from auth_views
    login_view,
    signup_view,
    logout_view,
    account_view
)

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload_video, name='upload_video'),
    path('webcam/', views.webcam_analysis, name='webcam_analysis'),
    path('live/', views.live_analysis_view, name='live_analysis'),
    path('process-frame/', csrf_exempt(views.process_webcam_frame), name='process_frame'),
    
    # Auth URLs
    path('login/', login_view, name='login'),
    path('signup/', signup_view, name='signup'),
    path('logout/', logout_view, name='logout'),
    path('account/', account_view, name='account'),
]
    #re_path(r'^offer/$', csrf_exempt(views.offer), name='offer'),
    #path('', views.home, name='home'),
   # path('analyze-video/', views.analyze_video, name='analyze_video'),
    #path('analyze-webcam/', views.analyze_webcam, name='analyze_webcam'),

    #re_path(r'^offer/$', csrf_exempt(views.offer), name='offer'),
    #path('', views.home, name='home'),
   # path('analyze-video/', views.analyze_video, name='analyze_video'),
    #path('analyze-webcam/', views.analyze_webcam, name='analyze_webcam'),

