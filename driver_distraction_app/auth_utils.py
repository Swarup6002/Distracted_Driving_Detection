from supabase import create_client, Client
import os
from django.conf import settings
from functools import wraps
from django.shortcuts import redirect

def get_supabase():
    url = settings.SUPABASE_URL
    key = settings.SUPABASE_KEY
    return create_client(url, key)

def login_required(view_func):
    """Decorator to check if user is authenticated with Supabase"""
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        if not request.session.get('supabase_session'):
            return redirect(f'/login/?next={request.path}')
        return view_func(request, *args, **kwargs)
    return wrapper

def supabase_signup(email: str, password: str):
    supabase = get_supabase()
    try:
        response = supabase.auth.sign_up({
            "email": email,
            "password": password,
        })
        return response
    except Exception as e:
        print(f"Signup error: {str(e)}")
        return None

def supabase_login(email: str, password: str):
    supabase = get_supabase()
    try:
        response = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        return response
    except Exception as e:
        print(f"Login error: {str(e)}")
        return None

def supabase_logout(request):
    try:
        if 'supabase_session' in request.session:
            # Clear Supabase session
            supabase = get_supabase()
            supabase.auth.sign_out(request.session['supabase_session']['access_token'])
            
            # Clear Django session
            request.session.flush()
            
            return True
    except Exception as e:
        print(f"Logout error: {str(e)}")
    return False

def get_user(request):
    if 'supabase_session' in request.session:
        supabase = get_supabase()
        try:
            user = supabase.auth.get_user(request.session['supabase_session']['access_token'])
            return user
        except:
            return None
    return None