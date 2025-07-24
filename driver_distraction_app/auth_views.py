# driver_distraction_app/auth_views.py
from django.shortcuts import render, redirect
from django.http import JsonResponse
from .auth_utils import *
from django.contrib import messages
from django.contrib.auth import logout as django_logout
from .auth_utils import supabase_logout
from .encoders import SupabaseEncoder  # Make sure this import matches
import json

def login_view(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        
        response = supabase_login(email, password)
        if response and not response.user.identities[0].identity_data.get('email_verified', True):
            messages.warning(request, 'Please verify your email before logging in.')
            return render(request, 'login.html')
        
        if response:
            # Serialize the session data using our custom encoder
            session_data = json.loads(
                json.dumps(response.session.dict(), cls=SupabaseEncoder)
            )
            request.session['supabase_session'] = session_data
            next_url = request.GET.get('next', '/')
            return redirect(next_url)
        else:
            messages.error(request, 'Invalid email or password')
    
    return render(request, 'login.html')


def signup_view(request):
    if request.method == 'POST':
        email = request.POST.get('email')
        password = request.POST.get('password')
        confirm_password = request.POST.get('confirm_password')
        
        if password != confirm_password:
            messages.error(request, 'Passwords do not match')
            return render(request, 'signup.html')
        
        response = supabase_signup(email, password)
        if response:
            messages.success(request, 'Account created successfully! Please check your email to verify your account.')
            return redirect('login')
        else:
            messages.error(request, 'Account creation failed. Please try again.')
    
    return render(request, 'signup.html')

def logout_view(request):
    # Clear the web session only
    request.session.flush()  # Completely clears Django session
    messages.success(request, "You have been logged out successfully")
    return redirect('home')  # Redirect to homepage, not login page  # Redirect to login page after logout
def account_view(request):
    user = get_user(request)
    if not user:
        return redirect('login')
    
    return render(request, 'account.html', {'user': user})