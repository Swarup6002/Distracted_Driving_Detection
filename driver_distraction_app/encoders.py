# driver_distraction_app/encoders.py
from datetime import datetime, date
import json

class SupabaseEncoder(json.JSONEncoder):
    """Custom JSON encoder for Supabase session data"""
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return super().default(obj)