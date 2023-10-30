from flask import Blueprint
from flask_login import login_required

admin = Blueprint('admin', __name__)

@admin.route('/')
@login_required
def setting():
    return "Hello Admin!"