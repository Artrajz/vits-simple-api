from flask import render_template, redirect, url_for, flash, Blueprint
from flask_login import login_user, logout_user, login_required

from tts_app.auth.forms import LoginForm
from tts_app.auth.models import users

auth = Blueprint('auth', __name__)


@auth.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = users.get(form.username.data)
        if user and user.password == form.password.data:
            login_user(user)
            flash('Logged in successfully.')
            return redirect(url_for('admin.setting'))
        flash('Wrong username or password.')
    return render_template('login.html', form=form)

@auth.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.')
    return redirect(url_for('auth.login'))


