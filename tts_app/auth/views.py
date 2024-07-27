from flask import render_template, redirect, url_for, flash, Blueprint, current_app
from flask_login import login_user, logout_user, login_required

from config import config
from tts_app.auth.forms import LoginForm

auth = Blueprint('auth', __name__)


@auth.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        admin = config.admin
        if admin.username == form.username.data and admin.password == form.password.data:
            login_user(admin)
            flash('Logged in successfully.')
            return redirect(url_for('admin.home'))
        flash('Wrong username or password.')
    return render_template('pages/login.html', form=form)


@auth.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.')
    return redirect(url_for('auth.login'))
