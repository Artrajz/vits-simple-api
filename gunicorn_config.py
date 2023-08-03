import multiprocessing
import config

bind = f"0.0.0.0:{getattr(config, 'PORT', 23456)}"
workers = multiprocessing.cpu_count() * 2 + 1
