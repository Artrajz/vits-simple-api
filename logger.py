import os
import sys
import logging
import logzero
import config
from logging.handlers import TimedRotatingFileHandler

logzero.loglevel(logging.WARNING)
logger = logging.getLogger("vits-simple-api")
level = getattr(config, "LOGGING_LEVEL", "DEBUG")
level_dict = {'DEBUG': logging.DEBUG, 'INFO': logging.INFO, 'WARNING': logging.WARNING, 'ERROR': logging.ERROR,
              'CRITICAL': logging.CRITICAL}
logging.basicConfig(level=level_dict[level])
logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger("langid.langid").setLevel(logging.INFO)
logging.getLogger("apscheduler.scheduler").setLevel(logging.INFO)

os.makedirs(config.LOGS_PATH, exist_ok=True)
log_file = os.path.join(config.LOGS_PATH, 'latest.log')
backup_count = getattr(config, "LOGS_BACKUPCOUNT", 30)
handler = TimedRotatingFileHandler(log_file, when="midnight", interval=1, backupCount=backup_count, encoding='utf-8')
handler.suffix = "%Y-%m-%d.log"
formatter = logging.Formatter('%(levelname)s:%(name)s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

logging.getLogger("werkzeug").addHandler(handler)
logging.getLogger("apscheduler.scheduler").addHandler(handler)


# Custom function to handle uncaught exceptions
def handle_exception(exc_type, exc_value, exc_traceback):
    # If it's a keyboard interrupt, don't handle it, just return
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


# Set the global exception handler in Python
sys.excepthook = handle_exception
