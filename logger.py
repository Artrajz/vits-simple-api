import os
import re
import sys
import logging
import logzero
import warnings
from config import config, BASE_DIR
from logging.handlers import TimedRotatingFileHandler


# 过滤警告
class SpecificWarningFilter(logging.Filter):
    def __init__(self, warning_messages):
        super().__init__()
        self.warning_messages = warning_messages

    def filter(self, record):
        return all(msg not in record.getMessage() for msg in self.warning_messages)


# 过滤警告
ignore_warning_messages = ["stft with return_complex=False is deprecated",
                           "1Torch was not compiled with flash attention",
                           "torch.nn.utils.weight_norm is deprecated",
                           "Some weights of the model checkpoint.*were not used.*initializing.*from the checkpoint.*",
                           ]

for message in ignore_warning_messages:
    warnings.filterwarnings(action="ignore", message=message)


class WarningFilter(logging.Filter):
    def filter(self, record):
        return record.levelno != logging.WARNING


logzero.loglevel(logging.WARNING)
logger = logging.getLogger("vits-simple-api")
level = config.log_config.logging_level.upper()
level_dict = {'DEBUG': logging.DEBUG, 'INFO': logging.INFO, 'WARNING': logging.WARNING, 'ERROR': logging.ERROR,
              'CRITICAL': logging.CRITICAL}
logging.getLogger().setLevel(level_dict[level])

# formatter = logging.Formatter('%(levelname)s:%(name)s %(message)s')
# formatter = logging.Formatter('%(asctime)s [%(levelname)s] [%(module)s.%(funcName)s:%(lineno)d] %(message)s',
#                               datefmt='%Y-%m-%d %H:%M:%S')

# 根据日志级别选择日志格式
if level == "DEBUG":
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s [in %(module)s.%(funcName)s:%(lineno)d]',
                                  datefmt='%Y-%m-%d %H:%M:%S')
elif level == "INFO":
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
else:
    # 如果日志级别既不是DEBUG也不是INFO，则使用默认的日志格式
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s [in %(module)s.%(funcName)s:%(lineno)d]',
                                  datefmt='%Y-%m-%d %H:%M:%S')

logs_path = os.path.join(BASE_DIR, config.log_config.logs_path)
os.makedirs(logs_path, exist_ok=True)
log_file = os.path.join(logs_path, 'latest.log')
backup_count = config.log_config.logs_backup_count
handler = TimedRotatingFileHandler(log_file, when="midnight", interval=1, backupCount=backup_count, encoding='utf-8')
handler.suffix = "%Y-%m-%d.log"
handler.setFormatter(formatter)

# remove all handlers (remove StreamHandler handle)
logging.getLogger().handlers = []

logging.getLogger().addHandler(handler)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

logging.getLogger('numba').setLevel(logging.WARNING)
logging.getLogger("langid.langid").setLevel(logging.INFO)
logging.getLogger("apscheduler.scheduler").setLevel(logging.INFO)
logging.getLogger("transformers.modeling_utils").addFilter(WarningFilter())


# Custom function to handle uncaught exceptions
def handle_exception(exc_type, exc_value, exc_traceback):
    # If it's a keyboard interrupt, don't handle it, just return
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


# Set the global exception handler in Python
sys.excepthook = handle_exception
