import logging
from io import StringIO
from contextlib import redirect_stdout
from modulus.utils.sfno.logging_utils import config_logger, disable_logging

def test_disable_logging():
    log_buffer = StringIO()
    with redirect_stdout(log_buffer):
        config_logger()
        with disable_logging():
            logging.info('This message should not appear')
    
    log_content = log_buffer.getvalue()
    assert 'This message should not appear' not in log_content, 'Disabled log message found in log_content'

