import logging
from datetime import datetime
from typing import Literal


class MessageHandler:
    def __init__(self, display: Literal['console',
                                        'logs',
                                        'webpanel'] = 'console',
                 webpanel=None):
        self.display = display

        if display == 'logs':
            now_datetime = datetime.now()
            log_filename = (f'logs/{now_datetime.day}.{now_datetime.month}.'
                            f'{now_datetime.year} {now_datetime.hour}:'
                            f'{now_datetime.minute}:{now_datetime.second}.log')
            logging.basicConfig(level=logging.INFO,
                                filename=log_filename,
                                filemode='w',
                                format='%(asctime)s %(levelname)s %(message)s')
            self.logger = logging.getLogger('handler')

        if display == 'webpanel':
            if webpanel is None:
                raise ValueError('webpanel is required')
            self.webpanel = webpanel

    def receive(self, message: str):
        match self.display:
            case 'console':
                print(message)
            case 'logs':
                self.logger.info(message)
            case 'webpanel':
                pass
