logconfig_dict = {
    'formatters': {
        'simple': {
            'format': '%(asctime)s [%(process)d] %(message)s',
            'datefmt': '[%Y/%m/%d %H:%M:%S %Z]'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.FileHandler',
            'formatter': 'simple',
            'filename': '../../data/ego.logs',
        },
        'error_console': {
            'class': 'logging.FileHandler',
            'formatter': 'simple',
            'filename': '../../data/ego.logs',
        },
    },
    'version': 1,
    'disable_existing_loggers': False,
}
