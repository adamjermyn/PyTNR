import logging

def makeLogger(name, level):
	logger = logging.getLogger(name)
	if level == 'info':
		logger.setLevel(logging.INFO)
	elif level == 'debug':
		logger.setLevel(logging.DEBUG)
	elif level == 'warning':
		logger.setLevel(logging.WARNING)
	elif level == 'error':
		logger.setLevel(logging.ERROR)
	elif level == 'critical':
		logger.setLevel(logging.CRITICAL)

	return logger
