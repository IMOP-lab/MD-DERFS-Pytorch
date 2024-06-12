import logging


class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }  # 日志级别关系映射

    def __init__(self, filename, level='info'):
        self.logger = logging.getLogger(filename)
        format = logging.Formatter('%(asctime)s - %(message)s')  # 设置日志格式
        self.logger.setLevel(self.level_relations.get(level))  # 设置日志级别

        sh = logging.StreamHandler()  # 往屏幕上输出
        # sh.setFormatter(format)  # 设置屏幕上显示的格式
        self.logger.addHandler(sh)  # 把对象加到logger里

        th = logging.FileHandler(filename)
        th.setFormatter(format)
        self.logger.addHandler(th)


if __name__ == '__main__':
    log = Logger('train_log.txt').logger
    log.debug('debug')
    log.info('info')
    log.warning('警告')
    log.error('报错')
    log.critical('严重')
