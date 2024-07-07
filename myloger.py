import logging

# 配置日志
def setup_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # 创建文件handler并设置级别为INFO
    fh = logging.FileHandler('Traininfo.log')
    fh.setLevel(logging.INFO)
    
    # 创建console handler并设置级别为INFO
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # 创建formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 添加formatter到handlers
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    # 给logger添加handlers
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

# 运行配置函数并获得一个配置好的logger实例
# 注意：在多次调用 getLogger(__name__) 时，会获得同一个logger实例；这样可以避免创建重复的handlers
logger = setup_logger()
