import datetime

# 获取当前时间
now = datetime.datetime.now()

# 格式化为字符串
time_str = now.strftime('%Y-%m-%d_%H_%M_%S')
print(time_str)