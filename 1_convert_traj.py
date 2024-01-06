# _*_coding:utf_8 _*_
# @Time: 2024/1/3 16:15
# @Author: li zhuoyuan
# @File: 1_convert_traj.py
# @Contact: 21377153@buaa.edu.cn
# @Software: PyCharm

import pandas as pd
from datetime import datetime
import pytz

if __name__ == '__main__':
    traj = pd.read_csv('data/traj.csv')
    df = pd.DataFrame(columns=['id;x;y;timestamp'])
    # for i in range(100):
    #     timestamp_dt = datetime.fromisoformat(traj['time'][i].replace("Z", "+00:00"))
    #     unix_timestamp = int(timestamp_dt.replace(tzinfo=pytz.UTC).timestamp())
    #     print(unix_timestamp)
    for i in range(len(traj)):
        timestamp_dt = datetime.fromisoformat(traj['time'][i].replace("Z", "+00:00"))
        unix_timestamp = timestamp_dt.replace(tzinfo=pytz.UTC).timestamp()
        df.loc[i] = str(traj['traj_id'][i]) + ';' + traj['coordinates'][i].strip('[]').split(',')[0] + ';' + traj['coordinates'][i].strip('[]').split(',')[1] + ';' + str(int(unix_timestamp))
        if i % 1000 == 0:
            print(i, len(traj))
    print(df)
    df.to_csv('fmm/data/traj_timestamps.csv', index=False)
