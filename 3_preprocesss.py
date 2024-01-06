# _*_coding:utf_8 _*_
# @Time: 2024/1/6 3:27
# @Author: li zhuoyuan
# @File: processs.py
# @Contact: 21377153@buaa.edu.cn
# @Software: PyCharm

import pandas as pd
from datetime import datetime
import pytz

if __name__ == '__main__':
    pd.set_option('display.width', 1200)
    pd.set_option('display.max_columns', 30)

    df = pd.read_csv('data/traj.csv')
    result = pd.DataFrame(columns=['distance', 'start_x', 'start_y', 'target_x', 'target_y', 'start_speed', 'target_speed', 'holidays', 'hour', 'time'], index=range(397244))
    counts = pd.value_counts(df['traj_id'])

    min_x = 116.24
    max_x = 116.5
    min_y = 39.75
    max_y = 40.02

    row = 0
    # i为traj_id
    for i in range(22000): # 22000
        if i in counts.index:
            df_temp = df[df['traj_id'] == i]
            for j in range(1, len(df_temp)):
                if row % 1000 == 0:
                    print(row)
                result['distance'][row] = df_temp['current_dis'].iloc[j]
                result['start_x'][row] = (float(df_temp['coordinates'].iloc[0].strip('[]').split(',')[0]) - min_x) / (max_x - min_x)
                result['start_y'][row] = (float(df_temp['coordinates'].iloc[0].strip('[]').split(',')[1]) - min_y) / (max_y - min_y)
                result['target_x'][row] = (float(df_temp['coordinates'].iloc[j].strip('[]').split(',')[0]) - min_x) / (max_x - min_x)
                result['target_y'][row] = (float(df_temp['coordinates'].iloc[j].strip('[]').split(',')[1]) - min_y) / (max_y - min_y)
                result['start_speed'][row] = df_temp['speeds'].iloc[0]
                result['target_speed'][row] = df_temp['speeds'].iloc[j]
                result['holidays'][row] = df_temp['holidays'].iloc[j]
                result['hour'][row] = int(df_temp['time'].iloc[j][11:13])
                timestamp_dt = datetime.fromisoformat(df_temp['time'].iloc[j].replace("Z", "+00:00"))
                unix_timestamp = timestamp_dt.replace(tzinfo=pytz.UTC).timestamp()
                timestamp_dt0 = datetime.fromisoformat(df_temp['time'].iloc[0].replace("Z", "+00:00"))
                unix_timestamp0 = timestamp_dt0.replace(tzinfo=pytz.UTC).timestamp()
                result['time'][row] = int(unix_timestamp) - int(unix_timestamp0)
                row += 1
    print(result)
    result.to_csv('data/data.csv', index=False)
