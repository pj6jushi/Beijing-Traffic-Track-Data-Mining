# _*_coding:utf_8 _*_
# @Time: 2024/1/3 1:24
# @Author: li zhuoyuan
# @File: try.py
# @Contact: 21377153@buaa.edu.cn
# @Software: PyCharm

import pandas as pd

if __name__ == '__main__':
    pd.set_option('display.width', 1200)
    pd.set_option('display.max_columns', 30)

    road = pd.read_csv('data/road.csv')
    node = pd.read_csv('data/node.csv')

    # 创建一个dataframe
    df = pd.DataFrame(columns=['WKT', 'id', 'source', 'target'])
    df['id'] = road['id']

    # 创建一个字典
    dic = {}
    for i in range(len(node)):
        dic[node['coordinates'][i].strip('[]')] = node['geo_id'][i]

    for i in range(len(road)):
        a = road['coordinates'][i][1:-1].strip('[]').split('], [')
        start = a[0]
        end = a[-1]
        df['source'][i] = int(dic[start])
        df['target'][i] = int(dic[end])
        for j in range(len(a)):
            a[j] = a[j].replace(',', '')
        df['WKT'][i] = 'LINESTRING (' + ','.join(a) + ')'
    #     flag_start = 0
    #     flag_end = 0
    #     for i in node['coordinates']:
    #         if i.strip('[]') == start:
    #             flag_start = 1
    #             break
    #     for i in node['coordinates']:
    #         if i.strip('[]') == end:
    #             flag_end = 1
    #             break
    #     if flag_start == 0:
    #         print(start)
    #     if flag_end == 0:
    #         print(end)

    print(df)
    df.to_csv('fmm/data/edges.csv', index=False)