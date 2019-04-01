'''一些用到的函数：
                load_data, organize_data, distance, plot_path'''
from math import radians, fabs, sin, cos, asin, sqrt
import numpy as np
from mpl_toolkits.basemap import Basemap
from matplotlib import pyplot


'''读入台风txt数据，返回台风list'''
def load_data(filename, long):
    
    f = open(filename, 'r')
    lines = f.readlines() # 读取全部行, 并以列表方式返回
    f.close()
        
    typhoon = []
    i = 0
    
    for line in lines:
        if line[0:5] == '66666':
            typhoon.append([])
            i += 1
            continue
        typhoon[i-1].append(line[0:10] + ' ' + line[11:13] + line[4:6] + ' ' + line[13:26] + ' ' + line[32:34])
        # 时间串、强度、月份、纬度、经度、气压、风速
    for j in range(len(typhoon)):
        for k in range(len(typhoon[j])):
            typhoon[j][k] = typhoon[j][k].split()
            typhoon[j][k][:] = [float(va) for va in typhoon[j][k]][:]
            typhoon[j][k][3] = typhoon[j][k][3] / 10
            typhoon[j][k][4] = typhoon[j][k][4] / 10
            
#    # 限定研究的区域（进入E123°以西、N10°~N23.5°的保留，即影响到南海的）
#    index = list()
#    for j in range(len(typhoon)):
#        for i in range(len(typhoon[j])):
#            if ((typhoon[j][i][4]<123) and (typhoon[j][i][3]<23.5) and (typhoon[j][i][3]>10)):
#                index.append(j)
#                break
#    # 保留筛选的
#    typhoon_s = list()
#    for i in range(len(index)):
#        typhoon_s.append(typhoon[index[i]])
    typhoon_s = typhoon
            
    # 最大风速有缺省值的时间点删去(缺省值都是在台风首尾，不存在删掉后破坏连贯性的问腿)
    # 以及最大风速标记为9的时间点也删去
    for j in range(len(typhoon_s)):
        i = 0
        while i < len(typhoon_s[j]):
            if typhoon_s[j][i][6] == 0 or typhoon_s[j][i][6] == 9:
                del typhoon_s[j][i]
            else:
                i += 1
    
    # 由于场物理量的关系
    index = list()
    for j in range(len(typhoon_s)):
        for i in range(len(typhoon_s[j])):
            if typhoon_s[j][i][4] < 95 or typhoon_s[j][i][3] > 55:
                index.append(j)
                break
    # 删除
    for i in range(len(index)-1, -1, -1):
        del typhoon_s[index[i]]
    
    # 长度小于long的台风删去
    j = 0
    while j < len(typhoon_s):
        if len(typhoon_s[j]) < long:
            del typhoon_s[j]
        else:
            j += 1
    
    return typhoon_s


'''将台风list组织成定长样本，numpy形式'''
def organize_data(typhoon, sequence_length):
    
    result = []
    for j in range(len(typhoon)):
        for index in range(len(typhoon[j]) - sequence_length - 3):
            result.append(typhoon[j][index: (index + sequence_length + 4)])
        
    result = np.array(result)
    return result


'''将台风list组织成定长样本，numpy形式'''
def organize_data_48(typhoon, sequence_length):
    
    result = []
    for j in range(len(typhoon)):
        for index in range(len(typhoon[j]) - sequence_length - 7):
            result.append(typhoon[j][index: (index + sequence_length + 8)])
        
    result = np.array(result)
    return result


'''输入两点经纬度返回距离（km)'''
def distance(lat1, lon1, lat2, lon2):
    
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    lon1 = radians(lon1)
    lon2 = radians(lon2)
    
    dlat = fabs(lat1 - lat2)
    dlon = fabs(lon1 - lon2)
    
    earth_radius = 6370
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    dis = 2 * asin(sqrt(a)) * earth_radius
    
    return dis


'''输入台风list，画出编号a~b的台风路径'''
def plot_path(data, a, b):
    
    m = Basemap(projection='merc',llcrnrlat=0,urcrnrlat=60,llcrnrlon=50,urcrnrlon=200,lat_ts=20,resolution='l')
    m.drawcoastlines(linewidth=0.25, linestyle='solid', color='k', antialiased=1, ax=None, zorder=None)
    m.drawcountries(linewidth=0.25, linestyle='solid', color='k', antialiased=1, ax=None, zorder=None)
    m.fillcontinents(color='coral',lake_color='#689CD2')
    m.drawparallels(np.arange(0.,60.,20.), linewidth=0.25, labels=[0, 1, 0, 0])
    m.drawmeridians(np.arange(50.,200.,50.), linewidth=0.25, labels=[0, 0, 0, 1])
    m.drawmapboundary(color='k', linewidth=0.7, fill_color='#689CD2', zorder=None, ax=None)
                      
    for x in range(a, b):
        lon_value = list()
        lat_value = list()
        for eachtime in data[x]:
            lon_value.append(eachtime[4])
            lat_value.append(eachtime[3])
    
        xx, yy = m(lon_value, lat_value)
        m.plot(xx, yy, linewidth=0.5, color='k')
        pyplot.title("")
    
    return


'''对于0.5° * 0.5°的场数据，输入实际经纬度，输出在mat中的(行，列)'''
def get_idx(lat, lon):
    #纬度近似
    lat_diff = lat - int(lat)
    if lat_diff < 0.25:
        lat = int(lat) + 0.0
    elif lat_diff > 0.75:
        lat = int(lat) + 1.0
    else:
        lat = int(lat) + 0.5
    #经度近似
    lon_diff = lon - int(lon)
    if lon_diff < 0.25:
        lon = int(lon) + 0.0
    elif lon_diff > 0.75:
        lon = int(lon) + 1.0
    else:
        lon = int(lon) + 0.5
    #索引值
    lat_idx = 180 - (lat + 20) *2
    lon_idx = (lon - 80) *2
    
    lat_idx = int(lat_idx)
    lon_idx = int(lon_idx)
    
    return lat_idx, lon_idx

