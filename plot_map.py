"""
Course: CSI5155
Tiffany Nien Fang Cheng
Group 33
Student ID: 300146741
"""
import json
import matplotlib.pyplot as plt
from descartes import PolygonPatch
from matplotlib.patches import Polygon
from shapely.geometry import mapping, shape, Point, MultiPoint
from numpy import genfromtxt
import pandas as pd

"""
The lib draws the contour of wards in ottawa and also the class are marked in different color
"""

with open("ottawa_municipal_geo.txt") as json_file:
    json_data = json.load(json_file) # or geojson.load(json_file)

with open("transportation_accident.csv",  encoding="utf-8") as cvs_file:
    cvs_data = pd.read_csv(cvs_file, sep=',')
    # cvs_data = genfromtxt(cvs_file, delimiter=',')
    # my_data = genfromtxt('my_file.csv', delimiter=',')

pd.options.display.max_columns = 500
print(cvs_data.head(2))
pd.options.display.max_columns = 0
cvs_np = cvs_data.to_records()
lon = cvs_np['LONGITUDE']
lag = cvs_np['LATITUDE']
class_col = cvs_np['CLASS_OF_ACCIDENT']

# print(json_data['features'][1]['geometry'].keys())
# print(len(json_data['features']))
for it in json_data['features']:
    # import ipdb
    # ipdb.set_trace()
    poly = it['geometry']
    BLUE = "#6699cc"
    fig = plt.figure()
    # fig.title(json_data['features'][0]["attributes"]['MUNICIPALN'])
    ax = fig.gca()
    location_name = it["properties"]['MUNICIPALN']
    plt.title(location_name)

    ax.add_patch(PolygonPatch(poly, fc="cyan", ec=BLUE, alpha=0.5, zorder=2))
    ax.axis('scaled')
    # plt.show()

    poly_sh = shape(poly)
    print(poly_sh.contains(Point(-76.02934248,45.30121701)))

    in_manul = list()

    lon_lag = map(Point,lon,lag)
    lon_lag = list(lon_lag)

    # print(list(log_lag))
    # do color mapping for classes
    for i in range(len(lon_lag)):
        it=lon_lag[i]
        col=class_col[i]
        if poly_sh.contains(it):
            in_manul.append(it)
            if '01' in col:
                col='r'
            if '02' in col:
                col='b'
            if '03' in col:
                col='g'
            plt.scatter(it.x, it.y, c=col)

    plt.savefig('{}.png'.format(location_name))
    plt.close()


