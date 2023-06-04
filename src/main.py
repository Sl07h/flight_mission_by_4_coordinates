import argparse
import os
import numpy as np
import folium
from folium.plugins import Draw, MeasureControl
from math import sqrt, sin, cos, radians, tan, acos, degrees

LENGTH_OF_1_DEGREE_IN_METERS = 111134.861111111111111111111111111


def calc_image_size(height, fov, W_const, H_const):
    diag = 2.0 * height * tan(radians(fov / 2.0))
    divizor = sqrt(W_const**2 + H_const**2)
    W = diag * W_const / divizor
    H = diag * H_const / divizor
    return W, H, diag


def convert_to_coord(coords):
    return coords[0] + coords[1]/60.0 + coords[2]/3600.0


def convert_lat_to_meters(latitude):
    ''' Широта. https://v-ipc.ru/guides/coord '''
    return latitude * LENGTH_OF_1_DEGREE_IN_METERS


def convert_meters_to_lat(latitude_m):
    ''' Широта '''
    return latitude_m / LENGTH_OF_1_DEGREE_IN_METERS


def convert_lon_to_meters(longitude, latitude):
    ''' Долгота '''
    ratio = LENGTH_OF_1_DEGREE_IN_METERS * cos(radians(latitude))
    return longitude * ratio


def convert_meters_to_lon(longitude_m, latitude):
    ''' Долгота '''
    ratio = LENGTH_OF_1_DEGREE_IN_METERS * cos(radians(latitude))
    return longitude_m / ratio


def rotate_in_meters(p, c, angle):
    # вращаем точку вокруг другой на заданный угол
    sin_a = sin(angle)
    cos_a = cos(angle)
    y, x = p - c
    X = x*cos_a - y*sin_a
    Y = x*sin_a + y*cos_a
    return [Y+c[0], X+c[1]]


def rotate_in_degrees(p, c, angle, lat):
    # вращаем точку вокруг другой на заданный угол, на заданной широте
    sin_a = sin(angle)
    cos_a = cos(angle)
    y, x = p - c
    y, x = calc_vector_to_meters(p - c, lat)
    X = x*cos_a - y*sin_a
    Y = x*sin_a + y*cos_a
    Y, X = calc_vector_from_meters([Y, X], lat)
    return [Y+c[0], X+c[1]]


def equation_by_two_points(p0, p1):
    '''
    returns b,c from "x = b*y + c"

        x  - x0   y  - y0
    1.  ------- = -------
        x1 - x0   y1 - y0

    2. (x  - x0)*(y1 - y0) = (y  - y0)*(x1 - x0)

    3. (x  - x0) = (y  - y0)*(x1 - x0)/(y1 - y0)

    4. x = (y  - y0)*(x1 - x0)/(y1 - y0) + x0

    5. x = y*(x1 - x0)/(y1 - y0) + x0 - y0*(x1 - x0)/(y1 - y0)

    '''
    y0, x0 = p0
    y1, x1 = p1
    # x = b*y + c
    b = (x1 - x0)/(y1 - y0)
    c = x0 - y0*(x1 - x0)/(y1 - y0)
    return b, c


def calc_ratio(image_cross):
    # считаем во сколько раз каждая сторона должна быть меньше в рассчётах, чтобы
    # процент перекрытия был больше или равен заданному
    return (1.0 - sqrt(1.0 - image_cross)) / 2.0


def calc_vector_to_meters(vector, lat):
    y, x = vector
    y = convert_lat_to_meters(y)
    x = convert_lon_to_meters(x, lat)
    return [y, x]


def calc_vector_from_meters(vector, lat):
    y, x = vector
    y = convert_meters_to_lat(y)
    x = convert_meters_to_lon(x, lat)
    return [y, x]


# -------------------------------------------------------------------------------
# Параметры полёта -------------------------------------------------------------
# -------------------------------------------------------------------------------
W_const = 5472.0        # ширина
H_const = 3648.0        # высота
# fov = 77.0              # угол обзора камеры DJI Fantom 4 PRO v2.0
fov = 65.5              # угол обзора камеры DJI Mavic 2

image_cross = 0.25      # пересечение изображений на k%
height = 3.0            # высота полёта
speed_m_s = 0.5         # скорость в метрах в секунду
photo_interval = 2.0    # интервал между кадрами в секундах
path_4_points = 'icg.txt'
# path_4_points = 'field0.txt'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Инструмент для создания лётного задания для Litchi')
    parser.add_argument('--path_4_points', help='путь к файлу с 4 точками')
    args = parser.parse_args()
    path_4_points = args.path_4_points
    file_name = os.path.splitext(os.path.basename(path_4_points))[0]

    W, H, diag = calc_image_size(height, fov, W_const, H_const)
    print(f'Исходный размер поля под дроном:\t\t{W:.3f}x{H:.3f}м')
    s_old = W*H
    w_old = W
    h_old = H
    ratio = calc_ratio(image_cross)
    dw = W * ratio
    dh = H * ratio
    W = W - 2*dw
    H = H - 2*dh
    s_new = W*H
    s_ratio = s_new / s_old
    capture_interval_s = (2*(H / 2.0 - dh) + dh) / speed_m_s
    print(f'При перекрытии {int(image_cross*100)}% размер поля под дроном:\t{W:.3f}x{H:.3f}м')
    print(f'Отношение площадей:\t\t\t\t{s_ratio:.3f} и {image_cross:.3f}')
    print(f'Интервал между снимками:\t\t\t{capture_interval_s:.3f} {photo_interval:.3f}')

    # геометрия для простоты кода
    x_axis = np.array([1.0, 0.0])
    y_axis = np.array([0.0, 1.0])
    i0 = 9000
    i1 = 9000
    max_len = -9000
    max_lat = -9000

    borders = np.ndarray((4, 2), np.float64)
    lengths = np.ndarray(4, np.float64)
    lengths_i = np.ndarray((4, 2), np.float64)
    points = np.loadtxt(path_4_points)
    if points.shape == (4,6):
        for i, point in enumerate(points):
            borders[i][0] = convert_to_coord(point[:3])
            borders[i][1] = convert_to_coord(point[3:])
    else:
        borders = points.copy()

    W_lat = convert_meters_to_lat(W)
    H_lat = convert_meters_to_lat(H)
    W_lon = convert_meters_to_lon(W, borders[0][0])
    H_lon = convert_meters_to_lon(H, borders[0][0])

    W_lat_old = convert_meters_to_lat(w_old)
    H_lat_old = convert_meters_to_lat(h_old)
    W_lon_old = convert_meters_to_lon(w_old, borders[0][0])
    H_lon_old = convert_meters_to_lon(h_old, borders[0][0])

    m_lat, m_long = borders.mean(axis=0)
    m = folium.Map([m_lat, m_long], tiles=None,
                   prefer_canvas=True, control_scale=True, zoom_start=18)
    base_map = folium.FeatureGroup(name='Basemap', overlay=True, control=False)
    folium.TileLayer(tiles='OpenStreetMap', max_zoom=23).add_to(base_map)
    base_map.add_to(m)

    borders_closed = np.append(borders.copy(), [borders.copy()[0]], axis=0)
    layer_source = folium.FeatureGroup(name='оригинал', show=False)
    layer_traces = folium.FeatureGroup(name='пути', show=True)
    layer_frames = folium.FeatureGroup(name='рамки', show=False)
    layer_result = folium.FeatureGroup(name='результат', show=False)

    folium.PolyLine(borders_closed, color='#007800').add_to(layer_source)
    folium.PolyLine(borders_closed, color='#007800').add_to(layer_result)

    for i in range(4):
        p0 = borders[i % 4]
        p1 = borders[(i+1) % 4]
        d_lat = abs(p0[0] - p1[0])
        d_lon = abs(p0[1] - p1[1])
        d_lat_meters = convert_lat_to_meters(d_lat)
        d_lon_meters = convert_lon_to_meters(d_lon, p0[0])
        edge_len = sqrt(d_lat_meters**2 + d_lon_meters**2)
        if edge_len > max_len:
            max_len = edge_len
            i0 = i % 4
            i1 = (i+1) % 4

    # ветктор должен быть направлен вдоль оси x
    if borders[i0][1] > borders[i1][1]:
        i1, i0 = i0, i1

    l = np.array([0, 1, 2, 3])
    l = l[l != i0]
    l = l[l != i1]
    i2, i3 = l
    if i2 > i3:
        i2, i3 = i3, i2

    #    2--------3
    #   /          \
    #  /            \
    # 0--------------1
    start_point_i = i0
    start_point = borders[start_point_i]
    point0 = borders[i0]
    point1 = borders[i1]
    point2 = borders[i2]
    point3 = borders[i3]

    vector = point1 - point0
    folium.Marker(list(point0), popup='i0').add_to(layer_source)
    folium.Marker(list(point1), popup='i1').add_to(layer_source)
    folium.Marker(list(point2), popup='i2').add_to(layer_source)
    folium.Marker(list(point3), popup='i3').add_to(layer_source)

    # повернём координаты вдоль оси x
    vector_m = calc_vector_to_meters(vector, m_lat)
    angle = degrees(
        acos(np.dot(x_axis, vector_m[::-1]) / (np.linalg.norm(vector_m))))
    if point1[0] > point0[0]:
        angle = -angle
    print(angle, vector_m, x_axis)

    for i, point in enumerate(borders):
        borders[i] = rotate_in_degrees(point, start_point, radians(angle), m_lat)
    borders_closed = np.append(borders.copy(), [borders.copy()[0]], axis=0)
    folium.PolyLine(borders_closed, color='#007800').add_to(layer_frames)
    folium.PolyLine(borders_closed, color='#007800').add_to(layer_traces)

    for point in borders:
        if point[0] > max_lat:
            max_lat = point[0]

    n_count = int((max_lat - point1[0]) / W_lat)
    n_count2 = convert_lat_to_meters(max_lat - point1[0]) / W
    roads = np.ndarray((n_count, 2), np.float64)

    #    2--------3
    #   /          \
    #  /            \
    # 0--------------1

    # левое ребро
    v02 = point2 - point0
    dy02 = point2[0]-point0[0]
    n02 = dy02 / W_lat
    v02 = v02 / n02
    n02 = int(n02)+2

    # правое ребро
    v13 = point3 - point1
    dy13 = point3[0]-point1[0]
    n13 = dy13 / W_lat
    v13 = v13 / n13
    n13 = int(n13)+2

    max_n = max(n02, n13)
    trace_l = np.ndarray((max_n-1, 2), np.float64)
    trace_c = []
    trace_r = np.ndarray((max_n-1, 2), np.float64)
    rotation_points = np.ndarray((max_n-1, 4), np.float64)

    points_l = np.ndarray((max_n, 2), np.float64)
    points_l[0] = point0
    for i in range(n02):
        point = point0 + v02*i
        points_l[i] = point
    p = point0+0.5*v02
    for i in range(n02-1):
        trace_l[i] = p + i*v02

    points_r = np.ndarray((max_n, 2), np.float64)
    points_r[0] = point1
    for i in range(n13):
        point = point1 + v13*i
        points_r[i] = point
    p = point1+0.5*v13
    for i in range(n13-1):
        trace_r[i] = p+i*v13

    min_n = min(n02, n13)
    if n02 < n13:
        point_start = points_l[n02-1]
    else:
        point_start = points_r[n13-1]

    max_n = max(n02, n13)
    b, c = equation_by_two_points(point2, point3)
    for i in range(max_n - min_n):
        y = point_start[0]+W_lat*(i)
        x = b*y+c
        min_n += 1
        if n02 < n13:
            points_l[n02] = [y+W_lat, x]
            trace_l[n02-1] = [y+W_lat/2.0, x]
            n02 += 1
        else:
            points_r[n13] = [y+W_lat, x]
            trace_r[n13-1] = [y+W_lat/2.0, x]
            n13 += 1

    for i in range(min_n):
        folium.PolyLine([points_l[i], points_r[i]]).add_to(layer_traces)

    for row_i, (pl, pr) in enumerate(zip(trace_l, trace_r)):
        width_lon_m = convert_lon_to_meters(pr[1] - pl[1], borders[0][0])
        n = round(width_lon_m / H)
        dx = (pr[1] - pl[1]) / n

        l_c = []
        for i in range(n):
            point = pl.copy() + np.array([0.0, dx/2.0 + dx*i])
            frame = np.array([
                [W_lat_old/2.0,  H_lon_old/2.0],
                [-W_lat_old/2.0,  H_lon_old/2.0],
                [-W_lat_old/2.0, -H_lon_old/2.0],
                [W_lat_old/2.0, -H_lon_old/2.0],
                [W_lat_old/2.0,  H_lon_old/2.0]
            ])
            folium.PolyLine(list(point+frame)).add_to(layer_frames)

            frame = np.array([
                [w_old/2.0,  h_old/2.0],
                [-w_old/2.0,  h_old/2.0],
                [-w_old/2.0, -h_old/2.0],
                [w_old/2.0, -h_old/2.0],
                [w_old/2.0,  h_old/2.0]
            ])

            point = rotate_in_degrees(point, start_point, radians(-angle), m_lat)
            for j, p in enumerate(frame):
                p = rotate_in_meters(p, np.array([0.0, 0.0]), radians(-angle))
                y, x = p
                y = convert_meters_to_lat(y)
                x = convert_meters_to_lon(x, m_lat)
                frame[j] = np.array([y, x])
            l_c += [list(point)]
            folium.PolyLine(list(point+frame)).add_to(layer_result)

        trace_c += [l_c]
        rotation_points[row_i] = np.array([*l_c[0], *l_c[-1]])
        folium.PolyLine([pl, pr], color='#FFDA47').add_to(layer_traces)

    for row in rotation_points:
        pl = row[:2]
        pr = row[2:]
        folium.PolyLine([pl, pr], color='#FFDA47').add_to(layer_result)

    # np.savetxt('rotation_points_{}%.txt'.format(
    #     int(image_cross*100)), rotation_points, fmt='%2.14f')

    layer_source.add_to(m)
    layer_traces.add_to(m)
    layer_frames.add_to(m)
    layer_result.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    MeasureControl().add_to(m)
    Draw(export=True).add_to(m)
    m.save(f'maps/fight_mission_{file_name}.html')