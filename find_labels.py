# class for unionfind
import numpy as np
class disjointSet:
    def __init__(self):
        self.elements = {}

    def makeSet(self, x):
        if x not in self.elements:
            self.elements[x] = [x, 0]
        return

    def find(self, x):
        if self.elements[x][0][:37] == x:
            return x
        self.elements[x][0] = self.find(self.elements[x][0])
        return self.elements[x][0][:37]

    def union(self, x, y):
        xRoot = self.find(x)
        yRoot = self.find(y)
        if xRoot == yRoot:
            return
        if self.elements[xRoot][1] < self.elements[yRoot][1]:
            self.elements[xRoot][0] = yRoot
        elif self.elements[xRoot][1] > self.elements[yRoot][1]:
            self.elements[yRoot][0] = xRoot
        else:
            self.elements[yRoot][0] = xRoot
            self.elements[xRoot][1] += 1

    def print(self,dest):
        fp = open(dest,'w')
        for k, l in sorted(self.elements.items()):
            if l[0][:37] != k:
                new_k = self.find(k)
                self.elements[new_k][0] += '; ' + k
                del self.elements[k]
        for k, l in self.elements.items():
            print(l[0],file=fp)
        fp.close()

coord_pair = disjointSet()
#minx = [0][1] maxx = [0][3]
#miny = [0][2] maxy = [0][0]

def union_and_find(dest, raw_result):

    DIFF_PIXEL = 20
    sorted_coord = sorted(raw_result, key=lambda raw_coord: raw_coord[0][1])


    for i in range(0,len(sorted_coord)):
        minx_i = sorted_coord[i][0][1]
        maxx_i = sorted_coord[i][0][3]
        miny_i = sorted_coord[i][0][2]
        maxy_i = sorted_coord[i][0][0]
        width_i = maxx_i - minx_i
        height_i = maxy_i - miny_i
        for j in range(0, i):
            minx_j = sorted_coord[j][0][1]
            maxx_j = sorted_coord[j][0][3]
            miny_j = sorted_coord[j][0][2]
            maxy_j = sorted_coord[j][0][0]
            width_j = maxx_j - minx_j
            height_j = maxy_j - miny_j

            is_width = abs(width_i - width_j) < DIFF_PIXEL
            is_height = abs(height_i - height_j) < DIFF_PIXEL
            is_x = abs(minx_i - maxx_j) < DIFF_PIXEL
            is_maxy = abs(maxy_i - maxy_j) < DIFF_PIXEL
            is_miny = abs(miny_i - miny_j) < DIFF_PIXEL
            is_pair = is_width and is_height and is_x and is_maxy and is_miny

            if is_pair:
                coord_list_i = list(sorted_coord[i])
                coord_list_j = list(sorted_coord[j])
                coord_i = "{}, {:.3f}, {:6d},{:6d},{:6d},{:6d}".format(int(np.asscalar(coord_list_i[1])), #char
                                                              float(np.asscalar(coord_list_i[2])), #prob
                                                              int(np.asscalar(coord_list_i[0][1])),#minx
                                                              int(np.asscalar(coord_list_i[0][2])),#miny
                                                              int(np.asscalar(coord_list_i[0][3])),#maxx
                                                              int(np.asscalar(coord_list_i[0][0]))#maxy
                                                              )

                coord_j = "{}, {:.3f}, {:6d},{:6d},{:6d},{:6d}".format(int(np.asscalar(coord_list_j[1])), #char
                                                              float(np.asscalar(coord_list_j[2])), #prob
                                                              int(np.asscalar(coord_list_j[0][1])),#minx
                                                              int(np.asscalar(coord_list_j[0][2])),#miny
                                                              int(np.asscalar(coord_list_j[0][3])),#maxx
                                                              int(np.asscalar(coord_list_j[0][0]))#maxy
                                                              )
                coord_pair.makeSet(coord_i)
                coord_pair.makeSet(coord_j)
                coord_pair.union(coord_i, coord_j)
                break
    coord_pair.print(dest)