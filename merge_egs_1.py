'''
Created on Jan 4, 2013

@author: itpp
'''

import iris
import iris.tests.stock


def tm(cubes):    
  return iris.cube.CubeList(cubes).merge()

t2d = iris.tests.stock.simple_2d(with_bounds=False)
print 't2d'
print t2d

nz = 5
t3d = [t2d.copy() for iz in range(nz)]
for iz in range(nz):
  t3d[iz].add_aux_coord(iris.coords.DimCoord([iz],long_name='baz'))
t3d = iris.cube.CubeList(t3d).merge()[0]
print 't3d'
print t3d

# order we wish to treat the names in (which is *not* in dimension order)
coord_names = ['baz','foo','bar']
# base-characters for naming the points
coord_id_start_chars = ['P','a','1']
  # so e.g. : a1, ab234, Qa1, QSa12

# point values and corresponding characters for each coord (by name)
coord_pts = {}
for (start_char, coord_name) in zip(coord_id_start_chars, coord_names):
    points = t3d.coord(coord_name).points
    ord_0 = ord(start_char)
    val2char_dict = {pt_val: chr(ord_0 + i_pt) for (i_pt, pt_val) in enumerate(points)}
    coord_pts[coord_name] = val2char_dict

def cube_2_notation_string(cube):
    """ Make a 'merge notation' string showing dimension points in a cube."""
    out_str = ''
    for name in coord_names:
        coords = cube.coords(name)
        if len(coords):
            coord, = coords
            v2c_dict = coord_pts[name]
            out_str += ''.join(v2c_dict[v] for v in coord.points)

    return out_str

def cube_from_notation_string(cube_string):
    """ Construct a cube from a 'merge notation' string."""
    coord_indices = {coord_name:[] for coord_name in coord_names}
    for this_char in cube_string:
        coord_name, = [name for name in coord_names
                            if this_char in coord_pts[name].values()]
        point_index = coord_pts[coord_name].values().index(this_char)
        coord_indices[coord_name].append(point_index)

    if len(coord_indices['baz']) == 0:
        result_cube = t2d[coord_indices['bar'], coord_indices['foo']]
    else:
        result_cube = t3d[coord_indices['baz'], coord_indices['bar'], coord_indices['foo']]
    return result_cube

# test cube_2_notation_string 
# print cube_2_notation_string(t2d)
# print cube_2_notation_string(t3d)
# print cube_2_notation_string(t3d[0,1:2,2:])

# test cube_from_notation_string
for tst_str in [
            'ab12',
            'Pac21',
            'PQa12',
        ]:
    print tst_str
    t = cube_from_notation_string(tst_str)
    print t


def mergetest_string(cubes):
    """ Make a notation string for a mergetest result. """
    cubelist = iris.cube.CubeList(cubes)
    result = ', '.join(cube_2_notation_string(cube) for cube in cubelist)
    merged_cubelist = cubelist.merge()
    merge_strings = [cube_2_notation_string(cube) for cube in merged_cubelist]
    if len(merge_strings) == 1:
        # single cube result
        result += '  --->  ' + merge_strings[0]
    else:
        # multiple result
        result += '  --[]-->  [' \
            + ', '.join(merge_strings) \
            + ']'
    return result

# test whole-test idea
def test_merge(in_specs, out_specs=None):
    """
    Perform a mergetest, print result, and whether successful.

    Args:
    * out_specs
        the expected result.  Can only fail if this is given.
    """
    result_string = mergetest_string(in_specs)
