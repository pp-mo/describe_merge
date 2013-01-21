'''
Define a 'cube notation' scheme for merge testing.

Can represent 2d and 3d cubes of a limited size, and extracted sections.
Also *create* cubes of the correct structure from a notation string.
The coordinate names associated with dimesions are fixed.
The dimension points values in each dimension(coord
Examples:
  2d, 2*3
'''

import numpy as np
import numpy.lib.index_tricks as numpy_index_tricks

import iris
import iris.tests.stock

# Dimension coordinate names to be used (*assumed* in notation code).
_NOTATION_COORD_NAMES = ['baz','bar','foo']
# Base-characters for naming the dimension points.
_NOTATION_COORD_START_CHARS = ['P','a','1']
  # so e.g. : a1, ab234, Qa1, QSa12

# Testcube dimensions.
_TEST_NZ = 5
_TEST_NY = 3
_TEST_NX = 4
_NOTATION_DIM_SIZES = [_TEST_NZ, _TEST_NY, _TEST_NX]

def stuff_array_with_indices(array):
    # Encode the cell indexes into the values of an array.
    # E.G. array(2,3) --> [[11,12,13],[21,22,23]]
    array[...] = np.zeros(array.shape)
    dim_arrays = numpy_index_tricks.mgrid[[slice(n) for n in array.shape]]
    for dim_array in dim_arrays:
        array[...] = 10.0*array + dim_array + 1


# Create testcubes...

# load stock 2d cube
t2d = iris.tests.stock.simple_2d(with_bounds=False)

# check as expected (as assumed for 'fake' operation)
for (coord_name, coord_len) in \
        zip(_NOTATION_COORD_NAMES[1:3],
            _NOTATION_DIM_SIZES[1:3]):
    coord = t2d.coord(coord_name)
    assert coord.shape == (coord_len,)

# construct 3d test cube
t3d = [t2d.copy() for iz in range(_TEST_NZ)]
for iz in range(_TEST_NZ):
  t3d[iz].add_aux_coord(
      iris.coords.DimCoord([iz],
                           long_name=_NOTATION_COORD_NAMES[0]))
t3d = iris.cube.CubeList(t3d).merge()[0]

# blast contents so we can easily see where indexing has taken us
for array in (t2d.data, t3d.data):
    stuff_array_with_indices(array)


# Define point values and notation characters for our 2d/3d test coords.
_NOTATION_COORD_CHARS = {}
for (start_char, coord_name) in \
        zip(_NOTATION_COORD_START_CHARS,
            _NOTATION_COORD_NAMES):
    points = t3d.coord(coord_name).points
    ord_0 = ord(start_char)
    val2char_dict = {pt_val: chr(ord_0 + i_pt) 
                     for (i_pt, pt_val) in enumerate(points)}
    _NOTATION_COORD_CHARS[coord_name] = val2char_dict

def cube_notation_string(cube):
    """ Make a 'merge notation' string showing dimension points in a cube."""
    out_str = ''
    for name in _NOTATION_COORD_NAMES:
        coords = cube.coords(name)
        if len(coords):
            coord, = coords
            v2c_dict = _NOTATION_COORD_CHARS[name]
            out_str += ''.join(v2c_dict[v] for v in coord.points)

    return out_str

def cube_from_notation_string(cube_string):
    """
    Construct a cube from a 'merge notation' string.

    Note: may specify out-of-order selection on a coordinate
    For example, 'ca312'.
    """
    coord_indices = {coord_name:[] for coord_name in _NOTATION_COORD_NAMES}
    for a_char in cube_string:
        coord_name, = [name 
                       for name in _NOTATION_COORD_NAMES
                       if a_char in _NOTATION_COORD_CHARS[name].values()]
        char_index = _NOTATION_COORD_CHARS[coord_name].values().index(a_char)
        coord_indices[coord_name].append(char_index)
    #print 'coord indices:', coord_indices
    if len(coord_indices['baz']) == 0:
#        print '2d slice: ',coord_indices['bar'], coord_indices['foo']
        result_cube = t2d
        result_cube = result_cube[coord_indices['bar'], :]
        result_cube = result_cube[:, coord_indices['foo']]
    else:
#        print '3d slice: ',coord_indices['baz'], coord_indices['bar'], coord_indices['foo']
        result_cube = t3d
        result_cube = result_cube[coord_indices['baz'], :, :]
        result_cube = result_cube[:, coord_indices['bar'], :]
        result_cube = result_cube[:, :, coord_indices['foo']]
    return result_cube

_TEST_NOTATION_SPECS = [
    ('t2d', t2d, 'abc1234'),
    ('t3d', t3d, 'PQRSTabc1234'),
    ('t3d[0,1:2,2:]', t3d[0,1:2,2:], 'Pb34'),
    ('t3d[[2,0,1], 1:3, 3:]', t3d[[2,0,1], 1:3, 3:], 'RPQbc4'),
    ('t3d[1:2, [3,0], 2:3]', t3d[1:2, [2,0], 2:3], 'Qca3'),
    ('t2d[:,0]', t2d[:,0], 'abc1'),
    ('t2d[1,2:]', t2d[1,2:], 'b34'),
    ('t3d[:,:,1]', t3d[:,:,1], 'PQRSTabc2'),
    ('t3d[:,2,:]', t3d[:,2,:], 'PQRSTc1234'),
]
    
def test_cube_notation_string():
    for (tst_str, tst_cube, expect_str) in _TEST_NOTATION_SPECS:
        print 'test cube = ', tst_str
#        print tst_cube
        tst_result = cube_notation_string(tst_cube)
        print '  --> ', tst_result
        if tst_result != expect_str:
            print ' !WRONG!, expected --> ', expect_str
            assert(tst_result == expect_str)

print 't2d'
print t2d
print
print 't3d'
print t3d
print

print
print '-----------------------------------------'
print 'TEST cube_notation_string...'
test_cube_notation_string()
print

def test_cube_from_notation_string():
    tst_specs = [
        'ab12',
        'Pac21',
        'PQa12',
    ]
    tst_specs += [expect_str for (tst_str, tst_cube, expect_str) in _TEST_NOTATION_SPECS]
    for tst_str in tst_specs:
        print tst_str
        tst_cube = cube_from_notation_string(tst_str)
        print tst_cube
        tst_result = cube_notation_string(tst_cube)
        print '  (back-convert -> {})'.format(tst_result)
        print
        assert(tst_result == tst_str)

print
print '-----------------------------------------'
print 'TEST cube_from_notation_string...'
test_cube_from_notation_string()
print


def _reduce_cube(cube):
    # Replace vector coords of length 1 with scalars to enable merges.
    while cube.ndim > 0 and cube.shape[0] == 1:
        cube = cube[0]
    while cube.ndim > 1 and cube.shape[1] == 1:
        cube = cube[:, 0]
    while cube.ndim > 2 and cube.shape[2] == 1:
        cube = cube[:, :, 0]
    return cube

def test_cube_merges():
    test_merge_specs = [
        ( ['ab1', 'ab2'], ['ab12']),
        ( ['a1', 'a3', 'a2'], ['a123']),
        ( ['a1', 'b3', 'a3', 'b1'], ['ab13']),
    ]
    for (in_speclist, out_speclist_expected) in test_merge_specs:
        print '  cubes merge input = ', ', '.join(in_speclist)
        in_cubelist = iris.cube.CubeList([
            _reduce_cube(cube_from_notation_string(cube_string))
            for cube_string in in_speclist])
        out_cubelist_actual = in_cubelist.merge()
        out_speclist_actual = [cube_notation_string(cube) 
                               for cube in out_cubelist_actual]
        print '          output = ', ', '.join(out_speclist_actual)
        if out_speclist_actual != out_speclist_expected:
            print ' !XXXX! expected = ', ', '.join(out_speclist_expected)
            assert(out_speclist_actual == out_speclist_expected)
        print

print
print '-----------------------------------------'
print 'TEST cube merges...'
test_cube_merges()
