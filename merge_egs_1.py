'''
Created on Jan 4, 2013

@author: itpp
'''

import numpy as np
import numpy.lib.index_tricks as numpy_index_tricks

# dimension coordinate names we will use/assume
coord_names = ['baz','bar','foo']
# base-characters for naming the points
coord_id_start_chars = ['P','a','1']
  # so e.g. : a1, ab234, Qa1, QSa12
# size of constructed 3rd dimension
_nz = 5
# size of dimensions
coord_lens = [_nz, 3, 4]

def stuff_array_with_indices(array):
#    array[...] = 100*np.ones(array.shape)
    array[...] = np.zeros(array.shape)
    dim_arrays = numpy_index_tricks.mgrid[[slice(n) for n in array.shape]]
    for dim_array in dim_arrays:
        array[...] = 10.0*array + dim_array + 1

do_real = True
if do_real:
    import iris
    import iris.tests.stock

    # load stock 2d cube
    t2d = iris.tests.stock.simple_2d(with_bounds=False)

    # check as expected (as assumed for 'fake' operation)
    for (coord_name, coord_len) in zip(coord_names[1:3], coord_lens[1:3]):
        coord = t2d.coord(coord_name)
        assert coord.shape == (coord_len,)

    # construct 3d test cube
    t3d = [t2d.copy() for iz in range(_nz)]
    for iz in range(_nz):
      t3d[iz].add_aux_coord(iris.coords.DimCoord([iz],long_name=coord_names[0]))
    t3d = iris.cube.CubeList(t3d).merge()[0]

    # blast contents so we can easily see where indexing has taken us
    for array in (t2d.data, t3d.data):
        stuff_array_with_indices(array)

else:   # (not do_real)
    import numpy as np

    class DummyCoord(object):
        def __init__(self, points):
            self.points = points

    class DummyCube(object):
        def __init__(self, array=None, dims_list=None):
            if array is not None:
                if not isinstance(array, np.ndarray):
                    dims_list = array
                    array = None
                else:
                    n_dims = len(array.shape)
                    names = coord_names[-n_dims:]
                    dims_list = zip(names, array.shape)
                    #print 'cube from array : shape=', array.shape
                    #print '  dimslist = ', dims_list
            self.shape = tuple([n_points for (name, n_points) in dims_list])
            self._coord_dims = {}
            self._coord_names = []
            self._coords = []
            for (name, n_points) in dims_list:
                self._coord_dims[name] = len(self._coord_names)
                self._coord_names.append(name)
                self._coords.append(DummyCoord(np.arange(n_points)))
            if array is None:
                stuff_array_with_indices(array)
            self.data = array

        def coord(self, name):
            return self._coords[self._coord_dims[name]]

        def coords(self, name):
            try_index = self._coord_dims.get(name)
            if try_index is None:
                return []
            return [self._coords[try_index]]
        
        def __getitem__(self, indexes):
#            print 'getitem : ',indexes
            # start by using numpy indexing on data
            array = self.data[indexes]
            result = DummyCube(array)
            # also fix resulting coordinates
            for (index,old_coord,coord_name) in zip(indexes, self._coords, self._coord_names):
#                print 'reindexing : ', coord_name
                new_coord_dim = result._coord_dims.get(coord_name, None)
                if new_coord_dim is not None:
                    new_coord = result._coords[new_coord_dim]
                    coord_points = old_coord.points[index]
                    new_coord.points = coord_points
                else:
#                    print ' ..not in new cube'
                    pass
            return result
        
        def __str__(self):
            return '<DummyCube: \n'+str(self.data)+'>'
            
    coord_dims = [_nz,3,4]
    coord_dims = dict(zip(coord_names, coord_dims))
    # note for these, coords must occur IN DIMENSION ORDER
    t2d = DummyCube([(name, coord_dims[name]) for name in ('bar', 'foo')])
    t3d = DummyCube([(name, coord_dims[name]) for name in ('baz', 'bar', 'foo')])

# point values and corresponding characters for each coord (by name)
coord_pts = {}
for (start_char, coord_name) in zip(coord_id_start_chars, coord_names):
    points = t3d.coord(coord_name).points
    ord_0 = ord(start_char)
    val2char_dict = {pt_val: chr(ord_0 + i_pt) for (i_pt, pt_val) in enumerate(points)}
    coord_pts[coord_name] = val2char_dict

def cube_notation_string(cube):
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
    """
    Construct a cube from a 'merge notation' string.
    
    Note: may specify out-of-order selection on a coordinate
    For example, 'ca312'.
    """
    coord_indices = {coord_name:[] for coord_name in coord_names}
    for this_char in cube_string:
        coord_name, = [name for name in coord_names
                            if this_char in coord_pts[name].values()]
        point_index = coord_pts[coord_name].values().index(this_char)
        coord_indices[coord_name].append(point_index)
    #print 'coord indices:', coord_indices
    if len(coord_indices['baz']) == 0:
        print '2d slice: ',coord_indices['bar'], coord_indices['foo']
        result_cube = t2d
        result_cube = result_cube[coord_indices['bar'], :]
        result_cube = result_cube[:, coord_indices['foo']]
    else:
        print '3d slice: ',coord_indices['baz'], coord_indices['bar'], coord_indices['foo']
        result_cube = t3d
        result_cube = result_cube[coord_indices['baz'], :, :]
        result_cube = result_cube[:, coord_indices['bar'], :]
        result_cube = result_cube[:, :, coord_indices['foo']]
    return result_cube

def test_cube_notation_string():
    tst_specs = [
        ('t2d', t2d),
        ('t3d', t3d),
        ('t3d[0,1:2,2:]', t3d[0,1:2,2:]),
        ('t3d[[2,0,1], 1:3, 3:]', t3d[[2,0,1], 1:3, 3:]),
        ('t3d[1:2, [3,0], 2:3]', t3d[1:2, [2,0], 2:3]),
    ]
    for (tst_str, tst_cube) in tst_specs:
        print 'test cube = ', tst_str
#        print tst_cube
        print '  --> ',cube_notation_string(tst_cube)


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

# early exit for now
#exit(0)

def test_cube_from_notation_string():
    tst_specs = [
        'ab12',
        'Pac21',
        'PQa12',
        'PQRSTabc1234',
        'abc1234',
        'Pb34',
        'RPQbc4',
        'Qca3',
    ]
    for tst_str in tst_specs:
        print tst_str
        tst_cube = cube_from_notation_string(tst_str)
        print tst_cube
        print '  (back-convert -> {})'.format(cube_notation_string(tst_cube))
        print

print
print '-----------------------------------------'
print 'TEST cube_FROM_notation_string...'
test_cube_from_notation_string()
print


def test_cube_merges():
    test_merge_specs = [
        ( ['ab1', 'ab2'], ['ab12']),
        ( ['a1', 'a3', 'a2'], ['a123']),
        ( ['a1', 'b3', 'a3', 'b1'], ['ab13']),
    ]
    for (in_speclist, out_speclist_expected) in test_merge_specs:
        print '  cubes merge input = ', ', '.join(in_speclist)
        in_cubelist = iris.cube.CubeList([
#            _reduce_cube(cube_from_notation_string(cube_string))
            cube_from_notation_string(cube_string)
            for cube_string in in_speclist])
        out_cubelist_actual = in_cubelist.merge()
        out_speclist_actual = [cube_notation_string(cube) 
                               for cube in out_cubelist_actual]
        print '          output = ', ', '.join(out_speclist_actual)
        if out_speclist_actual != out_speclist_expected:
            print ' !XXXX! expected = ', ', '.join(out_speclist_expected)
        print

print
print '-----------------------------------------'
print 'TEST cube merges...'
test_cube_merges()
