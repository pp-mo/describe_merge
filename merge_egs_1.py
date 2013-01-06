'''
Created on Jan 4, 2013

@author: itpp
'''
do_real = False
if do_real:
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

if not do_real:
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
				self._coords.append(DummyCoord(range(n_points)))
			if array is None:
				array = np.zeros(self.shape)
				array.flat = np.arange(array.size)
			self.data = array
			
		def coord(self, name):
			return self._coords[self._coord_dims[name]]

		def coords(self, name):
			try_index = self._coord_dims.get(name)
			if try_index is None:
				return []
			return [self._coords[try_index]]
		
		def __getitem__(self, indexes):
			#print 'getitem : ',indexes
			array = self.data[indexes]
			#print '  array = ', self.data
			#print '  subarray = ', array
			return DummyCube(array)
		
		def __str__(self):
			return '<DummyCube: '+str(self.data)+'>'
			
	nz = 5
	coord_dims = [3,4,nz]
	coord_dims = dict(zip(coord_names, coord_dims))
#	print 'coord dims = ', coord_dims
	# note for these, coords must occur IN DIMENSION ORDER
	t2d = DummyCube([(name, coord_dims[name]) for name in ('bar', 'foo')])
	t3d = DummyCube([(name, coord_dims[name]) for name in ('baz', 'bar', 'foo')])
#	print 'dummy t2d = ', t2d
#	print 'dummy t2d[2:,2:] = ',t2d[2:,2:]
#	print 'dummy t3d = ', t3d



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
    #print 'coord indices:', coord_indices
    if len(coord_indices['baz']) == 0:
        #print '2d slice: ',coord_indices['bar'], coord_indices['foo']
        #result_cube = t2d[coord_indices['bar'], coord_indices['foo']]
        result_cube = t2d[coord_indices['bar'], :]
        result_cube = result_cube[:, coord_indices['foo']]
    else:
        print '3d slice: ',coord_indices['bar'], coord_indices['foo']
        result_cube = t3d[coord_indices['baz'], coord_indices['bar'], coord_indices['foo']]
        result_cube = t2d[coord_indices['baz'], :, :]
        result_cube = result_cube[:, coord_indices['bar'], :]
        result_cube = result_cube[:, :, coord_indices['foo']]
    return result_cube

def test_cube_2_notation_string():
	tst_specs = [
		('t2d',t2d),
		('t3d',t3d),
		('t3d[0,1:2,2:]',t3d[0,1:2,2:])
	]
	for (tst_str, tst_cube) in tst_specs:
		print 'test cube = ', tst_str
		print tst_cube
		print '  --> ',cube_2_notation_string(tst_cube)

print
print 'TEST cube_2_notation_string...'
test_cube_2_notation_string()
print

def test_cube_from_notation_string():
	tst_specs = [
		'ab12',
		'Pac21',
		'PQa12',
	]
	for tst_str in tst_specs:
		print tst_str
		t = cube_from_notation_string(tst_str)
		print t

print
print 'TEST cube_FROM_notation_string...'
test_cube_from_notation_string()
print

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
