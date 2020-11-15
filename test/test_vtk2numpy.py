from python.vtk2numpy import Vtk2Numpy
import joblib

class TestVtk2Numpy:
    # pvpython does not support unittest
    def test_vtk2numpy(self):
        filename = '../thin_plate_case/thin_plate_case.foam'
        vtk2numpy = Vtk2Numpy(filename)
        result, coord = vtk2numpy(False)
        assert ((result['0.002']['U'] == result['0.002']['U']).all())
        assert not ((result['0.002']['U'] == result['0.004']['U']).all())
        assert ((coord['0.002'] == coord['0.002']).all())
        assert not ((coord['0.002'] == coord['0.004']).all())
        # joblib.dump(coord, "../python/data/thin_p_coordinates.pkl")
        # joblib.dump(result, "../python/data/thin_p_fields.pkl")
        print('ok')

if __name__ == '__main__':
    TestVtk2Numpy().test_vtk2numpy()