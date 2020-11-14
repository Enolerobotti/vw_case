from python.vtk2numpy import Vtk2Numpy

class TestVtk2Numpy:
    # pvpython does not support unittest
    def test_vtk2numpy(self):
        filename = '../thin_plate_case/thin_plate_case.foam'
        vtk2numpy = Vtk2Numpy(filename)
        result = vtk2numpy(False)
        assert ((result['0.002']['U'] == result['0.002']['U']).all())
        assert not ((result['0.002']['U'] == result['0.004']['U']).all())
        print('ok')

if __name__ == '__main__':
    TestVtk2Numpy().test_vtk2numpy()