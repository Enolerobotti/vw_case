import os
import numpy as np
from paraview.simple import OpenFOAMReader
from paraview.servermanager import Fetch
from paraview.numpy_support import vtk_to_numpy
from vtk.numpy_interface import dataset_adapter as dsa

class Vtk2Numpy:
    def __init__(self, filename):
        filename = os.path.abspath(filename)
        self.casefoam = OpenFOAMReader(FileName=filename)
        self.timesteps = self.get_time()

    def __call__(self, verbose=False, *args, **kwargs):
        res = {}
        for t in self.timesteps:
            dataset = self.vtk_wrapper(time=t, verbose=verbose)
            res[str(t)] = {k: self.to_numpy(dataset, k) for k in dataset.PointData.keys()}
        return res

    def get_time(self):
        return np.array(self.casefoam.TimestepValues)

    def vtk_wrapper(self, time, verbose):
        self.casefoam.SMProxy.UpdatePipeline(time)
        self.casefoam.UpdatePipelineInformation()
        data = Fetch(self.casefoam)
        if verbose:
            print(data)
        return dsa.WrapDataObject(data)

    @staticmethod
    def to_numpy(vtk_data, key:str):
        vtk_arr = vtk_data.PointData[key].GetArrays()
        assert len(vtk_arr) == 1, "Other unexpected"
        return vtk_to_numpy(vtk_arr[0])



