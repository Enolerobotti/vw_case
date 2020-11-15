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
        self.data = None

    def __call__(self, verbose=False, *args, **kwargs):
        fields = {}
        coordinates = {}
        for t in [0] + self.timesteps:
            self.update(time=t, verbose=verbose)
            dataset = self.vtk_wrapper()
            fields[str(t)] = {k: self.to_numpy(dataset, k) for k in dataset.PointData.keys()}
            coordinates[str(t)] = self.get_coordinates()
        return fields, coordinates

    def update(self, time, verbose):
        self.casefoam.SMProxy.UpdatePipeline(time)
        self.casefoam.UpdatePipelineInformation()
        self.data = Fetch(self.casefoam)
        if verbose:
            print(self.data)

    def get_time(self):
        return np.array(self.casefoam.TimestepValues)

    def get_coordinates(self):
        assert self.data.GetNumberOfBlocks() == 1, "Other is not a use case for current class"
        block = self.data.GetBlock(0)
        return np.array([block.GetPoint(point_idx) for point_idx in range(block.GetNumberOfPoints())])

    def vtk_wrapper(self):
        return dsa.WrapDataObject(self.data)

    @staticmethod
    def to_numpy(vtk_data, key:str):
        vtk_arr = vtk_data.PointData[key].GetArrays()
        assert len(vtk_arr) == 1, "Other unexpected"
        return vtk_to_numpy(vtk_arr[0])
