import functools
import os.path
from typing import Optional

import pandas as pd
import vtk
import qt
import matplotlib
import qimage2ndarray
import matplotlib.pyplot as plt
import numpy as np
import slicer
import pyopenms as oms
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import parameterNodeWrapper

from slicer import vtkMRMLScalarVolumeNode


matplotlib.use("Agg")


# Module Declaration


class MassSpecViewModule(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("Mass Spectrometry Viewer")
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "iKnife")]
        self.parent.dependencies = []
        self.parent.contributors = ["Mohammad Farmhand (Med-i Lab)"]
        self.parent.helpText = _("Displays mass spectrometry data in 3D.")
        self.parent.acknowledgementText = _("This is just a toy module.")

        # Additional initialization step after application startup is complete
        # slicer.app.connect("startupCompleted()", registerSampleData)


# Module Parameters


@parameterNodeWrapper
class MassSpecViewModuleParameterNode:
    """
    The parameters needed by module.

    inputRAW - The volume to threshold.
    imageThreshold - The value at which to threshold the input volume.
    invertThreshold - If true, will invert the threshold.
    thresholdedVolume - The output volume that will contain the thresholded volume.
    invertedVolume - The output volume that will contain the inverted thresholded volume.
    """

    inputRAW: str = ""


# Module Widget


class MassSpecViewModuleWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/MassSpecViewModule.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = MassSpecViewModuleLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        self.ui.displayButton.connect("clicked(bool)", self.onBrowseClicked)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._canDisplay)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        # if not self._parameterNode.inputVolume:
        #     firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
        #     if firstVolumeNode:
        #         self._parameterNode.inputVolume = firstVolumeNode

    def setParameterNode(self, inputParameterNode: Optional[MassSpecViewModuleParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)

    def onBrowseClicked(self) -> None:
        """Run processing when user clicks "Apply" button."""
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            # Compute output
            self.logic.process(self.ui.inputRAW.currentPath)


# Module Logic


class MassSpecViewModuleLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    cylinderRadius = 100
    cylinderHeight = 500
    cylinderResolution = 16
    cylinderCenter = [0, 0, 0]

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)
        self.experiment = None
        self.peaks = None
        self.widget = None

    def getParameterNode(self):
        return MassSpecViewModuleParameterNode(super().getParameterNode())

    def process(self, inputFilePath: str) -> None:
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputFilePath: target mzML directory.
        """

        if not inputFilePath.endswith('.mzML') or not os.path.exists(inputFilePath):
            slicer.util.messageBox('Bad File')
            return

        # Drawing the Cylinder

        cylinder = vtk.vtkCylinderSource()
        cylinder.SetRadius(self.cylinderRadius)
        cylinder.SetHeight(self.cylinderHeight)
        cylinder.SetCenter(*self.cylinderCenter)
        cylinder.SetResolution(self.cylinderResolution)
        # Create a model node that displays output of the source
        modelNode = slicer.modules.models.logic().AddModel(cylinder.GetOutputPort())
        # Adjust display properties
        modelNode.GetDisplayNode().SetColor(1, 0, 0)
        modelNode.GetDisplayNode().SetOpacity(0.8)

        # Reading the RAW File

        self.experiment = oms.MSExperiment()
        oms.MzMLFile().load(inputFilePath, self.experiment)

        chrom, = self.experiment.getChromatograms()
        arr = chrom.get_peaks()
        arr = np.array(arr).T
        arr[:, 1] /= arr[:, 1].max()
        slicer.util.plot(arr, xColumnIndex=0, title='Chromatogram')

        self.peaks = pd.read_csv(inputFilePath.replace('.mzML', '.csv'))
        coords = self.peaks[['x', 'y', 'z']].values

        # Adding Markups to Scene

        for index, (x, y, z) in enumerate(coords):
            markupsNode = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLMarkupsFiducialNode')
            markupsNode.SetName(f'{index}')
            title = f'{self.peaks.iloc[index].label}: {self.peaks.iloc[index].desc}'
            markupsNode.AddControlPoint(x, y, z, title)
            func = functools.partial(self.plot_spectrometry_data, index=index)
            markupsNode.AddObserver(slicer.vtkMRMLMarkupsNode.PointStartInteractionEvent, func)

    def plot_spectrometry_data(self, caller, event, index):
        spectra = self.experiment.getSpectra()
        mzs, ints = spectra[index].get_peaks()
        ints[ints > 1e5] = 0
        arr = np.array((mzs, ints)).T
        slicer.util.plot(arr, xColumnIndex=0, title='Spectrum')


# Module Test


class MassSpecViewModuleTest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_MassSpecViewModule1()

    def test_MassSpecViewModule1(self):
        """Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("There is no test!")
