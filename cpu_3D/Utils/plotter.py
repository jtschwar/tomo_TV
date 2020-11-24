from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg
import numpy as np

class plotter:

    def __init__(self, expType = None):
        self.app = QtGui.QApplication([])
        self.win = pg.GraphicsLayoutWidget(show=True)
        self.win.resize(800,600)

        self.reconPtr = self.win.addPlot(title='Reconstruction')
        self.reconPtr.hideAxis('left')
        self.reconPtr.hideAxis('bottom')
        
        self.ddPtr = self.win.addPlot(title='DD')
        self.ddCurve = self.ddPtr.plot(pen='b', name='DD')
        self.epsCurve = self.ddPtr.plot(pen='r', name='Eps')

        if expType == 'cs':
            self.win.nextRow()

        self.sinoPtr = self.win.addPlot(title='Sinogram')
        self.sinoPtr.hideAxis('left')
        self.sinoPtr.hideAxis('bottom')
        
        if expType == 'cs':
            self.tvPtr = self.win.addPlot(title='TV')
            self.tvPtr.setLabel('bottom', 'Iter')
            self.tvCurve = self.tvPtr.plot()

        # if expType == 'sim':
        #     self.rmsePtr = self.win.addPlot()
        #     self.rmsePtr.setLabel('left','RMSE')

    def dynamicCS_live_plot(self, tomo, logger, dd, eps, tv):

        if tomo.nproc() == 1:
            recSlice = tomo.getRecon(tomo.NsliceLoc()//2)
            sinoSlice = logger.log_projs[tomo.NsliceLoc()//2,:,:]            
        else:
            recSlice = tomo.getRecon(tomo.NsliceLoc()-1)
            sinoSlice = logger.log_projs[tomo.NsliceLoc(),:,:]

        # Show Reconstruction and Sinogram Slice
        self.iItem1 = pg.ImageItem(recSlice)
        self.reconPtr.addItem(self.iItem1)

        self.iItem2 = pg.ImageItem(sinoSlice)
        self.sinoPtr.addItem(self.iItem2)

        # Show Convergence of TV and DD
        iters = dd.shape[0]    
        self.ddCurve.setData(np.arange(iters), dd)
        self.epsCurve.setData(np.arange(iters), np.ones(iters)*eps)
        self.tvCurve.setData(np.arange(iters), tv)

        # if self.type == 'sim':
        #     self.rmsePtr.setData(iters, results['rmse'])

        self.app.processEvents()
        self.removeImageItems()

    def dynamicTomo_live_plot(self, tomo, logger, dd):

        if tomo.nproc() == 1:
            recSlice = tomo.getRecon(tomo.NsliceLoc()//2)
            sinoSlice = logger.log_projs[tomo.NsliceLoc()//2,:,:]            
        else:
            recSlice = tomo.getRecon(tomo.NsliceLoc()-1)
            sinoSlice = logger.log_projs[tomo.NsliceLoc(),:,:]

        # Show Reconstruction and Sinogram Slice
        self.iItem1 = pg.ImageItem(recSlice)
        self.reconPtr.addItem(self.iItem1)

        self.iItem2 = pg.ImageItem(sinoSlice)
        self.sinoPtr.addItem(self.iItem2)

        # Show Convergence of DD
        iters = dd.shape[0]    
        self.ddCurve.setData(np.arange(iters), dd)

        self.app.processEvents()
        self.removeImageItems()

    def removeImageItems(self):
        self.reconPtr.removeItem(self.iItem1)
        self.sinoPtr.removeItem(self.iItem2)
