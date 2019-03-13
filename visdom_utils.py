import visdom
import skimage.measure as sk
import numpy as np

# maintain a gloable handle for it
class VisdomInterface(object):
    
    def __init__(self, port=8097):
        self.port = port
        self.vis = None

        self.recons_loss_window = None
        self.transreg_loss_window = None

        self.n_samples_to_visualize = 8
        self.sample_windows = [None] * self.n_samples_to_visualize

        self.init_visdom()
    
    def init_visdom(self):
        self.vis = visdom.Visdom(port=self.port)
        return

    def update_losses(self, recons_loss, transreg_loss):
        if self.recons_loss_window is None:
            self.recons_loss_window = self.vis.line(Y=recons_loss)
        else:
            self.vis.line(X=np.arange(len(recons_loss))+1, Y=recons_loss, win=self.recons_loss_window, update='replace')
        if self.transreg_loss_window is None:
            self.transreg_loss_window = self.vis.line(Y=transreg_loss)
        else:
            self.vis.line(X=np.arange(len(transreg_loss))+1, Y=transreg_loss, win=self.transreg_loss_window, update='replace')
        return

