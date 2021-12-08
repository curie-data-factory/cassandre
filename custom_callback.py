########################################################## READ ME ##########################################################
#                                                                                                                           #
#                                            Custom Callbacks for keras models                                              #
#                                                                                                                           #                              
#############################################################################################################################


########################## Imports ##########################


import numpy as np
import keras
from utils import * 


########################## Callback class ##########################


class Batch_history(keras.callbacks.Callback):

	def __init__(self,metric,model_str):

		self.metric = metric
		self.model_str = model_str 

	def on_batch_end(self, batch, logs={}):
		"""We append loss and accuracy (it can be something else than acc, ie Dice coeff so)"""

		self.losses.append(logs.get('loss'))
		self.metrics.append(logs.get(self.metric))

	def on_train_begin(self, logs={}):
		"""We initialise an history dictionnary"""

		self.losses = []
		self.metrics = []

	def on_epoch_end(self, epoch, logs={}):
		"""We save & re-initialise an history dictionnary"""

		np.savez_compressed('../figure/data_batch_epoch_{0}_for_{1}'.format(epoch,self.model_str), a=self.losses, b=self.metrics )
		self.losses = []
		self.metrics = []