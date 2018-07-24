from keras.callbacks import Callback

class Logger(Callback):
	def __init__(self, fname):
		self.fname = fname
		with open(fname, 'w') as f:
			f.write('epoch,loss,val_loss,mean_iou,mean_val_iou\n')

	def on_epoch_end(self, epoch, logs={}):
		curr_loss = logs['loss']
		curr_val_loss = logs['val_loss']
		curr_iou = logs['mean_iou']
		curr_val_iou = logs['val_mean_iou']
		metrics = [curr_loss, curr_val_loss, curr_iou, curr_val_iou]

		with open(self.fname, 'a') as f:
			f.write(','.join([str(m) for m in metrics]) + '\n')