
import segmentation_models as sm
import keras


#TO-DO change these hardcoded & load from params
# Add documentation
class Model(object):
	def __init__(self):
		self.BACKBONE = 'mobilenetv2'
		BATCH_SIZE = 4
		self.CLASSES = ['person']
		self.h, self.w, self.c=224,224,3
		self.LR = 0.0001
		EPOCHS = 5

	def setup_model(self):
		# define network parameters
		n_classes = 1 if len(self.CLASSES) == 1 else (len(self.CLASSES) + 1)  # case for binary and multiclass segmentation
		activation = 'sigmoid' if n_classes == 1 else 'softmax'

		#create model
		model = sm.Unet(self.BACKBONE, classes=n_classes, activation=activation, input_shape=(self.h, self.w, self.c))

		#optimizer
		optim = keras.optimizers.Adam(self.LR)


		dice_loss = sm.losses.DiceLoss()
		focal_loss = sm.losses.BinaryFocalLoss() if n_classes == 1 else sm.losses.CategoricalFocalLoss()
		total_loss = dice_loss + (1 * focal_loss)

		metrics = [sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]

		# compile keras model with defined optimozer, loss and metrics
		model.compile(optim, total_loss, metrics)

		return model