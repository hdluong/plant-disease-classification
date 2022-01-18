from tensorflow.keras.preprocessing import image
#from tensorflow.keras.utils import layer_utils
#from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.utils import get_file, model_to_dot, plot_model
from nn.conv.simpleconvnet import SimpleNet

model = SimpleNet.create(256, 256, 3, 10)
model.summary()
# dot_img_file = 'model_simpleconvnet.png'
# plot_model(model, to_file=dot_img_file, show_shapes=True)
model.compile(optimizer = "Adam", loss = "categorical_crossentropy",  metrics = ["accuracy"])
#model.fit()
#model.evaluate()