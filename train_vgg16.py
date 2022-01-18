from nn.conv.vgg16 import VGG16_Net
from tensorflow.keras.utils import get_file, model_to_dot, plot_model

model, base_model = VGG16_Net.create(256, 256, 3, 10)
dot_img_file = 'model_vgg16_.png'
plot_model(model, to_file=dot_img_file, show_shapes=True)
model.summary()

# for i, layer in enumerate(base_model.layers):
#    print(i, layer.name)