import tensorflow as tf


# 이미지 증강
gen_train = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1 / 255,
    horizontal_flip=True,
    rotation_range=10,
)

flow_train = gen_train.flow_from_directory(
    'data/train',
    target_size=[224, 224],
    batch_size=32,
    class_mode='sparse'
)

# 검사 데이터는 이미지 증강하지 않는다
gen_test = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1 / 255,
)

flow_test = gen_test.flow_from_directory(
    'data/valid',
    target_size=[224, 224],
    batch_size=32,
    class_mode='sparse'
)

# sparse
# [0, 5, 2, 3]
# categorical
# [1, 0, 0, 0, 0, 0]
# [0, 0, 0, 0, 0, 1]
# [0, 0, 2, 0, 0, 0]
# [0, 0, 0, 3, 0, 0]

# for x, y in flow_train:
#     print(x.shape)      # (32, 224, 224, 3)
#     print(y)            # [1. 1. 0. 1. 0. 0. 1. 2. 0. 1. 1. 2. 2. 1. 2. 2. 1. 1. 0. ...]
#     break

model = tf.keras.Sequential()
model.add(tf.keras.layers.Input(shape=flow_train.image_shape))
# 224 -> 112 -> 64 -> 32 -> 16 -> 8
# model.add(tf.keras.layers.Conv2D(64, 3, 1, 'same', activation='relu'))
# model.add(tf.keras.layers.Conv2D(64, 3, 1, 'same', activation='relu'))
# model.add(tf.keras.layers.MaxPool2D(2))
# model.add(tf.keras.layers.Conv2D(128, 3, 1, 'same', activation='relu'))
# model.add(tf.keras.layers.Conv2D(128, 3, 1, 'same', activation='relu'))
# model.add(tf.keras.layers.MaxPool2D(2))
# model.add(tf.keras.layers.Conv2D(256, 3, 1, 'same', activation='relu'))
# model.add(tf.keras.layers.Conv2D(256, 3, 1, 'same', activation='relu'))
# model.add(tf.keras.layers.Conv2D(256, 3, 1, 'same', activation='relu'))
# model.add(tf.keras.layers.MaxPool2D(2))
# model.add(tf.keras.layers.Conv2D(512, 3, 1, 'same', activation='relu'))
# model.add(tf.keras.layers.Conv2D(512, 3, 1, 'same', activation='relu'))
# model.add(tf.keras.layers.Conv2D(512, 3, 1, 'same', activation='relu'))
# model.add(tf.keras.layers.MaxPool2D(2))
# model.add(tf.keras.layers.Conv2D(512, 3, 1, 'same', activation='relu'))
# model.add(tf.keras.layers.Conv2D(512, 3, 1, 'same', activation='relu'))
# model.add(tf.keras.layers.Conv2D(512, 3, 1, 'same', activation='relu'))
# model.add(tf.keras.layers.MaxPool2D(2))
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(4096, activation='relu'))
# model.add(tf.keras.layers.Dense(4096, activation='relu'))
# model.add(tf.keras.layers.Dense(1000, activation='softmax'))

# 전이(transfer)/사전(pre-trained) 학습
# vgg16 = tf.keras.applications.VGG16(include_top=False)
# vgg16.trainable = False
# model.add(vgg16)
# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(1024, activation='relu'))
# model.add(tf.keras.layers.Dense(256, activation='relu'))
# model.add(tf.keras.layers.Dense(3, activation='softmax'))

# 텐서플로 허브(tensorflow-hub)
import tensorflow_hub as hub
vgg16 = hub.KerasLayer("https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4", trainable=False)
model.add(vgg16)
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Dense(256, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              metrics=['acc'])

model.fit(flow_train, epochs=5, validation_data=flow_test)











