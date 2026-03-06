import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import os

# ---------------- Paths ---------------- #
train_dir = "dataset"
img_size = (96, 96)     # smaller image size = faster training
batch_size = 32
epochs = 2              # super fast

# ---------------- Data Generators ---------------- #
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=img_size, batch_size=batch_size,
    class_mode="categorical", subset="training", shuffle=True
)

val_gen = train_datagen.flow_from_directory(
    train_dir, target_size=img_size, batch_size=batch_size,
    class_mode="categorical", subset="validation", shuffle=False
)

# ---------------- Base Model ---------------- #
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(img_size[0], img_size[1], 3),
    include_top=False,
    weights="imagenet"
)

# ---------------- Build Model ---------------- #
input_layer = Input(shape=(img_size[0], img_size[1], 3))
x = base_model(input_layer, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
output = Dense(train_gen.num_classes, activation="softmax")(x)

model = Model(inputs=input_layer, outputs=output)

# ---------------- Train Top Layers Only ---------------- #
base_model.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

callbacks = [EarlyStopping(monitor="val_loss", patience=1, restore_best_weights=True)]

print("🔹 Training top layers (FAST MODE)...")
model.fit(train_gen, validation_data=val_gen,
          epochs=epochs, verbose=1, callbacks=callbacks)

# ---------------- Save Model ---------------- #
os.makedirs("model", exist_ok=True)
model.save("model/brain_tumor_model.h5")
np.save("model/class_labels.npy", np.array(list(train_gen.class_indices.keys())))

print("✅ Model trained & saved successfully (FAST MODE).")
print("Classes:", list(train_gen.class_indices.keys()))
