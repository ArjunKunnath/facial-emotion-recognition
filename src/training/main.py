import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, SeparableConv2D, BatchNormalization, ReLU, Add
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Enable mixed precision training for faster computation
tf.keras.mixed_precision.set_global_policy('mixed_float16')

def residual_block(x, filters, kernel_size=3, stride=1):
    shortcut = x

    # Simplified the residual block with fewer operations
    x = SeparableConv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU(max_value=6)(x)

    x = SeparableConv2D(filters, kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization()(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = SeparableConv2D(filters, kernel_size=1, strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = ReLU(max_value=6)(x)
    return x

def build_efficient_resnet(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # Reduced initial filter size for faster computation
    x = SeparableConv2D(32, kernel_size=5, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU(max_value=6)(x)
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    # First residual block with 64 filters
    x = residual_block(x, filters=64, stride=1)
    
    # Second residual block with 64 filters (no downsampling)
    x = residual_block(x, filters=64, stride=1)
    
    # Third residual block with 64 filters (no downsampling)
    x = residual_block(x, filters=64, stride=1)

    # First Pyramid Pooling Block
    pool_features1 = []
    pool_sizes = [(1, 1), (2, 2), (4, 4)]
    for pool_size in pool_sizes:
        pooled = GlobalAveragePooling2D()(x)
        pooled = Dense(128)(pooled)
        pooled = BatchNormalization()(pooled)
        pooled = ReLU()(pooled)
        pooled = tf.keras.layers.Reshape((1, 1, 128))(pooled)
        pooled = tf.keras.layers.UpSampling2D(size=x.shape[1:3])(pooled)
        pool_features1.append(pooled)
    x = tf.keras.layers.Concatenate(axis=-1)([x] + pool_features1)

    # Fourth residual block with 128 filters and downsampling
    x = residual_block(x, filters=128, stride=2)
    
    # Fifth residual block with 128 filters (no downsampling)
    x = residual_block(x, filters=128, stride=1)
    
    # Sixth residual block with 128 filters (no downsampling)
    x = residual_block(x, filters=128, stride=1)

    # Second Pyramid Pooling Block
    pool_features2 = []
    for pool_size in pool_sizes:
        pooled = GlobalAveragePooling2D()(x)
        pooled = Dense(256)(pooled)
        pooled = BatchNormalization()(pooled)
        pooled = ReLU()(pooled)
        pooled = tf.keras.layers.Reshape((1, 1, 256))(pooled)
        pooled = tf.keras.layers.UpSampling2D(size=x.shape[1:3])(pooled)
        pool_features2.append(pooled)
    x = tf.keras.layers.Concatenate(axis=-1)([x] + pool_features2)

    # Using Flatten instead of GlobalAveragePooling2D
    x = Flatten()(x)

    # Adding multiple dense layers with decreasing units
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)

    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    outputs = Dense(num_classes, activation='softmax', dtype='float32')(x)

    model = Model(inputs, outputs)
    return model  # Removed compilation from here

# Data paths
train_data_dir = '/kaggle/input/fer2013/train'
validation_data_dir = '/kaggle/input/fer2013/test'  # Fixed the validation path

# Optimized data generators with caching
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,  # Reduced for faster processing
    shear_range=0.2,    # Reduced for faster processing
    zoom_range=0.2,     # Reduced for faster processing
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)  # Simpler validation generator

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=64,  # Increased batch size for faster training
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=64,  # Increased batch size
    class_mode='categorical'
)

# After your validation_generator definition (around line 100)
import numpy as np

def extract_from_generator(generator):
    X, y = [], []

    # Reset the generator to ensure we start from the beginning
    generator.reset()

    # Calculate how many batches we need to process
    steps = len(generator)

    for i in range(steps):
        batch_x, batch_y = next(generator)
        X.extend(batch_x)
        y.extend(batch_y)

    return np.array(X), np.array(y)

# Extract data from the generators
print("Extracting training data...")
X_train, y_train = extract_from_generator(train_generator)
print("Extracting validation data...")
X_test, y_test = extract_from_generator(validation_generator)

print(f"Training data shape: {X_train.shape}, {y_train.shape}")
print(f"Validation data shape: {X_test.shape}, {y_test.shape}")

# No need to rescale again as the generators already applied rescaling (1./255)

# Build the optimized model
model = build_efficient_resnet(input_shape=(48, 48, 1), num_classes=7)
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
print(model.summary())

# Add this line to see total parameters in millions
print(f"Total parameters: {model.count_params() / 1000000:.2f}M")

# Train model
# Callbacks for better training efficiency
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,  # Reduced patience
    min_lr=1e-6,
    verbose=1
)


# Calculate validation steps to ensure all validation data is used
validation_steps = len(validation_generator)

# Train with fewer epochs but with early stopping
epochs = 50
history = model.fit(
    X_train, y_train,
    batch_size=64,
    epochs=epochs,
    validation_data=(X_test, y_test),
    callbacks=[reduce_lr]
)

# Save the model
model.save('optimized_model_4r2p.h5')