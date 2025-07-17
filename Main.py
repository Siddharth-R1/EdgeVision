import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tensorflow as tf
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.metrics import Precision, Recall
import logging
import cv2
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# Configuration
DAGM_BASE_PATH = "/Users/siddharth/Documents/EdgeVision/Main/datasets/DAGMdatasetzip/DAGM_KaggleUpload"
MVTEC_BASE_PATH = "/Users/siddharth/Documents/EdgeVision/Main/datasets/MVTecDatasetZip"
DAGM_CLASSES = [f"Class{i}" for i in range(1, 7)]
BATCH_SIZE = 32
TARGET_SIZE = (224, 224)

# Global metric instances
precision_metric = Precision(name='precision')
recall_metric = Recall(name='recall')

# Dataset loading functions
def load_dagm_metadata(class_path, split="Train"):
    label_path = os.path.join(class_path, split, "Label", "Labels.txt")
    metadata = []
    try:
        with open(label_path, 'r') as f:
            lines = f.readlines()[1:]  # Skip header
            for line in lines:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                parts = line.split('\t')  # Use tab delimiter
                if len(parts) < 3:
                    logger.warning(f"Skipping invalid line in {label_path}: {line}")
                    continue
                img_path = os.path.join(class_path, split, parts[2])
                try:
                    label = int(parts[1])
                except ValueError:
                    logger.warning(f"Invalid label in {label_path}: {line}")
                    continue
                metadata.append({
                    'image_path': img_path,
                    'label': label,
                    'dataset': 'DAGM'
                })
    except FileNotFoundError:
        logger.warning(f"DAGM metadata not found in {label_path}")
    return metadata

def load_mvtec_metadata(category_path, split="train"):
    metadata = []
    base_split_path = os.path.join(category_path, split)
    if not os.path.exists(base_split_path):
        logger.warning(f"MVTec split {split} not found in {category_path}")
        return metadata
    for defect_type in os.listdir(base_split_path):
        defect_path = os.path.join(base_split_path, defect_type)
        if not os.path.isdir(defect_path):
            continue
        for img_file in os.listdir(defect_path):
            if not img_file.endswith('.png'):
                continue
            img_path = os.path.join(defect_path, img_file)
            label = 0 if defect_type == 'good' else 1
            metadata.append({
                'image_path': img_path,
                'label': label,
                'dataset': 'MVTec'
            })
    return metadata

def load_and_preprocess_images(df, max_samples=None):
    images = []
    labels = []
    logger.info("Preloading and preprocessing images...")
    if max_samples:
        sample_size = min(max_samples, len(df))
        logger.info(f"Requested {max_samples} samples, using {sample_size} samples from {len(df)} available")
        sample_df = df.sample(n=sample_size, random_state=42)
    else:
        sample_df = df
    for idx, row in sample_df.iterrows():
        try:
            img = load_img(row['image_path'], target_size=TARGET_SIZE)
            img_array = img_to_array(img)
            img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
            images.append(img_array)
            labels.append(int(row['label']))
            if idx % 1000 == 0:
                logger.info(f"Processed {idx} images")
        except Exception as e:
            logger.warning(f"Failed to load image {row['image_path']}: {e}")
            continue
    return np.array(images), np.array(labels), sample_df

def load_combined_dataset():
    dagm_train, dagm_test, mvtec_train, mvtec_test = [], [], [], []
    for dagm_class in DAGM_CLASSES:
        class_path = os.path.join(DAGM_BASE_PATH, dagm_class)
        dagm_train.extend(load_dagm_metadata(class_path, 'Train'))
        dagm_test.extend(load_dagm_metadata(class_path, 'Test'))
    if os.path.exists(MVTEC_BASE_PATH):
        for category in os.listdir(MVTEC_BASE_PATH):
            category_path = os.path.join(MVTEC_BASE_PATH, category)
            if os.path.isdir(category_path):
                mvtec_train.extend(load_mvtec_metadata(category_path, 'train'))
                mvtec_test.extend(load_mvtec_metadata(category_path, 'test'))
    mvtec_test_train, mvtec_test_val = train_test_split(
        mvtec_test,
        test_size=0.2,
        random_state=42,
        stratify=[x['label'] for x in mvtec_test]
    )
    train_data = dagm_train + mvtec_train + mvtec_test_train
    val_data = dagm_test + mvtec_test_val
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)
    train_df['label'] = train_df['label'].astype(str)
    val_df['label'] = val_df['label'].astype(str)
    logger.info(f"Training data breakdown: Good={len(train_df[train_df['label'] == '0'])}, Defective={len(train_df[train_df['label'] == '1'])}")
    logger.info(f"Validation data breakdown: Good={len(val_df[val_df['label'] == '0'])}, Defective={len(val_df[val_df['label'] == '1'])}")
    logger.info(f"Validation DataFrame size: {len(val_df)}")
    return train_df, val_df

# Load dataset
train_df, val_df = load_combined_dataset()
logger.info(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")

# Preload images
train_images, train_labels, train_df_sampled = load_and_preprocess_images(train_df, max_samples=5000)
val_images, val_labels, val_df_sampled = load_and_preprocess_images(val_df, max_samples=2000)

# Aggressive oversampling (3x Defective samples with augmentation)
def oversample_defective(images, labels):
    defective_idx = np.where(labels == 1)[0]
    extra_images = []
    extra_labels = []
    for idx in defective_idx:
        img = images[idx]
        extra_images.extend([img, np.flip(img, axis=1), np.flip(img, axis=0)])
        extra_labels.extend([1, 1, 1])
    if extra_images:
        images = np.concatenate([images, extra_images], axis=0)
        labels = np.concatenate([labels, extra_labels], axis=0)
    logger.info(f"Oversampled training data: Good={np.sum(labels == 0)}, Defective={np.sum(labels == 1)}")
    return images, labels

train_images, train_labels = oversample_defective(train_images, train_labels)

# Focal loss with higher alpha_defective
def focal_loss(y_true, y_pred, gamma=2.0, alpha_defective=0.95):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    focal_factor = tf.pow(1 - p_t, gamma)
    alpha = y_true * alpha_defective + (1 - y_true) * (1 - alpha_defective)
    return tf.reduce_mean(focal_factor * alpha * bce)

# F1-score metric
def f1_score(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    precision_metric.reset_state()
    recall_metric.reset_state()
    precision_metric.update_state(y_true, y_pred)
    recall_metric.update_state(y_true, y_pred)
    p = precision_metric.result()
    r = recall_metric.result()
    return 2 * (p * r) / (p + r + tf.keras.backend.epsilon())

# LR schedule
def cosine_decay_with_warmup(epoch, initial_lr, total_epochs, warmup_epochs=2):
    if epoch < warmup_epochs:
        lr = initial_lr * (epoch + 1) / warmup_epochs
    else:
        decay = 0.5 * (1 + tf.math.cos(np.pi * (epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
        lr = initial_lr * decay
    return float(lr)

# Build MobileNetV2 with enhanced augmentation and regularization
def build_mobilenetv2(input_shape=(224, 224, 3), weights='imagenet'):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.RandomFlip("horizontal_and_vertical")(inputs)
    x = tf.keras.layers.RandomRotation(0.2)(x)
    x = tf.keras.layers.RandomZoom(0.2)(x)
    x = tf.keras.layers.RandomContrast(0.2)(x)
    base_model = MobileNetV2(
        include_top=False,
        weights=weights,
        input_tensor=x,
        pooling=None
    )
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=predictions)
    return model

# Load and configure
base_model = build_mobilenetv2()
for layer in base_model.layers[:-30]:
    layer.trainable = False
for layer in base_model.layers[-30:]:
    layer.trainable = True

# Compile with quantization-aware training
base_model = tf.keras.models.clone_model(
    base_model,
    clone_function=lambda layer: layer if not isinstance(layer, tf.keras.layers.Dropout) else tf.keras.layers.Dropout(layer.rate, noise_shape=layer.noise_shape)
)
base_model = tf.keras.models.Model(inputs=base_model.inputs, outputs=base_model.outputs)
base_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=focal_loss,
    metrics=['accuracy', precision_metric, recall_metric, f1_score]
)

# Class weights
class_weights = {0: 1.0, 1: 5.0}

# Train
steps_per_epoch = len(train_images) // BATCH_SIZE
val_steps = len(val_images) // BATCH_SIZE

history = base_model.fit(
    x=train_images,
    y=train_labels,
    batch_size=BATCH_SIZE,
    epochs=30,
    steps_per_epoch=steps_per_epoch,
    validation_data=(val_images, val_labels),
    validation_steps=val_steps,
    class_weight=class_weights,
    callbacks=[
        EarlyStopping(monitor='val_f1_score', mode='max', patience=10, restore_best_weights=True),
        ModelCheckpoint('mobilenetv2_tuned_best.keras', monitor='val_f1_score', mode='max', save_best_only=True),
        LearningRateScheduler(lambda epoch: cosine_decay_with_warmup(epoch, 1e-3, 30))
    ],
    verbose=1
)

# Fine-tune
for layer in base_model.layers[-50:]:
    layer.trainable = True

fine_tune_lr = 1e-4
optimizer = tf.keras.optimizers.Adam(learning_rate=fine_tune_lr)
base_model.compile(
    optimizer=optimizer,
    loss=focal_loss,
    metrics=['accuracy', precision_metric, recall_metric, f1_score]
)

history_fine = base_model.fit(
    x=train_images,
    y=train_labels,
    batch_size=BATCH_SIZE,
    epochs=15,
    steps_per_epoch=steps_per_epoch,
    validation_data=(val_images, val_labels),
    validation_steps=val_steps,
    class_weight=class_weights,
    callbacks=[
        EarlyStopping(monitor='val_f1_score', mode='max', patience=10, restore_best_weights=True),
        ModelCheckpoint('mobilenetv2_tuned_fine_best.keras', monitor='val_f1_score', mode='max', save_best_only=True),
        LearningRateScheduler(lambda epoch: cosine_decay_with_warmup(epoch, fine_tune_lr, 15))
    ],
    verbose=1
)

# Save final model
base_model.save('mobilenetv2_tuned_final.keras')

# Evaluate with threshold tuning
logger.info("Evaluating model with different thresholds...")
predictions = base_model.predict(val_images, batch_size=BATCH_SIZE, verbose=1)
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

for thresh in thresholds:
    predicted_labels = (predictions > thresh).astype(int).flatten()
    true_positives = np.sum((predicted_labels == 1) & (val_labels == 1))
    false_positives = np.sum((predicted_labels == 1) & (val_labels == 0))
    false_negatives = np.sum((predicted_labels == 0) & (val_labels == 1))
    true_negatives = np.sum((predicted_labels == 0) & (val_labels == 0))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (true_positives + true_negatives) / len(val_labels)
    
    logger.info(f"Threshold: {thresh}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1-Score: {f1:.4f}")
    logger.info(f"True Positives: {true_positives}")
    logger.info(f"False Positives: {false_positives}")
    logger.info(f"False Negatives: {false_negatives}")
    logger.info(f"True Negatives: {true_negatives}")
    logger.info("---")

# Improved defect detection with saliency maps and fallback
def compute_saliency_map(model, image):
    image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
    image_tensor = tf.expand_dims(image_tensor, axis=0)
    
    with tf.GradientTape() as tape:
        tape.watch(image_tensor)
        prediction = model(image_tensor, training=False)
        loss = prediction[0]
    
    gradient = tape.gradient(loss, image_tensor)
    gradient = tf.abs(gradient)
    saliency = tf.reduce_max(gradient, axis=-1)[0]
    saliency = (saliency - tf.reduce_min(saliency)) / (tf.reduce_max(saliency) - tf.reduce_min(saliency) + 1e-8)
    
    logger.debug(f"Saliency map stats: min={saliency.numpy().min():.4f}, max={saliency.numpy().max():.4f}, mean={saliency.numpy().mean():.4f}")
    return saliency.numpy()

def fallback_defect_detection(image):
    img_array = (image * 255).astype(np.uint8)
    if img_array.shape[-1] == 3:
        img_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_array
    blurred = cv2.GaussianBlur(img_gray, (21, 21), 0)
    diff = cv2.absdiff(img_gray, blurred)
    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 50:  # Reduced threshold
            x, y, w, h = cv2.boundingRect(largest_contour)
            logger.debug(f"Fallback detection: x={x}, y={y}, w={w}, h={h}")
            return (x, y, x + w, y + h)
    return None

def detect_defect_region(image, model):
    img_array = (image * 255).astype(np.uint8)
    if img_array.shape[-1] != 3:
        img_array = np.stack([img_array] * 3, axis=-1)
    
    # Compute saliency map
    saliency = compute_saliency_map(model, image)
    
    # Save saliency map for debugging
    os.makedirs("saliency_maps", exist_ok=True)
    saliency_path = f"saliency_maps/saliency_{np.random.randint(10000)}.png"
    plt.imsave(saliency_path, saliency, cmap='jet')
    logger.info(f"Saved saliency map to {saliency_path}")
    
    # Use Otsu's method for thresholding
    saliency_uint8 = (saliency * 255).astype(np.uint8)
    _, thresh = cv2.threshold(saliency_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        logger.warning("No defect regions detected in saliency map, falling back to intensity-based detection")
        return fallback_defect_detection(image)
    
    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest_contour) < 50:  # Reduced threshold
        logger.warning("Defect region too small, falling back to intensity-based detection")
        return fallback_defect_detection(image)
    
    x, y, w, h = cv2.boundingRect(largest_contour)
    logger.debug(f"Detected defect region: x={x}, y={y}, w={w}, h={h}")
    return (x, y, x + w, y + h)

def visualize_defective_images(val_df, val_images, val_labels, predictions, model, dataset_name, threshold=0.3, max_images=20):
    dataset_df = val_df[val_df['dataset'] == dataset_name]
    dataset_indices = dataset_df.index
    defective_indices = [i for i in range(len(val_labels)) if i in dataset_indices and val_labels[i] == 1 and predictions[i] > threshold]
    
    if len(defective_indices) == 0:
        logger.warning(f"No defective images found for {dataset_name} with predictions above threshold {threshold}.")
        return
    
    num_images = min(len(defective_indices), max_images)
    if num_images < max_images:
        logger.warning(f"Only {num_images} defective images available for {dataset_name}, requested {max_images}.")
    
    logger.info(f"Visualizing {num_images} defective images from {dataset_name}...")
    
    os.makedirs(f"defective_images/{dataset_name.lower()}", exist_ok=True)
    
    for i, idx in enumerate(defective_indices[:max_images]):
        img_path = val_df.iloc[idx]['image_path']
        img = load_img(img_path, target_size=TARGET_SIZE)
        img_array = img_to_array(img)
        img_normalized = tf.keras.applications.mobilenet_v2.preprocess_input(img_array.copy())
        img_display = img_array.astype(np.uint8)
        
        # Detect defect region using saliency
        bbox = detect_defect_region(img_normalized, model)
        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(img_display, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"Defect (Conf: {predictions[idx][0]:.2f})"
            cv2.putText(img_display, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        else:
            logger.warning(f"Could not detect defect region for image {img_path}")
        
        plt.figure(figsize=(5, 5))
        plt.imshow(img_display)
        plt.title(f"{dataset_name} Defective Image {i+1} (Pred: {predictions[idx][0]:.4f})")
        plt.axis('off')
        plt.show()
        
        save_path = f"defective_images/{dataset_name.lower()}/defective_{i+1}.png"
        cv2.imwrite(save_path, cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR))
        logger.info(f"Saved defective image to {save_path}")

# Run visualization for both datasets
logger.info("Visualizing defective images...")
visualize_defective_images(val_df_sampled, val_images, val_labels, predictions, base_model, 'DAGM', threshold=0.3)
visualize_defective_images(val_df_sampled, val_images, val_labels, predictions, base_model, 'MVTec', threshold=0.3)

# Convert to TFLite with quantization-aware training
def representative_dataset():
    for i in range(min(50, len(train_images) // BATCH_SIZE)):
        batch_x = train_images[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
        for img in batch_x:
            yield [np.expand_dims(img, axis=0).astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_keras_model(base_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
converter.representative_dataset = representative_dataset
tflite_model = converter.convert()

with open('mobilenetv2_tuned_optimized.tflite', 'wb') as f:
    f.write(tflite_model)
logger.info("Model trained and converted to TFLite for Raspberry Pi.")