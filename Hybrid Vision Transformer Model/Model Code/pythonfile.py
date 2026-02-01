!pip install -U tensorflow==2.19.0

!pip install -q scikit-learn matplotlib seaborn opencv-python-headless Pillow scipy networkx

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")
print("✓ Mixed precision enabled")

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
import cv2
from google.colab import drive
import networkx as nx
from scipy.spatial.distance import cdist
from keras.saving import register_keras_serializable

np.random.seed(42)
tf.random.set_seed(42)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ GPU available: {len(gpus)} device(s)")
    except RuntimeError as e:
        print(e)
else:
    print("⚠ No GPU detected. Training will be slower.")

print(f"TensorFlow version: {tf.__version__}")

drive.mount('/content/drive')

BASE_PATH = '/content/drive/MyDrive/NHL_DATA/Multi Cancer/Multi Cancer/Lymphoma'

if os.path.exists(BASE_PATH):
    print(f"✓ Data directory found: {BASE_PATH}")
    subdirs = [d for d in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, d))]
    print(f"  Subdirectories: {subdirs}")
else:
    print(f"✗ Data directory not found: {BASE_PATH}")
    print("  Please update BASE_PATH to match your data location")

MODEL_DIR = "/content/drive/MyDrive/NHL_Project/models"
!mkdir -p $MODEL_DIR

CONFIG = {
    'IMG_SIZE': (128, 128),
    'BATCH_SIZE': 32,
    'EPOCHS': 10,
    'LEARNING_RATE': 1e-4,
    'TRAIN_SPLIT': 0.70,
    'VAL_SPLIT': 0.15,
    'TEST_SPLIT': 0.15,
    'NUM_CLASSES': 3,
    'CLASS_NAMES': ['CLL', 'FL', 'MCL'],
    'CLASS_LABELS': {
        'lymph_cll': 'Chronic Lymphocytic Leukemia (CLL)',
        'lymph_fl': 'Follicular Lymphoma (FL)',
        'lymph_mcl': 'Mantle Cell Lymphoma (MCL)'
    },
    'RANDOM_SEED': 42
}

print("Configuration:")
for key, value in CONFIG.items():
    print(f"  {key}: {value}")

def create_train_val_test_split(source_dir, dest_dir, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    np.random.seed(seed)

    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(dest_dir, split), exist_ok=True)

    class_dirs = [d for d in os.listdir(source_dir)
                  if os.path.isdir(os.path.join(source_dir, d))]

    split_info = {}

    for class_dir in class_dirs:
        class_path = os.path.join(source_dir, class_dir)
        all_files = [f for f in os.listdir(class_path)
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

        np.random.shuffle(all_files)

        n_total = len(all_files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)

        train_files = all_files[:n_train]
        val_files = all_files[n_train:n_train + n_val]
        test_files = all_files[n_train + n_val:]

        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(dest_dir, split, class_dir), exist_ok=True)

        for fname in train_files:
            src = os.path.join(class_path, fname)
            dst = os.path.join(dest_dir, 'train', class_dir, fname)
            shutil.copy2(src, dst)

        for fname in val_files:
            src = os.path.join(class_path, fname)
            dst = os.path.join(dest_dir, 'val', class_dir, fname)
            shutil.copy2(src, dst)

        for fname in test_files:
            src = os.path.join(class_path, fname)
            dst = os.path.join(dest_dir, 'test', class_dir, fname)
            shutil.copy2(src, dst)

        split_info[class_dir] = {
            'total': n_total,
            'train': len(train_files),
            'val': len(val_files),
            'test': len(test_files)
        }

    return split_info

SPLIT_DIR = '/content/drive/MyDrive/NHL_Project/data/lymphoma_split'

print("Creating train/val/test split...")
print("This ensures NO data leakage between sets.\n")

split_info = create_train_val_test_split(
    source_dir=BASE_PATH,
    dest_dir=SPLIT_DIR,
    train_ratio=CONFIG['TRAIN_SPLIT'],
    val_ratio=CONFIG['VAL_SPLIT'],
    test_ratio=CONFIG['TEST_SPLIT'],
    seed=CONFIG['RANDOM_SEED']
)

print("\n" + "="*70)
print("DATA SPLIT STATISTICS (NO LEAKAGE)")
print("="*70)
for class_name, counts in split_info.items():
    print(f"\n{CONFIG['CLASS_LABELS'].get(class_name, class_name)}:")
    print(f"  Total: {counts['total']} images")
    print(f"  Train: {counts['train']} ({counts['train']/counts['total']*100:.1f}%)")
    print(f"  Val:   {counts['val']} ({counts['val']/counts['total']*100:.1f}%)")
    print(f"  Test:  {counts['test']} ({counts['test']/counts['total']*100:.1f}%)")
print("="*70)

print("\n✅ Split complete! Train/Val/Test are now completely independent.")
print("   Test set is held-out and will ONLY be used for final evaluation.")

plt.figure(figsize=(8,5))
sns.barplot(x=list(split_info.keys()), y=[split_info[c]['total'] for c in split_info])
plt.title("Class Distribution (Total Images per Class)", fontsize=14, fontweight='bold')
plt.ylabel("Number of Images")
plt.xlabel("NHL Subtype")
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(15,5))
for i, class_dir in enumerate(split_info.keys()):
    sample_img_path = os.path.join(BASE_PATH, class_dir, os.listdir(os.path.join(BASE_PATH, class_dir))[0])
    img = cv2.imread(sample_img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    axes[i].imshow(img)
    axes[i].set_title(CONFIG['CLASS_LABELS'][class_dir])
    axes[i].axis('off')
plt.suptitle("Sample Histopathology Images per NHL Subtype", fontsize=16, fontweight='bold')
plt.show()

import os
import matplotlib.pyplot as plt
import seaborn as sns

SPLIT_DIR = '/content/drive/MyDrive/NHL_Project/data/lymphoma_split'

class_folders = ['lymph_cll', 'lymph_fl', 'lymph_mcl']

splits = ['train', 'val', 'test']

split_counts = {}
for split in splits:
    split_path = os.path.join(SPLIT_DIR, split)
    counts = {folder: len(os.listdir(os.path.join(split_path, folder))) for folder in class_folders}
    split_counts[split] = counts

import pandas as pd
df_counts = pd.DataFrame(split_counts).T
df_counts.index.name = 'Split'
df_counts.columns = ['CLL', 'FL', 'MCL']

print("Image counts per split:")
print(df_counts)

df_counts.plot(kind='bar', figsize=(8,6))
plt.title('Number of Images per Class in Each Split')
plt.ylabel('Number of Images')
plt.xlabel('Data Split')
plt.xticks(rotation=0)
plt.show()

train_counts = df_counts.loc['train']
plt.figure(figsize=(6,6))
plt.pie(train_counts, labels=train_counts.index, autopct='%1.1f%%', startangle=140, colors=['#ff9999','#66b3ff','#99ff99'])
plt.title('Class Distribution in Training Set')
plt.show()

import random
from PIL import Image
image_sizes = {cls: [] for cls in class_folders}

for cls in class_folders:
    cls_path = os.path.join(SPLIT_DIR, 'train', cls)
    for img_name in os.listdir(cls_path):
        img_path = os.path.join(cls_path, img_name)
        img = Image.open(img_path)
        image_sizes[cls].append(img.size)

sizes_df = pd.DataFrame({cls: [h for w,h in image_sizes[cls]] for cls in class_folders})
plt.figure(figsize=(8,6))
sns.boxplot(data=sizes_df)
plt.title('Image Height Distribution per Class (Train Set)')
plt.ylabel('Height (pixels)')
plt.show()

for split in splits:
    counts = [len(os.listdir(os.path.join(SPLIT_DIR, split, cls))) for cls in class_folders]
    plt.figure(figsize=(6,6))
    plt.pie(counts, labels=[cls.upper() for cls in class_folders], autopct='%1.1f%%', startangle=140)
    plt.title(f'Class Distribution in {split.capitalize()} Set')
    plt.show()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

print("✓ Data augmentation configured")
print("  Training: rotation, shift, zoom, horizontal flip")
print("  Val/Test: rescaling only (no augmentation)")

train_generator = train_datagen.flow_from_directory(
    os.path.join(SPLIT_DIR, 'train'),
    target_size=CONFIG['IMG_SIZE'],
    batch_size=CONFIG['BATCH_SIZE'],
    class_mode='categorical',
    shuffle=True,
    seed=CONFIG['RANDOM_SEED']
)

validation_generator = val_test_datagen.flow_from_directory(
    os.path.join(SPLIT_DIR, 'val'),
    target_size=CONFIG['IMG_SIZE'],
    batch_size=CONFIG['BATCH_SIZE'],
    class_mode='categorical',
    shuffle=False,
    seed=CONFIG['RANDOM_SEED']
)

test_generator = val_test_datagen.flow_from_directory(
    os.path.join(SPLIT_DIR, 'test'),
    target_size=CONFIG['IMG_SIZE'],
    batch_size=CONFIG['BATCH_SIZE'],
    class_mode='categorical',
    shuffle=False,
    seed=CONFIG['RANDOM_SEED']
)

class_indices = train_generator.class_indices
index_to_class = {v: k for k, v in class_indices.items()}

print("\n" + "="*70)
print("DATA GENERATORS (FROM INDEPENDENT DIRECTORIES)")
print("="*70)
print(f"Training samples:   {train_generator.samples}")
print(f"Validation samples: {validation_generator.samples}")
print(f"Test samples:       {test_generator.samples}")
print(f"\nClass mapping: {class_indices}")
print("="*70)
print("\n✅ All generators use SEPARATE directories - NO DATA LEAKAGE")

class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)

class_weights = dict(enumerate(class_weights_array))
print("Class Weights (to handle imbalance):")
for idx, weight in class_weights.items():
    print(f"  {index_to_class[idx]}: {weight:.3f}")

@register_keras_serializable(package="Custom")
class TransformerBlock(layers.Layer):

    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        self.att = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim,
            dropout=dropout_rate
        )
        self.ffn = models.Sequential([
            layers.Dense(ff_dim, activation='gelu'),
            layers.Dropout(dropout_rate),
            layers.Dense(embed_dim)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout_rate': self.dropout_rate
        })
        return config

def create_hvit_nhl_model(input_shape=(128, 128, 3), num_classes=3,
                          transformer_blocks=1, embed_dim=128, num_heads=2):
    inputs = layers.Input(shape=input_shape, name='input_image')

    base_model = MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs,
        pooling=None
    )

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output

    H, W, C = x.shape[1], x.shape[2], x.shape[3]
    x = layers.Reshape((H * W, C))(x)
    x = layers.Dense(embed_dim)(x)

    positions = tf.range(H * W)
    position_embedding = layers.Embedding(
        input_dim=H * W,
        output_dim=embed_dim
    )(positions)
    x = x + position_embedding

    for i in range(transformer_blocks):
        x = TransformerBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=embed_dim * 2
        )(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)

    model = models.Model(inputs=inputs, outputs=outputs, name='H-ViT-NHL')
    return model

baseline_model = create_hvit_nhl_model(
    input_shape=(*CONFIG['IMG_SIZE'], 3),
    num_classes=CONFIG['NUM_CLASSES'],
    transformer_blocks=1,
    embed_dim=128,
    num_heads=2
)

print("\n" + "="*70)
print("BASELINE MODEL: Hybrid CNN + Vision Transformer (H-ViT-NHL)")
print("="*70)
baseline_model.summary()
print("="*70)

trainable_params = np.sum([tf.keras.backend.count_params(w)
                          for w in baseline_model.trainable_weights])
non_trainable_params = np.sum([tf.keras.backend.count_params(w)
                               for w in baseline_model.non_trainable_weights])
total_params = trainable_params + non_trainable_params

print("Model Parameters:")
print(f"  Total:        {total_params:,}")
print(f"  Trainable:    {trainable_params:,}")
print(f"  Non-trainable: {non_trainable_params:,}")
print(f"  Trainable ratio: {trainable_params/total_params*100:.1f}%")

baseline_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=CONFIG['LEARNING_RATE']),
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',
        keras.metrics.AUC(name='auc', multi_label=False)
    ]
)

print("✓ Baseline model compiled")
print(f"  Optimizer: Adam (lr={CONFIG['LEARNING_RATE']})")
print(f"  Loss: Categorical Cross-Entropy")
print(f"  Metrics: Accuracy, AUC")

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=f"{MODEL_DIR}/best_hvit_baseline.keras",
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

print("✓ Training callbacks configured")

history = baseline_model.fit(
    train_generator,
    epochs=CONFIG['EPOCHS'],
    validation_data=validation_generator,
    callbacks=callbacks,
    class_weight=class_weights
)

import os
os.listdir("/content/drive/MyDrive/NHL_Project/models")

def plot_training_history(history, title="Training History"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(history.history['loss'], label='Training', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0].set_title('Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.history['accuracy'], label='Training', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[1].set_title('Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(history.history['auc'], label='Training', linewidth=2)
    axes[2].plot(history.history['val_auc'], label='Validation', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[2].set_ylabel('AUC', fontsize=12, fontweight='bold')
    axes[2].set_title('AUC', fontsize=14, fontweight='bold')
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

plot_training_history(history, "Baseline H-ViT-NHL Training")

from tensorflow import keras

baseline_model = keras.models.load_model(
    f"{MODEL_DIR}/best_hvit_baseline.keras",
    compile=False
)

from tensorflow.keras.layers import GlobalAveragePooling1D

gap_layer = [layer for layer in baseline_model.layers if isinstance(layer, GlobalAveragePooling1D)][0]

feature_extractor = keras.Model(
    inputs=baseline_model.input,
    outputs=gap_layer.output
)

import numpy as np
dummy_input = np.random.rand(1, 128, 128, 3).astype(np.float32)
features = feature_extractor({'input_image': dummy_input})
print("Feature shape:", features.shape)

print("Evaluating baseline on TRULY HELD-OUT test set...\n")

test_steps = test_generator.samples // CONFIG['BATCH_SIZE']
y_pred_probs_baseline = baseline_model.predict(test_generator, steps=test_steps, verbose=1)
y_pred_classes_baseline = np.argmax(y_pred_probs_baseline, axis=1)
y_true_classes = test_generator.classes[:len(y_pred_classes_baseline)]

baseline_accuracy = np.mean(y_pred_classes_baseline == y_true_classes)

print(f"\n✅ Baseline Test Accuracy: {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
print(f"   Evaluated on {len(y_pred_classes_baseline)} truly held-out test samples")

cm_baseline = confusion_matrix(y_true_classes, y_pred_classes_baseline)
cm_baseline_norm = confusion_matrix(y_true_classes, y_pred_classes_baseline, normalize='true')

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Blues',
            xticklabels=CONFIG['CLASS_NAMES'],
            yticklabels=CONFIG['CLASS_NAMES'],
            ax=axes[0])
axes[0].set_xlabel('Predicted', fontsize=12, fontweight='bold')
axes[0].set_ylabel('True', fontsize=12, fontweight='bold')
axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')

sns.heatmap(cm_baseline_norm, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=CONFIG['CLASS_NAMES'],
            yticklabels=CONFIG['CLASS_NAMES'],
            ax=axes[1])
axes[1].set_xlabel('Predicted', fontsize=12, fontweight='bold')
axes[1].set_ylabel('True', fontsize=12, fontweight='bold')
axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')

plt.suptitle('Baseline H-ViT-NHL Performance (Held-Out Test Set)',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

sns.heatmap(cm_baseline_norm, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=CONFIG['CLASS_NAMES'], yticklabels=CONFIG['CLASS_NAMES'])
plt.title("Baseline H-ViT Confusion Matrix (Normalized)")
plt.show()

plt.figure(figsize=(8,6))
for i, class_name in enumerate(CONFIG['CLASS_NAMES']):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs_baseline[:, i])
    plt.plot(fpr, tpr, label=f"{class_name} (AUC = {baseline_roc_auc[i]:.2f})")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves - Baseline H-ViT")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

X_features = feature_extractor.predict(test_generator, steps=test_steps, verbose=1)

y_true = test_generator.classes[:X_features.shape[0]]

n_samples = X_features.shape[0]
perplexity = min(30, max(5, n_samples // 3))

print(f"t-SNE: n_samples={n_samples}, using perplexity={perplexity}")

tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
X_2d = tsne.fit_transform(X_features)

plt.figure(figsize=(8,6))
for idx, class_name in enumerate(CONFIG['CLASS_NAMES']):
    mask = (y_true == idx)
    plt.scatter(X_2d[mask, 0], X_2d[mask, 1], label=class_name, alpha=0.7, s=50)

plt.title("t-SNE Visualization of Test Set Features")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

print("\n" + "="*70)
print("BASELINE CLASSIFICATION REPORT (HELD-OUT TEST SET)")
print("="*70)
report_baseline = classification_report(
    y_true_classes,
    y_pred_classes_baseline,
    target_names=CONFIG['CLASS_NAMES'],
    digits=4
)
print(report_baseline)
print("="*70)

baseline_precision, baseline_recall, baseline_f1, _ = precision_recall_fscore_support(
    y_true_classes, y_pred_classes_baseline, average='macro'
)

y_true_bin = label_binarize(y_true_classes, classes=[0, 1, 2])

baseline_roc_auc = {}
for i in range(CONFIG['NUM_CLASSES']):
    baseline_roc_auc[i] = roc_auc_score(y_true_bin[:, i], y_pred_probs_baseline[:, i])

baseline_macro_auc = roc_auc_score(y_true_bin, y_pred_probs_baseline, average='macro')

print("\nBaseline ROC-AUC (Held-Out Test):")
for i, class_name in enumerate(CONFIG['CLASS_NAMES']):
    print(f"  {class_name}: {baseline_roc_auc[i]:.4f}")
print(f"  Macro-Average: {baseline_macro_auc:.4f}")
