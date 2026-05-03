"""
Knowledge distillation: compress the top-3 teacher ensemble into a small MLP student.

Teacher: RF + XGBoost + LightGBM (weighted by 1/RMSE)
Student: 3-layer MLP (64 → 32 → n_outputs)
Loss:    α × MSE(student, soft_targets) + (1−α) × MSE(student, true_labels)
         where α = 0.7 (favour teacher guidance)
"""
import numpy as np
import pickle
from pathlib import Path
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2

ALPHA        = 0.7   # weight for soft-target loss
TEMPERATURE  = 2.0   # temperature scaling for soft targets
STUDENT_PATH = Path(__file__).parent.parent.parent / "distilled_mlp.keras"


def _soft_targets(teachers: list, X: np.ndarray, weights: list) -> np.ndarray:
    """Compute weighted average of teacher predictions (soft targets)."""
    preds = np.stack([t.predict(X) for t in teachers], axis=0)   # (n_teachers, n_samples, n_outputs)
    w     = np.array(weights) / sum(weights)
    return np.einsum("i,ijk->jk", w, preds)                       # (n_samples, n_outputs)


def _build_student(n_features: int, n_outputs: int) -> tf.keras.Model:
    model = Sequential([
        Dense(64, activation="relu", input_shape=(n_features,), kernel_regularizer=l2(1e-4)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation="relu", kernel_regularizer=l2(1e-4)),
        Dropout(0.2),
        Dense(n_outputs),
    ])
    return model


def distill(
    teachers: list,
    teacher_rmses: list,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 150,
    batch_size: int = 64,
) -> tf.keras.Model:
    """
    Train a student MLP that learns from both the teacher ensemble and true labels.

    Args:
        teachers      : list of fitted teacher models
        teacher_rmses : RMSE for each teacher (lower = higher weight)
        X_train, y_train : training features and true AQI targets
        X_val, y_val     : validation set (for early stopping)

    Returns:
        Fitted Keras student model.
    """
    # Inverse-RMSE weights: better teachers get higher weight
    weights = [1.0 / max(r, 1e-6) for r in teacher_rmses]

    soft = _soft_targets(teachers, X_train, weights)           # (n_train, n_outputs)
    soft_val = _soft_targets(teachers, X_val, weights)

    n_features = X_train.shape[1]
    n_outputs  = y_train.shape[1] if y_train.ndim > 1 else 1

    student = _build_student(n_features, n_outputs)

    @tf.function
    def distill_loss(y_true_batch, soft_batch, y_pred):
        mse_true = tf.reduce_mean(tf.square(y_pred - tf.cast(y_true_batch, tf.float32)))
        mse_soft = tf.reduce_mean(tf.square(y_pred - tf.cast(soft_batch, tf.float32)))
        return ALPHA * mse_soft + (1.0 - ALPHA) * mse_true

    optimizer = Adam(learning_rate=1e-3)

    best_val_loss = np.inf
    patience_counter = 0
    patience = 15

    for epoch in range(epochs):
        # Mini-batch training
        indices  = np.random.permutation(len(X_train))
        epoch_loss = []
        for start in range(0, len(X_train), batch_size):
            idx = indices[start:start + batch_size]
            X_b = tf.constant(X_train[idx], dtype=tf.float32)
            y_b = tf.constant(y_train[idx], dtype=tf.float32)
            s_b = tf.constant(soft[idx],    dtype=tf.float32)
            with tf.GradientTape() as tape:
                preds = student(X_b, training=True)
                loss  = distill_loss(y_b, s_b, preds)
            grads = tape.gradient(loss, student.trainable_variables)
            optimizer.apply_gradients(zip(grads, student.trainable_variables))
            epoch_loss.append(float(loss))

        # Validation
        val_preds = student(tf.constant(X_val, dtype=tf.float32), training=False)
        val_loss  = float(distill_loss(
            tf.constant(y_val, dtype=tf.float32),
            tf.constant(soft_val, dtype=tf.float32),
            val_preds,
        ))
        if epoch % 20 == 0:
            print(f"[distillation] Epoch {epoch:3d} | train={np.mean(epoch_loss):.4f} | val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            student.save(STUDENT_PATH)   # save best weights
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[distillation] Early stopping at epoch {epoch}")
                break

    # Reload best weights
    student = tf.keras.models.load_model(STUDENT_PATH)
    print(f"[distillation] Student MLP trained. Best val loss: {best_val_loss:.4f}")
    return student


def load_student() -> tf.keras.Model:
    """Load the saved distilled student model."""
    return tf.keras.models.load_model(STUDENT_PATH)
