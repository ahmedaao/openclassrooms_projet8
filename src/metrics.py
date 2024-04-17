import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy


@tf.keras.utils.register_keras_serializable()
def dice_coeff(y_true, y_pred):
    """Compute metric dice_coeff

    Args:
        y_true (array): Ground True 3D array
        y_pred (array): Segmentation mask

    Returns:
        float: Score between 0 (no match) and 1 (perfect match)
    """
    smooth = 1.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score


@tf.keras.utils.register_keras_serializable()
def dice_loss(y_true, y_pred):
    """Compute metric dice_loss

    Args:
        y_true (array): Ground True 3D array
        y_pred (array): Segmented mask 3D array

    Returns:
        float: Loss value. As the value tends towards 0, it is better
    """
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss


@tf.keras.utils.register_keras_serializable()
def total_loss(y_true, y_pred):
    """Compute metric total_loss

    Args:
        y_true (array): Ground True 3D array
        y_pred (_type_): Segmented mask 3D array

    Returns:
        float: Total loss value. As the value tends towards 0, it is better
    """
    loss = binary_crossentropy(y_true, y_pred) + (3 * dice_loss(y_true, y_pred))
    return loss


@tf.keras.utils.register_keras_serializable()
def jaccard(y_true, y_pred):
    """Compute metric Jaccard

    Args:
        y_true (array): Ground True 3D array
        y_pred (_type_): Segmented mask 3D array

    Returns:
        float: As the value tends towards 0, it is better
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)
