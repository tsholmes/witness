import pandas as pd
import numpy as np
from PIL import Image
from keras import losses, backend as K, layers
from tqdm import tqdm
import wandb

def make_segmentation_dataframe(
  model,
  image_filenames,
  mask_filenames,
  image_size=None,
  loss_function=losses.binary_crossentropy
):
  example_ids = []
  images = []
  labels = []
  predictions = []
  predictions_discrete = []
  incorrect_labels = []
  losses = []
  accuracies = []
  precisions = []
  recalls = []
  
  if image_size is None:
    image_size = Image.open(image_filenames[0]).size[:2] # only use the width/height
    image_size = (image_size[1], image_size[0])

  # PIL takes width,height but keras takes height,width
  im_resize_sz = (image_size[1], image_size[0])
  
  true_mask_input = layers.Input(image_size + (1,))
  pred_mask_input = layers.Input(image_size + (1,))
  loss_output = loss_function(true_mask_input, pred_mask_input)
  kloss_func = K.function(inputs=[true_mask_input, pred_mask_input], outputs=[loss_output])
  
  print('Building result dataframe')
  for img_file, mask_file in tqdm(zip(image_filenames, mask_filenames), total=len(image_filenames)):
    img = np.array(Image.open(img_file).resize(im_resize_sz)) / 255.0
    
    true_mask = np.array(Image.open(mask_file).convert('L').resize(im_resize_sz)) / 255.0
    true_mask = true_mask[:,:,np.newaxis]
    true_mask[true_mask > 0.5] = 1.
    true_mask[true_mask <= 0.5] = 0.
    
    pred_mask = model.predict([[img]])[0]
    
    pred_mask_discrete = pred_mask.copy()
    pred_mask_discrete[pred_mask_discrete > 0.5] = 1.
    pred_mask_discrete[pred_mask_discrete <= 0.5] = 0.
    
    pred_mask_incorrect = pred_mask_discrete != true_mask
    
    loss = kloss_func([[true_mask], [pred_mask]])[0]
    acc = np.mean(true_mask == pred_mask_discrete)
    
    true_positive = np.sum(np.logical_and(true_mask, pred_mask_discrete))
    false_positive = np.sum(np.logical_and(np.logical_not(true_mask), pred_mask_discrete))
    false_negative = np.sum(np.logical_and(true_mask, np.logical_not(pred_mask_discrete)))
    
    # small value to make the denominators non-zero
    epsilon = 1e-9
    
    precision = true_positive / (true_positive + false_positive + epsilon)
    recall = true_positive / (true_positive + false_negative + epsilon)
    
    example_ids.append(img_file)
    images.append(wandb.Image(img))
    labels.append(wandb.Image(true_mask))
    predictions.append(wandb.Image(pred_mask))
    predictions_discrete.append(wandb.Image(pred_mask_discrete))
    incorrect_labels.append(wandb.Image(pred_mask_incorrect))
    losses.append(loss)
    accuracies.append(acc)
    precisions.append(precision)
    recalls.append(recall)

  return pd.DataFrame({
    'wandb_example_id': example_ids,
    'image': images,
    'label': labels,
    'prediction': predictions,
    'prediction_discrete': predictions_discrete,
    'incorrect_label': incorrect_labels,
    'loss': losses,
    'accuracy': accuracies,
    'precision': precisions,
    'recall': recalls,
  }, columns=['wandb_example_id', 'image', 'label', 'prediction', 'prediction_discrete', 'incorrect_label', 'loss', 'accuracy', 'precision', 'recall'])