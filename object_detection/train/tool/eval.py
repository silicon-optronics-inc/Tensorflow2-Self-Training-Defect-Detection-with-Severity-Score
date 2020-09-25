# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 13:19:27 2020

@author: martinchen
"""

from absl import flags
from object_detection import eval_util
from object_detection import inputs
from object_detection import model_lib

from object_detection.core import standard_fields as fields
from object_detection.utils import label_map_util
from object_detection.utils import ops
from object_detection.utils import visualization_utils as vutils
import os
import tensorflow.compat.v1 as tf
import time

# pylint: disable=g-import-not-at-top
try:
  from tensorflow.contrib import tpu as contrib_tpu
except ImportError:
  # TF 2.0 doesn't ship with contrib.
  pass
# pylint: enable=g-import-not-at-top
  
  
flags.DEFINE_string('model_dir', None, 'Path to output model directory '
                    'where event and checkpoint files will be written.')
flags.DEFINE_string('pipeline_config_path', None, 'Path to pipeline config '
                    'file.')
flags.DEFINE_string('mode', '', 'Mode of program,' 
                    '0: train only,'
                    '1: train + evaluate,'
                    '2: semi-supervised learning,'
                    '3: evaluate only')
flags.DEFINE_string('eval_index', 'DetectionBoxes_Precision/mAP@.50IOU', 'Evaluation index to start auto-label')
flags.DEFINE_string('eval_threshold', '0.9', 'Evaluation threshold to start auto-label')
                    
                    
flags.DEFINE_integer('wait_interval', 300, 'Number of seconds to wait before evaluating latest model.')
flags.DEFINE_integer('eval_timeout', 3600, 'Number of seconds to wait before exiting.')

FLAGS = flags.FLAGS



def _compute_losses_and_predictions_dicts(
    model, features, labels,
    add_regularization_loss=True):
  """Computes the losses dict and predictions dict for a model on inputs.

  Args:
    model: a DetectionModel (based on Keras).
    features: Dictionary of feature tensors from the input dataset.
      Should be in the format output by `inputs.train_input` and
      `inputs.eval_input`.
        features[fields.InputDataFields.image] is a [batch_size, H, W, C]
          float32 tensor with preprocessed images.
        features[HASH_KEY] is a [batch_size] int32 tensor representing unique
          identifiers for the images.
        features[fields.InputDataFields.true_image_shape] is a [batch_size, 3]
          int32 tensor representing the true image shapes, as preprocessed
          images could be padded.
        features[fields.InputDataFields.original_image] (optional) is a
          [batch_size, H, W, C] float32 tensor with original images.
    labels: A dictionary of groundtruth tensors post-unstacking. The original
      labels are of the form returned by `inputs.train_input` and
      `inputs.eval_input`. The shapes may have been modified by unstacking with
      `model_lib.unstack_batch`. However, the dictionary includes the following
      fields.
        labels[fields.InputDataFields.num_groundtruth_boxes] is a
          int32 tensor indicating the number of valid groundtruth boxes
          per image.
        labels[fields.InputDataFields.groundtruth_boxes] is a float32 tensor
          containing the corners of the groundtruth boxes.
        labels[fields.InputDataFields.groundtruth_classes] is a float32
          one-hot tensor of classes.
        labels[fields.InputDataFields.groundtruth_weights] is a float32 tensor
          containing groundtruth weights for the boxes.
        -- Optional --
        labels[fields.InputDataFields.groundtruth_instance_masks] is a
          float32 tensor containing only binary values, which represent
          instance masks for objects.
        labels[fields.InputDataFields.groundtruth_keypoints] is a
          float32 tensor containing keypoints for each box.
        labels[fields.InputDataFields.groundtruth_dp_num_points] is an int32
          tensor with the number of sampled DensePose points per object.
        labels[fields.InputDataFields.groundtruth_dp_part_ids] is an int32
          tensor with the DensePose part ids (0-indexed) per object.
        labels[fields.InputDataFields.groundtruth_dp_surface_coords] is a
          float32 tensor with the DensePose surface coordinates.
        labels[fields.InputDataFields.groundtruth_group_of] is a tf.bool tensor
          containing group_of annotations.
        labels[fields.InputDataFields.groundtruth_labeled_classes] is a float32
          k-hot tensor of classes.
        labels[fields.InputDataFields.groundtruth_track_ids] is a int32
          tensor of track IDs.
    add_regularization_loss: Whether or not to include the model's
      regularization loss in the losses dictionary.

  Returns:
    A tuple containing the losses dictionary (with the total loss under
    the key 'Loss/total_loss'), and the predictions dictionary produced by
    `model.predict`.

  """
  model_lib.provide_groundtruth(model, labels)
  preprocessed_images = features[fields.InputDataFields.image]

  prediction_dict = model.predict(
      preprocessed_images,
      features[fields.InputDataFields.true_image_shape],
      **model.get_side_inputs(features))
  prediction_dict = ops.bfloat16_to_float32_nested(prediction_dict)

  losses_dict = model.loss(
      prediction_dict, features[fields.InputDataFields.true_image_shape])
  losses = [loss_tensor for loss_tensor in losses_dict.values()]
  if add_regularization_loss:
    # TODO(kaftan): As we figure out mixed precision & bfloat 16, we may
    ## need to convert these regularization losses from bfloat16 to float32
    ## as well.
    regularization_losses = model.regularization_losses()
    if regularization_losses:
      regularization_losses = ops.bfloat16_to_float32_nested(
          regularization_losses)
      regularization_loss = tf.add_n(
          regularization_losses, name='regularization_loss')
      losses.append(regularization_loss)
      losses_dict['Loss/regularization_loss'] = regularization_loss

  total_loss = tf.add_n(losses, name='total_loss')
  losses_dict['Loss/total_loss'] = total_loss

  return losses_dict, prediction_dict


def eager_eval_loop(
    detection_model,
    configs,
    eval_dataset,
    use_tpu=False,
    postprocess_on_cpu=False,
    global_step=None):
  """Evaluate the model eagerly on the evaluation dataset.

  This method will compute the evaluation metrics specified in the configs on
  the entire evaluation dataset, then return the metrics. It will also log
  the metrics to TensorBoard.

  Args:
    detection_model: A DetectionModel (based on Keras) to evaluate.
    configs: Object detection configs that specify the evaluators that should
      be used, as well as whether regularization loss should be included and
      if bfloat16 should be used on TPUs.
    eval_dataset: Dataset containing evaluation data.
    use_tpu: Whether a TPU is being used to execute the model for evaluation.
    postprocess_on_cpu: Whether model postprocessing should happen on
      the CPU when using a TPU to execute the model.
    global_step: A variable containing the training step this model was trained
      to. Used for logging purposes.

  Returns:
    A dict of evaluation metrics representing the results of this evaluation.
  """
  train_config = configs['train_config']
  eval_input_config = configs['eval_input_config']
  eval_config = configs['eval_config']
  add_regularization_loss = train_config.add_regularization_loss

  is_training = False
  detection_model._is_training = is_training  # pylint: disable=protected-access
  tf.keras.backend.set_learning_phase(is_training)

  evaluator_options = eval_util.evaluator_options_from_eval_config(
      eval_config)

  class_agnostic_category_index = (
      label_map_util.create_class_agnostic_category_index())
  class_agnostic_evaluators = eval_util.get_evaluators(
      eval_config,
      list(class_agnostic_category_index.values()),
      evaluator_options)

  class_aware_evaluators = None
  if eval_input_config.label_map_path:
    class_aware_category_index = (
        label_map_util.create_category_index_from_labelmap(
            eval_input_config.label_map_path))
    class_aware_evaluators = eval_util.get_evaluators(
        eval_config,
        list(class_aware_category_index.values()),
        evaluator_options)

  evaluators = None
  loss_metrics = {}

  @tf.function
  def compute_eval_dict(features, labels):
    """Compute the evaluation result on an image."""
    # For evaling on train data, it is necessary to check whether groundtruth
    # must be unpadded.
    boxes_shape = (
        labels[fields.InputDataFields.groundtruth_boxes].get_shape().as_list())
    unpad_groundtruth_tensors = boxes_shape[1] is not None and not use_tpu
    labels = model_lib.unstack_batch(
        labels, unpad_groundtruth_tensors=unpad_groundtruth_tensors)

    losses_dict, prediction_dict = _compute_losses_and_predictions_dicts(
        detection_model, features, labels, add_regularization_loss)

    def postprocess_wrapper(args):
      return detection_model.postprocess(args[0], args[1])

    # TODO(kaftan): Depending on how postprocessing will work for TPUS w/
    ## TPUStrategy, may be good to move wrapping to a utility method
    if use_tpu and postprocess_on_cpu:
      detections = contrib_tpu.outside_compilation(
          postprocess_wrapper,
          (prediction_dict, features[fields.InputDataFields.true_image_shape]))
    else:
      detections = postprocess_wrapper(
          (prediction_dict, features[fields.InputDataFields.true_image_shape]))

    class_agnostic = (
        fields.DetectionResultFields.detection_classes not in detections)
    # TODO(kaftan) (or anyone): move `_prepare_groundtruth_for_eval to eval_util
    ## and call this from there.
    groundtruth = model_lib._prepare_groundtruth_for_eval(  # pylint: disable=protected-access
        detection_model, class_agnostic, eval_input_config.max_number_of_boxes)
    use_original_images = fields.InputDataFields.original_image in features
    if use_original_images:
      eval_images = features[fields.InputDataFields.original_image]
      true_image_shapes = tf.slice(
          features[fields.InputDataFields.true_image_shape], [0, 0], [-1, 3])
      original_image_spatial_shapes = features[
          fields.InputDataFields.original_image_spatial_shape]
    else:
      eval_images = features[fields.InputDataFields.image]
      true_image_shapes = None
      original_image_spatial_shapes = None

    eval_dict = eval_util.result_dict_for_batched_example(
        eval_images,
        features[inputs.HASH_KEY],
        detections,
        groundtruth,
        class_agnostic=class_agnostic,
        scale_to_absolute=True,
        original_image_spatial_shapes=original_image_spatial_shapes,
        true_image_shapes=true_image_shapes)

    return eval_dict, losses_dict, class_agnostic

  agnostic_categories = label_map_util.create_class_agnostic_category_index()
  per_class_categories = label_map_util.create_category_index_from_labelmap(
      eval_input_config.label_map_path)
  keypoint_edges = [
      (kp.start, kp.end) for kp in eval_config.keypoint_edge]

  for i, (features, labels) in enumerate(eval_dataset):
    eval_dict, losses_dict, class_agnostic = compute_eval_dict(features, labels)

    if class_agnostic:
      category_index = agnostic_categories
    else:
      category_index = per_class_categories

    if i % 100 == 0:
      tf.logging.info('Finished eval step %d', i)

    use_original_images = fields.InputDataFields.original_image in features
    if use_original_images and i < eval_config.num_visualizations:
      sbys_image_list = vutils.draw_side_by_side_evaluation_image(
          eval_dict,
          category_index=category_index,
          max_boxes_to_draw=eval_config.max_num_boxes_to_visualize,
          min_score_thresh=eval_config.min_score_threshold,
          use_normalized_coordinates=False,
          keypoint_edges=keypoint_edges or None)
      sbys_images = tf.concat(sbys_image_list, axis=0)
      tf.compat.v2.summary.image(
          name='eval_side_by_side_' + str(i),
          step=global_step,
          data=sbys_images,
          max_outputs=eval_config.num_visualizations)
      if eval_util.has_densepose(eval_dict):
        dp_image_list = vutils.draw_densepose_visualizations(
            eval_dict)
        dp_images = tf.concat(dp_image_list, axis=0)
        tf.compat.v2.summary.image(
            name='densepose_detections_' + str(i),
            step=global_step,
            data=dp_images,
            max_outputs=eval_config.num_visualizations)

    if evaluators is None:
      if class_agnostic:
        evaluators = class_agnostic_evaluators
      else:
        evaluators = class_aware_evaluators

    for evaluator in evaluators:
      evaluator.add_eval_dict(eval_dict)

    for loss_key, loss_tensor in iter(losses_dict.items()):
      if loss_key not in loss_metrics:
        loss_metrics[loss_key] = tf.keras.metrics.Mean()
      # Skip the loss with value equal or lower than 0.0 when calculating the
      # average loss since they don't usually reflect the normal loss values
      # causing spurious average loss value.
      if loss_tensor <= 0.0:
        continue
      loss_metrics[loss_key].update_state(loss_tensor)

  eval_metrics = {}

  for evaluator in evaluators:
    eval_metrics.update(evaluator.evaluate())
  for loss_key in loss_metrics:
    eval_metrics[loss_key] = loss_metrics[loss_key].result()

  eval_metrics = {str(k): v for k, v in eval_metrics.items()}
  tf.logging.info('Eval metrics at step %d', global_step)
  for k in eval_metrics:
    tf.compat.v2.summary.scalar(k, eval_metrics[k], step=global_step)
    tf.logging.info('\t+ %s: %f', k, eval_metrics[k])

  return eval_metrics


def generate_model_list(model_dir):
  """Generate model list.
  Args:
    model_dir: Path to model be saved.
    
  Returns:
    A list of model name.
  """
  file_list = os.listdir(model_dir)
  model_list = []
  for file in file_list:
      if 'ckpt' in file:
          if file.split('.')[0] not in model_list:
              model_list.append(file.split('.')[0])
  # sort files in reverse
  model_list.sort(key=lambda x: int(x.split('.')[0].split('-')[1]), reverse=False)
  return model_list
    
def generate_eval_metric(detection_model, model_dir, model_step, eval_inputs, configs, eval_index=0):
  global_step = tf.compat.v2.Variable(
            0, trainable=False, dtype=tf.compat.v2.dtypes.int64)
  ckpt = tf.compat.v2.train.Checkpoint(
              step=global_step, model=detection_model)
        
  ckpt.restore(os.path.join(model_dir, model_step)).expect_partial()
  if eval_index is not None:
      eval_inputs = [eval_inputs[eval_index]]

  for eval_name, eval_input in eval_inputs:
      summary_writer = tf.compat.v2.summary.create_file_writer(
          os.path.join(model_dir, 'eval', eval_name))
      with summary_writer.as_default():
          eval_metric = eager_eval_loop(
              detection_model,
              configs,
              eval_input,
              use_tpu=False,
              postprocess_on_cpu=False,
              global_step=global_step)
  return eval_metric
    
def show_time_taken(tStart):
  t_hour, temp_sec = divmod(time.time() - tStart, 3600)
  t_min, t_sec = divmod(temp_sec, 60)
  msg = '{} hours, {} mins, {} seconds'.format(int(t_hour), int(t_min), int(t_sec))
  return msg

def main(unused_argv):
    MODEL_BUILD_UTIL_MAP = model_lib.MODEL_BUILD_UTIL_MAP
    
    get_configs_from_pipeline_file = MODEL_BUILD_UTIL_MAP[
          'get_configs_from_pipeline_file']
    configs = get_configs_from_pipeline_file(
          FLAGS.pipeline_config_path, config_override=None)
    
    model_config = configs['model']
    eval_config = configs['eval_config']
    eval_input_configs = configs['eval_input_configs']
    detection_model = MODEL_BUILD_UTIL_MAP['detection_model_fn_base'](
          model_config=model_config, is_training=True)
    
    # Create the inputs.
    eval_inputs = []
    for eval_input_config in eval_input_configs:
        next_eval_input = inputs.eval_input(
            eval_config=eval_config,
            eval_input_config=eval_input_config,
            model_config=model_config,
            model=detection_model)
        eval_inputs.append((eval_input_config.name, next_eval_input))
    
    
    
    if FLAGS.mode != '3':
        # Scan for new model files
        model_list = generate_model_list(FLAGS.model_dir)
        timeout_count = 0
        latest_model = 'ckpt-0' if len(model_list) == 0 else model_list[-1]
        start_model_num = int(latest_model.split('-')[-1])
        while(timeout_count<=FLAGS.eval_timeout):
            print('Wait for {} seconds before starting a new evaluation.'.format(FLAGS.wait_interval))
            time.sleep(FLAGS.wait_interval)
            timeout_count += FLAGS.wait_interval
            model_list = generate_model_list(FLAGS.model_dir)
            
            if model_list[-1] != latest_model:
                timeout_count = 0
                print('\nEvaluating {}'.format(model_list[-1]))
                tStart = time.time()
                latest_model = model_list[-1]
                eval_metric = generate_eval_metric(detection_model, FLAGS.model_dir, latest_model, eval_inputs, configs)
                print('Takes {} to evaluate {}'.format(show_time_taken(tStart), latest_model))
                
                # Automatic labeling restart after at least 10 models produced
                model_num = int(latest_model.split('-')[-1])
                if model_num >= start_model_num+10:
                    eval_index = eval_metric[FLAGS.eval_index]
                    if eval_index >= float(FLAGS.eval_threshold) and FLAGS.mode == '2':
                        break
                

                
    else:
        model_list = generate_model_list(FLAGS.model_dir)
        for model_step in model_list:
            print('\nEvaluating {}'.format(model_step))
            tStart = time.time()
            eval_metric = generate_eval_metric(detection_model, FLAGS.model_dir, model_step, eval_inputs, configs)
            print('Takes {} to evaluate {}'.format(show_time_taken(tStart), model_step))
            
        
if __name__ == '__main__':
    tf.compat.v1.app.run()