# Copyright 2018 Google LLC
#
# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.
# =============================================================================
"""Artifact conversion to and from Python Keras."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

import h5py

import h5_conversion  # pylint: disable=import-error


def dispatch_pykeras_conversion(h5_path, output_dir=None):
  """Converts a Keras HDF5 saved-model file to TensorFlow.js format.

  Auto-detects saved_model versus weights-only and generates the correct
  json in either case. This function accepts Keras HDF5 files in two formats:
    - A weights-only HDF5 (e.g., generated with Keras Model's `save_weights()`
      method),
    - A topology+weights combined HDF5 (e.g., generated with
      `keras.model.save_model`).

  Args:
    h5_path: path to an HDF5 file containing keras model data as a `str`.
    output_dir: Output directory to which the TensorFlow.js-format model JSON
      file and weights files will be written. If the directory does not exist,
      it will be created.

  Returns:
    (model_json, groups)
      model_json: a json dictionary (empty if unused) for model topology.
        If `h5_path` points to a weights-only HDF5 file, this return value
        will be `None`.
      groups: an array of weight_groups as defined in tfjs weights_writer.
  """
  converter = h5_conversion.HDF5Converter()

  h5_file = h5py.File(h5_path)
  if 'layer_names' in h5_file.attrs:
    model_json = None
    groups = converter.h5_weights_to_tfjs_format(h5_file)
  else:
    model_json, groups = converter.h5_merged_saved_model_to_tfjs_format(h5_file)

  if output_dir:
    if os.path.isfile(output_dir):
      raise ValueError('Output path "%s" already exists as a file' % output_dir)
    elif not os.path.isdir(output_dir):
      os.makedirs(output_dir)
    converter.write_artifacts(model_json, groups, output_dir)

  return model_json, groups


def main():
  parser = argparse.ArgumentParser(
      'Convert saved models generated by Python Keras to TensorFlow.js'
      ' format.')
  parser.add_argument(
      'h5_path', type=str, help='Path to the input Keras HDF5 (.h5) file.'
      'This file can be one of the two following formants:\n'
      '  - A topology+weights combined HDF5 (e.g., generated with'
      '    `keras.model.save_model()` method).\n'
      '  - A weights-only HDF5 (e.g., generated with Keras Model\'s '
      '    `save_weights()` method).')
  parser.add_argument(
      'output_dir', type=str, help='Path for all output artifacts.')

  FLAGS = parser.parse_args()
  dispatch_pykeras_conversion(FLAGS.h5_path, output_dir=FLAGS.output_dir)


if __name__ == '__main__':
  main()
