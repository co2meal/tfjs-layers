/**
 * @license
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 * =============================================================================
 */

/* Original Source: losses.py */
import * as tfc from '@tensorflow/tfjs-core';
import {Tensor, tidy} from '@tensorflow/tfjs-core';

import * as K from './backend/tfjs_backend';
import {ValueError} from './errors';
import {LossOrMetricFn} from './types';

/**
 * Loss or metric function: Mean squared error.
 *
 * ```js
 * const yTrue = tf.tensor2d([[0, 1], [3, 4]]);
 * const yPred = tf.tensor2d([[0, 1], [-3, -4]]);
 * const mse = tf.metrics.meanSquaredError(yTrue, yPred);
 * mse.print();
 * ```
 *
 * Aliases: `tf.metrics.MSE`, `tf.metrics.mse`.
 *
 * @param yTrue Truth Tensor.
 * @param yPred Prediction Tensor.
 * @return Mean squared error Tensor.
 */
export function meanSquaredError(yTrue: Tensor, yPred: Tensor): Tensor {
  return tidy(() => {
    return tfc.mean(K.square(tfc.sub(yPred, yTrue)), -1);
  });
}

/**
 * Loss or metric function: Mean absolute error.
 *
 * Mathematically, mean absolute error is defined as:
 *   `mean(abs(yPred - yTrue))`,
 * wherein the `mean` is applied over feature dimensions.
 *
 * ```js
 * const yTrue = tf.tensor2d([[0, 1], [0, 0], [2, 3]]);
 * const yPred = tf.tensor2d([[0, 1], [0, 1], [-2, -3]]);
 * const mse = tf.metrics.meanAbsoluteError(yTrue, yPred);
 * mse.print();
 * ```
 *
 * @param yTrue Truth Tensor.
 * @param yPred Prediction Tensor.
 * @return Mean absolute error Tensor.
 */
export function meanAbsoluteError(yTrue: Tensor, yPred: Tensor): Tensor {
  return tidy(() => {
    return tfc.mean(tfc.abs(tfc.sub(yPred, yTrue)), -1);
  });
}

/**
 * Loss or metric function: Mean absolute percentage error.
 *
 * ```js
 * const yTrue = tf.tensor2d([[0, 1], [10, 20]]);
 * const yPred = tf.tensor2d([[0, 1], [11, 24]]);
 * const mse = tf.metrics.meanAbsolutePercentageError(yTrue, yPred);
 * mse.print();
 * ```
 *
 * Aliases: `tf.metrics.MAPE`, `tf.metrics.mape`.
 *
 * @param yTrue Truth Tensor.
 * @param yPred Prediction Tensor.
 * @return Mean absolute percentage error Tensor.
 */
export function meanAbsolutePercentageError(
    yTrue: Tensor, yPred: Tensor): Tensor {
  return tidy(() => {
    const diff = tfc.sub(yTrue, yPred);
    const clippedTrue =
        tfc.clipByValue(tfc.abs(yTrue), K.epsilon(), Number.MAX_VALUE);
    const absResult = tfc.abs(tfc.div(diff, clippedTrue));
    return K.scalarTimesArray(K.getScalar(100.0), tfc.mean(absResult, -1));
  });
}

export function meanSquaredLogarithmicError(
    yTrue: Tensor, yPred: Tensor): Tensor {
  return tidy(() => {
    const one = K.getScalar(1.0);

    const clippedPred = tfc.clipByValue(yPred, K.epsilon(), Number.MAX_VALUE);
    const firstLog = tfc.log(K.scalarPlusArray(one, clippedPred));

    const clippedTrue = tfc.clipByValue(yTrue, K.epsilon(), Number.MAX_VALUE);
    const secondLog = tfc.log(K.scalarPlusArray(one, clippedTrue));

    return tfc.mean(K.square(tfc.sub(firstLog, secondLog)), -1);
  });
}

export function squaredHinge(yTrue: Tensor, yPred: Tensor): Tensor {
  return tidy(() => {
    const zeroTensor = K.getScalar(0.0);
    const one = K.getScalar(1.0);
    const maxResult =
        tfc.maximum(zeroTensor, tfc.sub(one, tfc.mul(yTrue, yPred)));
    return tfc.mean(K.square(maxResult), -1);
  });
}

export function hinge(yTrue: Tensor, yPred: Tensor): Tensor {
  return tidy(() => {
    const zeroTensor = K.getScalar(0.0);
    const one = K.getScalar(1.0);
    const maxResult =
        tfc.maximum(zeroTensor, tfc.sub(one, tfc.mul(yTrue, yPred)));
    return tfc.mean(maxResult, -1);
  });
}

export function categoricalHinge(yTrue: Tensor, yPred: Tensor): Tensor {
  return tidy(() => {
    const zeroTensor = K.getScalar(0.0);
    const one = K.getScalar(1.0);
    const pos = tfc.sum(tfc.mul(yTrue, yPred), -1);
    const neg = tfc.max(tfc.mul(tfc.sub(one, yTrue), yPred), -1);
    return tfc.maximum(zeroTensor, K.scalarPlusArray(one, tfc.sub(neg, pos)));
  });
}

/**
 * Logarithm of the hyperbolic cosine of the prediction error.
 *
 * `log(cosh(x))` is approximately equal to `(x ** 2) / 2` for small `x` and
 * to `abs(x) - log(2)` for large `x`. This means that 'logcosh' works mostly
 * like the mean squared error, but will not be so strongly affected by the
 * occasional wildly incorrect prediction.
 */
export function logcosh(yTrue: Tensor, yPred: Tensor): Tensor {
  return tidy(() => {
    const log2 = K.getScalar(Math.log(2.0));
    const predictionDiff = tfc.sub(yPred, yTrue);
    const logcoshResult = tfc.sub(
        tfc.add(
            predictionDiff,
            K.softplus(K.scalarTimesArray(K.getScalar(-2.0), predictionDiff))),
        log2);
    return tfc.mean(logcoshResult, -1);
  });
}

export function categoricalCrossentropy(yTrue: Tensor, yPred: Tensor): Tensor {
  return tidy(() => {
    return K.categoricalCrossentropy(yTrue, yPred);
  });
}

export function sparseCategoricalCrossentropy(
    yTrue: Tensor, yPred: Tensor): Tensor {
  return tidy(() => {
    return K.sparseCategoricalCrossentropy(yTrue, yPred);
  });
}

export function binaryCrossentropy(yTrue: Tensor, yPred: Tensor): Tensor {
  return tidy(() => {
    return tfc.mean(K.binaryCrossentropy(yTrue, yPred), -1);
  });
}

export function kullbackLeiblerDivergence(
    yTrue: Tensor, yPred: Tensor): Tensor {
  return tidy(() => {
    const clippedTrue = tfc.clipByValue(yTrue, K.epsilon(), 1);
    const clippedPred = tfc.clipByValue(yPred, K.epsilon(), 1);
    return tfc.sum(
        tfc.mul(yTrue, tfc.log(tfc.div(clippedTrue, clippedPred))), -1);
  });
}

export function poisson(yTrue: Tensor, yPred: Tensor): Tensor {
  return tidy(() => {
    const logPred = tfc.log(K.scalarPlusArray(K.getScalar(K.epsilon()), yPred));
    return tfc.mean(tfc.sub(yPred, tfc.mul(yTrue, logPred)), -1);
  });
}

/**
 * Loss or metric function: Cosine proximity.
 *
 * Mathematically, cosine proximity is defined as:
 *   `-sum(l2Normalize(yTrue) * l2Normalize(yPred))`,
 * wherein `l2Normalize()` normalizes the L2 norm of the input to 1 and `*`
 * represents element-wise multiplication.
 *
 * ```js
 * const yTrue = tf.tensor2d([[1, 0], [1, 0]]);
 * const yPred = tf.tensor2d([[1 / Math.sqrt(2), 1 / Math.sqrt(2)], [0, 1]]);
 * const proximity = tf.metrics.cosineProximity(yTrue, yPred);
 * proximity.print();
 * ```
 *
 * @param yTrue Truth Tensor.
 * @param yPred Prediction Tensor.
 * @return Cosine proximity Tensor.
 */
export function cosineProximity(yTrue: Tensor, yPred: Tensor): Tensor {
  return tidy(() => {
    const trueNormalized = K.l2Normalize(yTrue, -1);
    const predNormalized = K.l2Normalize(yPred, -1);
    const trueXPred = tfc.mul(trueNormalized, predNormalized);
    return tfc.neg(tfc.sum(trueXPred, -1));
  });
}

export const mse = meanSquaredError;
export const MSE = meanSquaredError;
export const mae = meanAbsoluteError;
export const MAE = meanAbsoluteError;
export const mape = meanAbsolutePercentageError;
export const MAPE = meanAbsolutePercentageError;
export const msle = meanSquaredLogarithmicError;
export const MSLE = meanSquaredLogarithmicError;
export const kld = kullbackLeiblerDivergence;
export const KLD = kullbackLeiblerDivergence;
export const cosine = cosineProximity;

// TODO(michaelterry): Add deserialize() function.

// Porting note: This diverges from the PyKeras implementation and may need to
// change based on (de)serialization requirements.
export function get(identifierOrFn: string|LossOrMetricFn): LossOrMetricFn {
  const lossesMap: {[functionName: string]: LossOrMetricFn} = {
    meanSquaredError,
    meanAbsoluteError,
    meanAbsolutePercentageError,
    meanSquaredLogarithmicError,
    squaredHinge,
    hinge,
    categoricalHinge,
    logcosh,
    categoricalCrossentropy,
    sparseCategoricalCrossentropy,
    binaryCrossentropy,
    kullbackLeiblerDivergence,
    poisson,
    cosineProximity
  };
  if (typeof identifierOrFn === 'string') {
    if (identifierOrFn in lossesMap) {
      return lossesMap[identifierOrFn];
    }
    throw new ValueError(`Unknown loss ${identifierOrFn}`);
  } else {
    return identifierOrFn;
  }
}
