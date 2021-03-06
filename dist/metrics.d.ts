import { Tensor } from '@tensorflow/tfjs-core';
import { categoricalCrossentropy as categoricalCrossentropyLoss, cosineProximity, meanAbsoluteError, meanAbsolutePercentageError, meanSquaredError, sparseCategoricalCrossentropy as sparseCategoricalCrossentropyLoss } from './losses';
import { LossOrMetricFn } from './types';
export declare function binaryAccuracy(yTrue: Tensor, yPred: Tensor): Tensor;
export declare function categoricalAccuracy(yTrue: Tensor, yPred: Tensor): Tensor;
export declare function binaryCrossentropy(yTrue: Tensor, yPred: Tensor): Tensor;
export declare function sparseCategoricalAccuracy(yTrue: Tensor, yPred: Tensor): Tensor;
export declare function topKCategoricalAccuracy(yTrue: Tensor, yPred: Tensor): Tensor;
export declare function sparseTopKCategoricalAccuracy(yTrue: Tensor, yPred: Tensor): Tensor;
export declare const mse: typeof meanSquaredError;
export declare const MSE: typeof meanSquaredError;
export declare const mae: typeof meanAbsoluteError;
export declare const MAE: typeof meanAbsoluteError;
export declare const mape: typeof meanAbsolutePercentageError;
export declare const MAPE: typeof meanAbsolutePercentageError;
export declare const categoricalCrossentropy: typeof categoricalCrossentropyLoss;
export declare const cosine: typeof cosineProximity;
export declare const sparseCategoricalCrossentropy: typeof sparseCategoricalCrossentropyLoss;
export declare function get(identifier: string | LossOrMetricFn): LossOrMetricFn;
