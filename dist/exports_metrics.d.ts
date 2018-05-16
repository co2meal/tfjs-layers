import { Tensor } from '@tensorflow/tfjs-core';
export declare function binaryAccuracy(yTrue: Tensor, yPred: Tensor): Tensor;
export declare function binaryCrossentropy(yTrue: Tensor, yPred: Tensor): Tensor;
export declare function categoricalAccuracy(yTrue: Tensor, yPred: Tensor): Tensor;
export declare function categoricalCrossentropy(yTrue: Tensor, yPred: Tensor): Tensor;
export declare function cosineProximity(yTrue: Tensor, yPred: Tensor): Tensor;
export declare function meanAbsoluteError(yTrue: Tensor, yPred: Tensor): Tensor;
export declare function meanAbsolutePercentageError(yTrue: Tensor, yPred: Tensor): Tensor;
export declare function MAPE(yTrue: Tensor, yPred: Tensor): Tensor;
export declare function mape(yTrue: Tensor, yPred: Tensor): Tensor;
export declare function meanSquaredError(yTrue: Tensor, yPred: Tensor): Tensor;
export declare function MSE(yTrue: Tensor, yPred: Tensor): Tensor;
export declare function mse(yTrue: Tensor, yPred: Tensor): Tensor;