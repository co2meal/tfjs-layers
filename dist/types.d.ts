import { DataType, Scalar, Tensor } from '@tensorflow/tfjs-core';
import { Layer } from './engine/topology';
export declare type Shape = number[];
export declare function getNextUniqueTensorId(): number;
export declare class SymbolicTensor {
    readonly dtype: DataType;
    readonly shape: Shape;
    sourceLayer: Layer;
    readonly inputs: SymbolicTensor[];
    readonly callArgs: Kwargs;
    readonly outputTensorIndex: number;
    readonly id: number;
    readonly name?: string;
    readonly originalName?: string;
    readonly rank: number;
    nodeIndex: number;
    tensorIndex: number;
    constructor(dtype: DataType, shape: Shape, sourceLayer: Layer, inputs: SymbolicTensor[], callArgs: Kwargs, name?: string, outputTensorIndex?: number);
}
export declare type LossOrMetricFn = (yTrue: Tensor, yPred: Tensor) => Tensor;
export declare type RegularizerFn = () => Scalar;
export declare type RnnStepFunction = (inputs: Tensor, states: Tensor[]) => [Tensor, Tensor[]];
export declare type NamedTensorMap = {
    [name: string]: Tensor;
};
export declare type JsonValue = boolean | number | string | null | JsonArray | JsonDict;
export interface JsonDict {
    [key: string]: JsonValue;
}
export interface JsonArray extends Array<JsonValue> {
}
export declare type Kwargs = {
    [key: string]: any;
};
