import { serialization, Tensor } from '@tensorflow/tfjs-core';
import { Layer, LayerConfig, SymbolicTensor } from '../engine/topology';
import { Kwargs, Shape } from '../types';
export declare abstract class Merge extends Layer {
    protected reshapeRequired: boolean;
    constructor(config?: LayerConfig);
    protected mergeFunction(inputs: Tensor[]): Tensor;
    private computeElementwiseOpOutputShape(shape1, shape2);
    build(inputShape: Shape | Shape[]): void;
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    computeOutputShape(inputShape: Shape | Shape[]): Shape | Shape[];
}
export declare class Add extends Merge {
    static className: string;
    constructor(config?: LayerConfig);
    protected mergeFunction(inputs: Tensor[]): Tensor;
}
export declare function add(config?: SymbolicTensor[] | Tensor[] | LayerConfig): Layer | SymbolicTensor | Tensor;
export declare class Multiply extends Merge {
    static className: string;
    constructor(config?: LayerConfig);
    protected mergeFunction(inputs: Tensor[]): Tensor;
}
export declare function multiply(config?: SymbolicTensor[] | Tensor[] | LayerConfig): Layer | SymbolicTensor | Tensor;
export declare class Average extends Merge {
    static className: string;
    constructor(config?: LayerConfig);
    protected mergeFunction(inputs: Tensor[]): Tensor;
}
export declare function average(config?: SymbolicTensor[] | Tensor[] | LayerConfig): Layer | SymbolicTensor | Tensor;
export declare class Maximum extends Merge {
    static className: string;
    constructor(config?: LayerConfig);
    protected mergeFunction(inputs: Tensor[]): Tensor;
}
export declare function maximum(config?: SymbolicTensor[] | Tensor[] | LayerConfig): Layer | SymbolicTensor | Tensor;
export declare class Minimum extends Merge {
    static className: string;
    constructor(config?: LayerConfig);
    protected mergeFunction(inputs: Tensor[]): Tensor;
}
export declare function minimum(config?: SymbolicTensor[] | Tensor[] | LayerConfig): Layer | SymbolicTensor | Tensor;
export interface ConcatenateLayerConfig extends LayerConfig {
    axis?: number;
}
export declare class Concatenate extends Merge {
    static className: string;
    readonly DEFAULT_AXIS: number;
    private readonly axis;
    constructor(config?: ConcatenateLayerConfig);
    build(inputShape: Shape | Shape[]): void;
    protected mergeFunction(inputs: Tensor[]): Tensor;
    computeOutputShape(inputShape: Shape | Shape[]): Shape | Shape[];
    getConfig(): serialization.ConfigDict;
}
export declare function concatenate(config?: SymbolicTensor[] | Tensor[] | ConcatenateLayerConfig): Layer | SymbolicTensor | Tensor;
