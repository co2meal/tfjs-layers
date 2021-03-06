import { serialization, Tensor } from '@tensorflow/tfjs-core';
import { Activation } from '../activations';
import { DataFormat, PaddingMode } from '../common';
import { Constraint, ConstraintIdentifier } from '../constraints';
import { InputSpec, Layer, LayerConfig } from '../engine/topology';
import { Initializer, InitializerIdentifier } from '../initializers';
import { Regularizer, RegularizerIdentifier } from '../regularizers';
import { Kwargs, Shape } from '../types';
import { LayerVariable } from '../variables';
export declare function preprocessConv2DInput(x: Tensor, dataFormat: DataFormat): Tensor;
export declare function conv1dWithBias(x: Tensor, kernel: Tensor, bias: Tensor, strides?: number, padding?: string, dataFormat?: DataFormat, dilationRate?: number): Tensor;
export declare function conv1d(x: Tensor, kernel: Tensor, strides?: number, padding?: string, dataFormat?: DataFormat, dilationRate?: number): Tensor;
export declare function conv2d(x: Tensor, kernel: Tensor, strides?: number[], padding?: string, dataFormat?: DataFormat, dilationRate?: [number, number]): Tensor;
export declare function conv2dWithBias(x: Tensor, kernel: Tensor, bias: Tensor, strides?: number[], padding?: string, dataFormat?: DataFormat, dilationRate?: [number, number]): Tensor;
export interface BaseConvLayerConfig extends LayerConfig {
    kernelSize: number | number[];
    strides?: number | number[];
    padding?: PaddingMode;
    dataFormat?: DataFormat;
    dilationRate?: number | [number] | [number, number];
    activation?: string;
    useBias?: boolean;
    kernelInitializer?: InitializerIdentifier | Initializer;
    biasInitializer?: InitializerIdentifier | Initializer;
    kernelConstraint?: ConstraintIdentifier | Constraint;
    biasConstraint?: ConstraintIdentifier | Constraint;
    kernelRegularizer?: RegularizerIdentifier | Regularizer;
    biasRegularizer?: RegularizerIdentifier | Regularizer;
    activityRegularizer?: RegularizerIdentifier | Regularizer;
}
export interface ConvLayerConfig extends BaseConvLayerConfig {
    filters: number;
}
export declare abstract class BaseConv extends Layer {
    protected readonly rank: number;
    protected readonly kernelSize: number[];
    protected readonly strides: number[];
    protected readonly padding: PaddingMode;
    protected readonly dataFormat: DataFormat;
    protected readonly activation: Activation;
    protected readonly useBias: boolean;
    protected readonly dilationRate: number[];
    protected readonly biasInitializer?: Initializer;
    protected readonly biasConstraint?: Constraint;
    protected readonly biasRegularizer?: Regularizer;
    protected bias: LayerVariable;
    readonly DEFAULT_KERNEL_INITIALIZER: InitializerIdentifier;
    readonly DEFAULT_BIAS_INITIALIZER: InitializerIdentifier;
    constructor(rank: number, config: BaseConvLayerConfig);
    protected static verifyConfig(config: BaseConvLayerConfig): void;
    getConfig(): serialization.ConfigDict;
}
export declare abstract class Conv extends BaseConv {
    protected readonly filters: number;
    protected kernel: LayerVariable;
    protected readonly kernelInitializer?: Initializer;
    protected readonly kernelConstraint?: Constraint;
    protected readonly kernelRegularizer?: Regularizer;
    constructor(rank: number, config: ConvLayerConfig);
    build(inputShape: Shape | Shape[]): void;
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    computeOutputShape(inputShape: Shape | Shape[]): Shape | Shape[];
    getConfig(): serialization.ConfigDict;
    protected static verifyConfig(config: ConvLayerConfig): void;
}
export declare class Conv2D extends Conv {
    static className: string;
    constructor(config: ConvLayerConfig);
    getConfig(): serialization.ConfigDict;
    protected static verifyConfig(config: ConvLayerConfig): void;
}
export declare class Conv2DTranspose extends Conv2D {
    static className: string;
    inputSpec: InputSpec[];
    constructor(config: ConvLayerConfig);
    build(inputShape: Shape | Shape[]): void;
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    computeOutputShape(inputShape: Shape | Shape[]): Shape | Shape[];
    getConfig(): serialization.ConfigDict;
}
export interface SeparableConvLayerConfig extends ConvLayerConfig {
    depthMultiplier?: number;
    depthwiseInitializer?: InitializerIdentifier | Initializer;
    pointwiseInitializer?: InitializerIdentifier | Initializer;
    depthwiseRegularizer?: RegularizerIdentifier | Regularizer;
    pointwiseRegularizer?: RegularizerIdentifier | Regularizer;
    depthwiseConstraint?: ConstraintIdentifier | Constraint;
    pointwiseConstraint?: ConstraintIdentifier | Constraint;
}
export declare class SeparableConv extends Conv {
    static className: string;
    readonly depthMultiplier: number;
    protected readonly depthwiseInitializer?: Initializer;
    protected readonly depthwiseRegularizer?: Regularizer;
    protected readonly depthwiseConstraint?: Constraint;
    protected readonly pointwiseInitializer?: Initializer;
    protected readonly pointwiseRegularizer?: Regularizer;
    protected readonly pointwiseConstraint?: Constraint;
    readonly DEFAULT_DEPTHWISE_INITIALIZER: InitializerIdentifier;
    readonly DEFAULT_POINTWISE_INITIALIZER: InitializerIdentifier;
    protected depthwiseKernel: LayerVariable;
    protected pointwiseKernel: LayerVariable;
    constructor(rank: number, config?: SeparableConvLayerConfig);
    build(inputShape: Shape | Shape[]): void;
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    getConfig(): serialization.ConfigDict;
}
export declare class SeparableConv2D extends SeparableConv {
    static className: string;
    constructor(config?: SeparableConvLayerConfig);
}
export declare class Conv1D extends Conv {
    static className: string;
    constructor(config: ConvLayerConfig);
    getConfig(): serialization.ConfigDict;
    static verifyConfig(config: ConvLayerConfig): void;
}
export interface Cropping2DLayerConfig extends LayerConfig {
    cropping: number | [number, number] | [[number, number], [number, number]];
    dataFormat?: DataFormat;
}
export declare class Cropping2D extends Layer {
    static className: string;
    protected readonly cropping: [[number, number], [number, number]];
    protected readonly dataFormat: DataFormat;
    constructor(config: Cropping2DLayerConfig);
    computeOutputShape(inputShape: Shape): Shape;
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    getConfig(): serialization.ConfigDict;
}
export interface UpSampling2DLayerConfig extends LayerConfig {
    size?: number[];
    dataFormat?: DataFormat;
}
export declare class UpSampling2D extends Layer {
    static className: string;
    protected readonly DEFAULT_SIZE: number[];
    protected readonly size: number[];
    protected readonly dataFormat: DataFormat;
    constructor(config: UpSampling2DLayerConfig);
    computeOutputShape(inputShape: Shape): Shape;
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    getConfig(): serialization.ConfigDict;
}
