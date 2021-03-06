import * as tfc from '@tensorflow/tfjs-core';
import { serialization, Tensor } from '@tensorflow/tfjs-core';
import { Activation, ActivationIdentifier } from '../activations';
import { Constraint, ConstraintIdentifier } from '../constraints';
import { InputSpec, SymbolicTensor } from '../engine/topology';
import { Layer, LayerConfig } from '../engine/topology';
import { Initializer, InitializerIdentifier } from '../initializers';
import { Regularizer, RegularizerIdentifier } from '../regularizers';
import { Kwargs, RnnStepFunction, Shape } from '../types';
import { LayerVariable } from '../variables';
export declare function standardizeArgs(inputs: Tensor | Tensor[] | SymbolicTensor | SymbolicTensor[], initialState: Tensor | Tensor[] | SymbolicTensor | SymbolicTensor[], constants: Tensor | Tensor[] | SymbolicTensor | SymbolicTensor[], numConstants?: number): {
    inputs: Tensor | SymbolicTensor;
    initialState: Tensor[] | SymbolicTensor[];
    constants: Tensor[] | SymbolicTensor[];
};
export declare function rnn(stepFunction: RnnStepFunction, inputs: Tensor, initialStates: Tensor[], goBackwards?: boolean, mask?: Tensor, constants?: Tensor[], unroll?: boolean, inputLength?: number): [Tensor, Tensor, Tensor[]];
export interface BaseRNNLayerConfig extends LayerConfig {
    cell?: RNNCell | RNNCell[];
    returnSequences?: boolean;
    returnState?: boolean;
    goBackwards?: boolean;
    stateful?: boolean;
    unroll?: boolean;
    inputDim?: number;
    inputLength?: number;
}
export interface RNNLayerConfig extends BaseRNNLayerConfig {
    cell: RNNCell | RNNCell[];
}
export declare class RNN extends Layer {
    static className: string;
    readonly cell: RNNCell;
    readonly returnSequences: boolean;
    readonly returnState: boolean;
    readonly goBackwards: boolean;
    readonly unroll: boolean;
    stateSpec: InputSpec[];
    states: Tensor[];
    private numConstants;
    constructor(config: RNNLayerConfig);
    getStates(): Tensor[];
    setStates(states: Tensor[]): void;
    computeOutputShape(inputShape: Shape | Shape[]): Shape | Shape[];
    computeMask(inputs: Tensor | Tensor[], mask?: Tensor | Tensor[]): Tensor;
    build(inputShape: Shape | Shape[]): void;
    resetStates(states?: Tensor | Tensor[]): void;
    apply(inputs: Tensor | Tensor[] | SymbolicTensor | SymbolicTensor[], kwargs?: Kwargs): Tensor | Tensor[] | SymbolicTensor | SymbolicTensor[];
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    getInitialState(inputs: Tensor): Tensor[];
    readonly trainableWeights: LayerVariable[];
    readonly nonTrainableWeights: LayerVariable[];
    getConfig(): serialization.ConfigDict;
}
export declare abstract class RNNCell extends Layer {
    stateSize: number | number[];
    dropoutMask: Tensor | Tensor[];
    recurrentDropoutMask: Tensor | Tensor[];
}
export interface SimpleRNNCellLayerConfig extends LayerConfig {
    units: number;
    activation?: ActivationIdentifier;
    useBias?: boolean;
    kernelInitializer?: InitializerIdentifier | Initializer;
    recurrentInitializer?: InitializerIdentifier | Initializer;
    biasInitializer?: InitializerIdentifier | Initializer;
    kernelRegularizer?: RegularizerIdentifier | Regularizer;
    recurrentRegularizer?: RegularizerIdentifier | Regularizer;
    biasRegularizer?: RegularizerIdentifier | Regularizer;
    kernelConstraint?: ConstraintIdentifier | Constraint;
    recurrentConstraint?: ConstraintIdentifier | Constraint;
    biasConstraint?: ConstraintIdentifier | Constraint;
    dropout?: number;
    recurrentDropout?: number;
}
export declare class SimpleRNNCell extends RNNCell {
    static className: string;
    readonly units: number;
    readonly activation: Activation;
    readonly useBias: boolean;
    readonly kernelInitializer: Initializer;
    readonly recurrentInitializer: Initializer;
    readonly biasInitializer: Initializer;
    readonly kernelConstraint: Constraint;
    readonly recurrentConstraint: Constraint;
    readonly biasConstraint: Constraint;
    readonly kernelRegularizer: Regularizer;
    readonly recurrentRegularizer: Regularizer;
    readonly biasRegularizer: Regularizer;
    readonly dropout: number;
    readonly recurrentDropout: number;
    readonly stateSize: number;
    kernel: LayerVariable;
    recurrentKernel: LayerVariable;
    bias: LayerVariable;
    readonly DEFAULT_ACTIVATION: string;
    readonly DEFAULT_KERNEL_INITIALIZER: string;
    readonly DEFAULT_RECURRENT_INITIALIZER: string;
    readonly DEFAULT_BIAS_INITIALIZER: InitializerIdentifier;
    constructor(config: SimpleRNNCellLayerConfig);
    build(inputShape: Shape | Shape[]): void;
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    getConfig(): serialization.ConfigDict;
}
export interface SimpleRNNLayerConfig extends BaseRNNLayerConfig {
    units: number;
    activation?: ActivationIdentifier;
    useBias?: boolean;
    kernelInitializer?: InitializerIdentifier | Initializer;
    recurrentInitializer?: InitializerIdentifier | Initializer;
    biasInitializer?: InitializerIdentifier | Initializer;
    kernelRegularizer?: RegularizerIdentifier | Regularizer;
    recurrentRegularizer?: RegularizerIdentifier | Regularizer;
    biasRegularizer?: RegularizerIdentifier | Regularizer;
    kernelConstraint?: ConstraintIdentifier | Constraint;
    recurrentConstraint?: ConstraintIdentifier | Constraint;
    biasConstraint?: ConstraintIdentifier | Constraint;
    dropout?: number;
    recurrentDropout?: number;
}
export declare class SimpleRNN extends RNN {
    static className: string;
    constructor(config: SimpleRNNLayerConfig);
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    readonly units: number;
    readonly activation: Activation;
    readonly useBias: boolean;
    readonly kernelInitializer: Initializer;
    readonly recurrentInitializer: Initializer;
    readonly biasInitializer: Initializer;
    readonly kernelRegularizer: Regularizer;
    readonly recurrentRegularizer: Regularizer;
    readonly biasRegularizer: Regularizer;
    readonly kernelConstraint: Constraint;
    readonly recurrentConstraint: Constraint;
    readonly biasConstraint: Constraint;
    readonly dropout: number;
    readonly recurrentDropout: number;
    getConfig(): serialization.ConfigDict;
}
export interface GRUCellLayerConfig extends SimpleRNNCellLayerConfig {
    recurrentActivation?: string;
    implementation?: number;
}
export declare class GRUCell extends RNNCell {
    static className: string;
    readonly units: number;
    readonly activation: Activation;
    readonly recurrentActivation: Activation;
    readonly useBias: boolean;
    readonly kernelInitializer: Initializer;
    readonly recurrentInitializer: Initializer;
    readonly biasInitializer: Initializer;
    readonly kernelRegularizer: Regularizer;
    readonly recurrentRegularizer: Regularizer;
    readonly biasRegularizer: Regularizer;
    readonly kernelConstraint: Constraint;
    readonly recurrentConstraint: Constraint;
    readonly biasConstraint: Constraint;
    readonly dropout: number;
    readonly recurrentDropout: number;
    readonly stateSize: number;
    readonly implementation: number;
    readonly DEFAULT_ACTIVATION: string;
    readonly DEFAULT_RECURRENT_ACTIVATION: string;
    readonly DEFAULT_KERNEL_INITIALIZER: string;
    readonly DEFAULT_RECURRENT_INITIALIZER: string;
    readonly DEFAULT_BIAS_INITIALIZER: InitializerIdentifier;
    kernel: LayerVariable;
    recurrentKernel: LayerVariable;
    bias: LayerVariable;
    constructor(config: GRUCellLayerConfig);
    build(inputShape: Shape | Shape[]): void;
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    getConfig(): serialization.ConfigDict;
}
export interface GRULayerConfig extends SimpleRNNLayerConfig {
    recurrentActivation?: string;
    implementation?: number;
}
export declare class GRU extends RNN {
    static className: string;
    constructor(config: GRULayerConfig);
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    readonly units: number;
    readonly activation: Activation;
    readonly recurrentActivation: Activation;
    readonly useBias: boolean;
    readonly kernelInitializer: Initializer;
    readonly recurrentInitializer: Initializer;
    readonly biasInitializer: Initializer;
    readonly kernelRegularizer: Regularizer;
    readonly recurrentRegularizer: Regularizer;
    readonly biasRegularizer: Regularizer;
    readonly kernelConstraint: Constraint;
    readonly recurrentConstraint: Constraint;
    readonly biasConstraint: Constraint;
    readonly dropout: number;
    readonly recurrentDropout: number;
    readonly implementation: number;
    getConfig(): serialization.ConfigDict;
    static fromConfig<T extends serialization.Serializable>(cls: serialization.SerializableConstructor<T>, config: serialization.ConfigDict): T;
}
export interface LSTMCellLayerConfig extends SimpleRNNCellLayerConfig {
    recurrentActivation?: ActivationIdentifier;
    unitForgetBias?: boolean;
    implementation?: 1 | 2;
}
export declare class LSTMCell extends RNNCell {
    static className: string;
    readonly units: number;
    readonly activation: Activation;
    readonly recurrentActivation: Activation;
    readonly useBias: boolean;
    readonly kernelInitializer: Initializer;
    readonly recurrentInitializer: Initializer;
    readonly biasInitializer: Initializer;
    readonly unitForgetBias: boolean;
    readonly kernelConstraint: Constraint;
    readonly recurrentConstraint: Constraint;
    readonly biasConstraint: Constraint;
    readonly kernelRegularizer: Regularizer;
    readonly recurrentRegularizer: Regularizer;
    readonly biasRegularizer: Regularizer;
    readonly dropout: number;
    readonly recurrentDropout: number;
    readonly stateSize: number[];
    readonly implementation: number;
    readonly DEFAULT_ACTIVATION: string;
    readonly DEFAULT_RECURRENT_ACTIVATION: string;
    readonly DEFAULT_KERNEL_INITIALIZER: string;
    readonly DEFAULT_RECURRENT_INITIALIZER: string;
    readonly DEFAULT_BIAS_INITIALIZER: string;
    kernel: LayerVariable;
    recurrentKernel: LayerVariable;
    bias: LayerVariable;
    constructor(config: LSTMCellLayerConfig);
    build(inputShape: Shape | Shape[]): void;
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    getConfig(): serialization.ConfigDict;
}
export interface LSTMLayerConfig extends SimpleRNNLayerConfig {
    recurrentActivation?: string;
    unitForgetBias?: boolean;
    implementation?: 1 | 2;
}
export declare class LSTM extends RNN {
    static className: string;
    constructor(config: LSTMLayerConfig);
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    readonly units: number;
    readonly activation: Activation;
    readonly recurrentActivation: Activation;
    readonly useBias: boolean;
    readonly kernelInitializer: Initializer;
    readonly recurrentInitializer: Initializer;
    readonly biasInitializer: Initializer;
    readonly unitForgetBias: boolean;
    readonly kernelRegularizer: Regularizer;
    readonly recurrentRegularizer: Regularizer;
    readonly biasRegularizer: Regularizer;
    readonly kernelConstraint: Constraint;
    readonly recurrentConstraint: Constraint;
    readonly biasConstraint: Constraint;
    readonly dropout: number;
    readonly recurrentDropout: number;
    readonly implementation: number;
    getConfig(): serialization.ConfigDict;
    static fromConfig<T extends serialization.Serializable>(cls: serialization.SerializableConstructor<T>, config: serialization.ConfigDict): T;
}
export interface StackedRNNCellsConfig extends LayerConfig {
    cells: RNNCell[];
}
export declare class StackedRNNCells extends RNNCell {
    static className: string;
    protected cells: RNNCell[];
    constructor(config: StackedRNNCellsConfig);
    readonly stateSize: number[];
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    build(inputShape: Shape | Shape[]): void;
    getConfig(): serialization.ConfigDict;
    static fromConfig<T extends serialization.Serializable>(cls: serialization.SerializableConstructor<T>, config: serialization.ConfigDict, customObjects?: tfc.serialization.ConfigDict): T;
    readonly trainableWeights: LayerVariable[];
    readonly nonTrainableWeights: LayerVariable[];
    getWeights(): Tensor[];
    setWeights(weights: Tensor[]): void;
}
