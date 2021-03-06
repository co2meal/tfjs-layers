import * as tfc from '@tensorflow/tfjs-core';
import { io, ModelPredictConfig, Optimizer, Scalar, Tensor, Tensor1D } from '@tensorflow/tfjs-core';
import { BaseCallback, CustomCallbackConfig, History, YieldEveryOptions } from '../base_callbacks';
import { LossOrMetricFn, NamedTensorMap, Shape } from '../types';
import { Container, ContainerConfig } from './container';
export declare function isDataTensor(x: Tensor | Tensor[] | {
    [inputName: string]: Tensor;
} | {
    [inputName: string]: Tensor[];
}): boolean;
export declare function isDataArray(x: Tensor | Tensor[] | {
    [inputName: string]: Tensor;
}): boolean;
export declare function isDataDict(x: Tensor | Tensor[] | {
    [inputName: string]: Tensor;
}): boolean;
export declare function standardizeInputData(data: Tensor | Tensor[] | {
    [inputName: string]: Tensor;
}, names: string[], shapes?: Shape[], checkBatchAxis?: boolean, exceptionPrefix?: string): Tensor[];
export declare function checkArrayLengths(inputs: Tensor[], targets: Tensor[], weights?: Tensor[]): void;
export declare function makeBatches(size: number, batchSize: number): Array<[number, number]>;
export declare function sliceArraysByIndices(arrays: Tensor | Tensor[], indices: Tensor1D): Tensor | Tensor[];
export declare enum ModelLoggingVerbosity {
    SILENT = 0,
    VERBOSE = 1,
}
export interface ModelEvaluateConfig {
    batchSize?: number;
    verbose?: ModelLoggingVerbosity;
    sampleWeight?: Tensor;
    steps?: number;
}
export interface ModelFitConfig {
    batchSize?: number;
    epochs?: number;
    verbose?: ModelLoggingVerbosity;
    callbacks?: BaseCallback[] | CustomCallbackConfig | CustomCallbackConfig[];
    validationSplit?: number;
    validationData?: [Tensor | Tensor[], Tensor | Tensor[]] | [Tensor | Tensor[], Tensor | Tensor[], Tensor | Tensor[]];
    shuffle?: boolean;
    classWeight?: {
        [classIndex: string]: number;
    };
    sampleWeight?: Tensor;
    initialEpoch?: number;
    stepsPerEpoch?: number;
    validationSteps?: number;
    yieldEvery?: YieldEveryOptions;
}
export interface ModelCompileConfig {
    optimizer: string | Optimizer;
    loss: string | string[] | {
        [outputName: string]: string;
    } | LossOrMetricFn | LossOrMetricFn[] | {
        [outputName: string]: LossOrMetricFn;
    };
    metrics?: string[] | {
        [outputName: string]: string;
    };
}
export declare class Model extends Container implements tfc.InferenceModel {
    static className: string;
    optimizer: Optimizer;
    loss: string | string[] | {
        [outputName: string]: string;
    } | LossOrMetricFn | LossOrMetricFn[] | {
        [outputName: string]: LossOrMetricFn;
    };
    lossFunctions: LossOrMetricFn[];
    private feedOutputShapes;
    private feedLossFns;
    private collectedTrainableWeights;
    private testFunction;
    history: History;
    private stopTraining_;
    private isTraining;
    metrics: string[] | {
        [outputName: string]: string;
    };
    metricsNames: string[];
    metricsTensors: Array<[LossOrMetricFn, number]>;
    constructor(config: ContainerConfig);
    summary(lineLength?: number, positions?: number[], printFn?: (message?: any, ...optionalParams: any[]) => void): void;
    compile(config: ModelCompileConfig): void;
    private checkTrainableWeightsConsistency();
    evaluate(x: Tensor | Tensor[], y: Tensor | Tensor[], config?: ModelEvaluateConfig): Scalar | Scalar[];
    private checkNumSamples(ins, batchSize?, steps?, stepsName?);
    execute(inputs: Tensor | Tensor[] | NamedTensorMap, outputs: string | string[]): Tensor | Tensor[];
    private retrieveSymbolicTensors(symbolicTensorNames);
    private predictLoop(ins, batchSize?, verbose?);
    predict(x: Tensor | Tensor[], config?: ModelPredictConfig): Tensor | Tensor[];
    predictOnBatch(x: Tensor): Tensor | Tensor[];
    protected standardizeUserData(x: Tensor | Tensor[] | {
        [inputName: string]: Tensor;
    }, y: Tensor | Tensor[] | {
        [inputName: string]: Tensor;
    }, checkBatchAxis?: boolean, batchSize?: number): [Tensor[], Tensor[], Tensor[]];
    private fitLoop(f, ins, outLabels?, batchSize?, epochs?, verbose?, callbacks?, valF?, valIns?, shuffle?, callbackMetrics?, initialEpoch?, stepsPerEpoch?, validationSteps?, yieldEvery?);
    private testLoop(f, ins, batchSize?, verbose?, steps?);
    private getDedupedMetricsNames();
    private makeTestFunction();
    fit(x: Tensor | Tensor[] | {
        [inputName: string]: Tensor;
    }, y: Tensor | Tensor[] | {
        [inputName: string]: Tensor;
    }, config?: ModelFitConfig): Promise<History>;
    protected getNamedWeights(config?: io.SaveConfig): NamedTensorMap;
    stopTraining: boolean;
    save(handlerOrURL: io.IOHandler | string, config?: io.SaveConfig): Promise<io.SaveResult>;
}
