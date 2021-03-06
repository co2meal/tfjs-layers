import { io, Scalar, serialization, Tensor } from '@tensorflow/tfjs-core';
import { History } from './base_callbacks';
import { Layer } from './engine/topology';
import { Model, ModelCompileConfig, ModelEvaluateConfig, ModelFitConfig } from './engine/training';
import { Kwargs, Shape } from './types';
import { JsonDict } from './types';
export declare function modelFromJSON(modelAndWeightsConfig: ModelAndWeightsConfig, customObjects?: serialization.ConfigDict): Promise<Model>;
export interface ModelAndWeightsConfig {
    modelTopology: JsonDict;
    weightsManifest?: io.WeightsManifestConfig;
    pathPrefix?: string;
}
export interface ModelPredictConfig {
    batchSize?: number;
    verbose?: boolean;
}
export declare function loadModelInternal(pathOrIOHandler: string | io.IOHandler, strict?: boolean): Promise<Model>;
export declare function loadModelFromIOHandler(handler: io.IOHandler, customObjects?: serialization.ConfigDict, strict?: boolean): Promise<Model>;
export interface SequentialConfig {
    layers?: Layer[];
    name?: string;
}
export declare class Sequential extends Model {
    static className: string;
    private model;
    private _updatable;
    constructor(config?: SequentialConfig);
    add(layer: Layer): void;
    pop(): void;
    call(inputs: Tensor | Tensor[], kwargs: Kwargs): Tensor | Tensor[];
    build(inputShape?: Shape | Shape[]): void;
    countParams(): number;
    summary(lineLength?: number, positions?: number[], printFn?: (message?: any, ...optionalParams: any[]) => void): void;
    setWeights(weights: Tensor[]): void;
    updatable: boolean;
    evaluate(x: Tensor | Tensor[], y: Tensor | Tensor[], config?: ModelEvaluateConfig): Scalar | Scalar[];
    predict(x: Tensor | Tensor[], config?: ModelPredictConfig): Tensor | Tensor[];
    predictOnBatch(x: Tensor): Tensor | Tensor[];
    compile(config: ModelCompileConfig): void;
    fit(x: Tensor | Tensor[] | {
        [inputName: string]: Tensor;
    }, y: Tensor | Tensor[] | {
        [inputName: string]: Tensor;
    }, config?: ModelFitConfig): Promise<History>;
    static fromConfig<T extends serialization.Serializable>(cls: serialization.SerializableConstructor<T>, config: serialization.ConfigDict): T;
    stopTraining: boolean;
    getConfig(): any;
}
