import { serialization, Tensor, Rank } from '@tensorflow/tfjs-core';
import { Layer, LayerConfig } from "../engine/topology";
import { Kwargs } from '../types';
export interface UpsamplingLayerConfig extends LayerConfig {
    newSize: number[];
}
export declare class Upsampling extends Layer {
    static className: string;
    private readonly newSize;
    constructor(config: UpsamplingLayerConfig);
    call<T extends Tensor<Rank.R3> | Tensor<Rank.R4>>(inputs: T | T[], kwargs: Kwargs): T;
    computeOutputShape(inputShape: any): any[];
    getConfig(): serialization.ConfigDict;
}
