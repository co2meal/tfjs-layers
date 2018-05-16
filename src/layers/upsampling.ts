import {serialization, Tensor, image, Rank} from '@tensorflow/tfjs-core';
import {Layer, LayerConfig} from "../engine/topology";
import {getExactlyOneTensor} from '../utils/types_utils';
import {Kwargs} from '../types';

/**
 * TensorFlow.js Layers: Basic Layers.
 */

// tslint:enable:max-line-length

export interface UpsamplingLayerConfig extends LayerConfig {
  newSize: number[];
}


export class Upsampling extends Layer {
  static className = 'Upsampling';
  private readonly newSize: number[];

  constructor(config: UpsamplingLayerConfig) {
    super(config);
    this.newSize = config.newSize;
  }

  call<T extends Tensor<Rank.R3> | Tensor<Rank.R4>>(inputs: T | T[], kwargs: Kwargs): T {
    const [newHeight, newWidth] = this.newSize;
    const input = getExactlyOneTensor(inputs) as T;
    const resized = image.resizeBilinear(input, [newHeight, newWidth],
                                         /* alignCorners= */true)
    return resized;
  }

  computeOutputShape(inputShape: any) {
    return [null, this.newSize[0], this.newSize[1], inputShape[3]]
  }

  getConfig(): serialization.ConfigDict {
    const config = {
      newSize: this.newSize,
    };
    const baseConfig = super.getConfig();
    Object.assign(config, baseConfig);
    return config;
  }
}
serialization.SerializationMap.register(Upsampling);
