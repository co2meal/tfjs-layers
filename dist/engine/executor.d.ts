import { Tensor } from '@tensorflow/tfjs-core';
import { Kwargs } from '../types';
import { SymbolicTensor } from './topology';
export interface Feed {
    key: SymbolicTensor;
    value: Tensor;
}
export declare class FeedDict {
    private id2Value;
    constructor(feeds?: Feed[] | FeedDict);
    add(key: SymbolicTensor, value: Tensor): FeedDict;
    addFeed(feed: Feed): void;
    hasKey(key: SymbolicTensor): boolean;
    getValue(key: SymbolicTensor): Tensor;
}
export declare function execute(fetches: SymbolicTensor | SymbolicTensor[], feedDict: FeedDict, kwargs?: Kwargs): Tensor | Tensor[] | [Tensor | Tensor[]];
