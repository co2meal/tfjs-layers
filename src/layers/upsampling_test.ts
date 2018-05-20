import * as tfl from '../index';

describe('upsampling', () => {
  const inputs = new tfl.SymbolicTensor('float32', [3, 4, 5, 6], null, [], null);

  it('basic usage', () => {
    const upsamplingLayer = tfl.layers.upsampling({ newSize: [10, 20] });
    const output = upsamplingLayer.apply(inputs) as tfl.SymbolicTensor;
    expect(output.shape).toEqual([null, 10, 20, 6]);
  })
})