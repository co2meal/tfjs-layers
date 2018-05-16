"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tfl = require("../index");
describe('upsampling', function () {
    var inputs = new tfl.SymbolicTensor('float32', [3, 4, 5, 6], null, [], null);
    it('basic usage', function () {
        var upsamplingLayer = tfl.layers.upsampling({ newSize: [10, 20] });
        var output = upsamplingLayer.apply(inputs);
        expect(output.shape).toEqual([null, 10, 20, 6]);
    });
});
//# sourceMappingURL=upsampling_test.js.map