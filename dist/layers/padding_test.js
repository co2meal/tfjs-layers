"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tfc = require("@tensorflow/tfjs-core");
var tfjs_core_1 = require("@tensorflow/tfjs-core");
var topology_1 = require("../engine/topology");
var tfl = require("../index");
var serialization_utils_1 = require("../utils/serialization_utils");
var test_utils_1 = require("../utils/test_utils");
var padding_1 = require("./padding");
test_utils_1.describeMathCPUAndGPU('temporalPadding', function () {
    it('default padding 1-1', function () {
        var x = tfjs_core_1.tensor3d([[[1, 2], [3, 4]], [[-1, -2], [-3, -4]]]);
        var y = padding_1.temporalPadding(x);
        test_utils_1.expectTensorsClose(y, tfjs_core_1.tensor3d([
            [[0, 0], [1, 2], [3, 4], [0, 0]],
            [[0, 0], [-1, -2], [-3, -4], [0, 0]]
        ]));
    });
    it('custom padding 2-2', function () {
        var x = tfjs_core_1.tensor3d([[[1, 2], [3, 4]], [[-1, -2], [-3, -4]]]);
        var y = padding_1.temporalPadding(x, [2, 2]);
        test_utils_1.expectTensorsClose(y, tfjs_core_1.tensor3d([
            [[0, 0], [0, 0], [1, 2], [3, 4], [0, 0], [0, 0]],
            [[0, 0], [0, 0], [-1, -2], [-3, -4], [0, 0], [0, 0]]
        ]));
    });
});
test_utils_1.describeMathCPUAndGPU('spatial2dPadding', function () {
    it('default padding 1-1-1-1', function () {
        var x = tfc.ones([2, 3, 4, 3]);
        var y = padding_1.spatial2dPadding(x);
        expect(y.shape).toEqual([2, 5, 6, 3]);
        test_utils_1.expectTensorsClose(tfc.slice(y, [0, 1, 1, 0], [2, 3, 4, 3]), x);
        test_utils_1.expectTensorsClose(tfc.slice(y, [0, 0, 0, 0], [2, 1, 6, 3]), tfc.zeros([2, 1, 6, 3]));
        test_utils_1.expectTensorsClose(tfc.slice(y, [0, 4, 0, 0], [2, 1, 6, 3]), tfc.zeros([2, 1, 6, 3]));
        test_utils_1.expectTensorsClose(tfc.slice(y, [0, 0, 0, 0], [2, 5, 1, 3]), tfc.zeros([2, 5, 1, 3]));
        test_utils_1.expectTensorsClose(tfc.slice(y, [0, 0, 5, 0], [2, 5, 1, 3]), tfc.zeros([2, 5, 1, 3]));
    });
    it('custom padding 2-2-3-0', function () {
        var x = tfc.ones([2, 3, 4, 3]);
        var y = padding_1.spatial2dPadding(x, [[2, 2], [3, 0]]);
        expect(y.shape).toEqual([2, 7, 7, 3]);
        test_utils_1.expectTensorsClose(tfc.slice(y, [0, 2, 3, 0], [2, 3, 4, 3]), x);
        test_utils_1.expectTensorsClose(tfc.slice(y, [0, 0, 0, 0], [2, 2, 7, 3]), tfc.zeros([2, 2, 7, 3]));
        test_utils_1.expectTensorsClose(tfc.slice(y, [0, 5, 0, 0], [2, 2, 7, 3]), tfc.zeros([2, 2, 7, 3]));
        test_utils_1.expectTensorsClose(tfc.slice(y, [0, 0, 0, 0], [2, 7, 3, 3]), tfc.zeros([2, 7, 3, 3]));
    });
});
test_utils_1.describeMathCPU('ZeroPadding2D: Symbolic', function () {
    var dataFormats = [undefined, 'channelsFirst', 'channelsLast'];
    var _loop_1 = function (dataFormat) {
        it('Default padding 1-1-1-1, dataFormat=' + dataFormat, function () {
            var x = new topology_1.SymbolicTensor('float32', [1, 2, 3, 4], null, [], null);
            var layer = tfl.layers.zeroPadding2d();
            var y = layer.apply(x);
            expect(y.dtype).toEqual('float32');
            if (dataFormat === 'channelsFirst') {
                expect(y.shape).toEqual([1, 2, 5, 6]);
            }
            else {
                expect(y.shape).toEqual([1, 4, 5, 4]);
            }
        });
        it('All symmetric padding 2, dataFormat=' + dataFormat, function () {
            var x = new topology_1.SymbolicTensor('float32', [1, 2, 3, 4], null, [], null);
            var layer = tfl.layers.zeroPadding2d({ padding: 2 });
            var y = layer.apply(x);
            expect(y.dtype).toEqual('float32');
            if (dataFormat === 'channelsFirst') {
                expect(y.shape).toEqual([1, 6, 7, 8]);
            }
            else {
                expect(y.shape).toEqual([1, 6, 7, 4]);
            }
        });
        it('Symmetric padding 2-3, dataFormat=' + dataFormat, function () {
            var x = new topology_1.SymbolicTensor('float32', [1, 2, 3, 4], null, [], null);
            var layer = tfl.layers.zeroPadding2d({ padding: [2, 3] });
            var y = layer.apply(x);
            expect(y.dtype).toEqual('float32');
            if (dataFormat === 'channelsFirst') {
                expect(y.shape).toEqual([1, 2, 7, 10]);
            }
            else {
                expect(y.shape).toEqual([1, 6, 9, 4]);
            }
        });
        it('Asymmetric padding 2-3-4-5, dataFormat=' + dataFormat, function () {
            var x = new topology_1.SymbolicTensor('float32', [1, 2, 3, 4], null, [], null);
            var layer = tfl.layers.zeroPadding2d({ padding: [[2, 3], [4, 5]] });
            var y = layer.apply(x);
            expect(y.dtype).toEqual('float32');
            if (dataFormat === 'channelsFirst') {
                expect(y.shape).toEqual([1, 2, 7, 13]);
            }
            else {
                expect(y.shape).toEqual([1, 7, 12, 4]);
            }
        });
    };
    for (var dataFormat in dataFormats) {
        _loop_1(dataFormat);
    }
    it('Incorrect array length leads to error', function () {
        expect(function () { return tfl.layers.zeroPadding2d({ padding: [2, 3, 4] }); })
            .toThrowError(/length-2 array/);
    });
    it('Incorrect height array length leads to error', function () {
        expect(function () { return tfl.layers.zeroPadding2d({ padding: [[2, 3, 4], [5, 6]] }); })
            .toThrowError(/height.*length-2 array/);
    });
    it('Incorrect height array length leads to error', function () {
        expect(function () { return tfl.layers.zeroPadding2d({ padding: [[1, 1], [2, 3, 4]] }); })
            .toThrowError(/width.*length-2 array/);
    });
    it('Serialization round trip', function () {
        var layer = tfl.layers.zeroPadding2d({ padding: [2, 4] });
        var pythonicConfig = serialization_utils_1.convertTsToPythonic(layer.getConfig());
        var tsConfig = serialization_utils_1.convertPythonicToTs(pythonicConfig);
        var layerPrime = tfl.layers.zeroPadding2d(tsConfig);
        expect(layerPrime.padding).toEqual(layer.padding);
        expect(layerPrime.dataFormat).toEqual(layer.dataFormat);
    });
});
test_utils_1.describeMathCPUAndGPU('ZeroPadding2D: Tensor', function () {
    it('Default padding 1-1-1-1, channelsLast', function () {
        var x = tfc.ones([2, 2, 2, 3]);
        var layer = tfl.layers.zeroPadding2d();
        var y = layer.apply(x);
        expect(y.shape).toEqual([2, 4, 4, 3]);
        test_utils_1.expectTensorsClose(tfc.slice(y, [0, 1, 1, 0], [2, 2, 2, 3]), x);
        test_utils_1.expectTensorsClose(tfc.slice(y, [0, 0, 0, 0], [2, 1, 4, 3]), tfc.zeros([2, 1, 4, 3]));
        test_utils_1.expectTensorsClose(tfc.slice(y, [0, 3, 0, 0], [2, 1, 4, 3]), tfc.zeros([2, 1, 4, 3]));
        test_utils_1.expectTensorsClose(tfc.slice(y, [0, 0, 0, 0], [2, 4, 1, 3]), tfc.zeros([2, 4, 1, 3]));
        test_utils_1.expectTensorsClose(tfc.slice(y, [0, 0, 3, 0], [2, 4, 1, 3]), tfc.zeros([2, 4, 1, 3]));
    });
    it('Default padding 1-1-1-1, channelFirst', function () {
        var x = tfc.ones([2, 3, 2, 2]);
        var layer = tfl.layers.zeroPadding2d({ dataFormat: 'channelsFirst' });
        var y = layer.apply(x);
        expect(y.shape).toEqual([2, 3, 4, 4]);
        test_utils_1.expectTensorsClose(tfc.slice(y, [0, 0, 1, 1], [2, 3, 2, 2]), x);
        test_utils_1.expectTensorsClose(tfc.slice(y, [0, 0, 0, 0], [2, 3, 1, 4]), tfc.zeros([2, 3, 1, 4]));
        test_utils_1.expectTensorsClose(tfc.slice(y, [0, 0, 3, 0], [2, 3, 1, 4]), tfc.zeros([2, 3, 1, 4]));
        test_utils_1.expectTensorsClose(tfc.slice(y, [0, 0, 0, 0], [2, 3, 4, 1]), tfc.zeros([2, 3, 4, 1]));
        test_utils_1.expectTensorsClose(tfc.slice(y, [0, 0, 0, 3], [2, 3, 4, 1]), tfc.zeros([2, 3, 4, 1]));
    });
    it('Symmetric padding 2-2, channelsLast', function () {
        var x = tfc.ones([2, 2, 2, 3]);
        var layer = tfl.layers.zeroPadding2d({ padding: [2, 2] });
        var y = layer.apply(x);
        expect(y.shape).toEqual([2, 6, 6, 3]);
        test_utils_1.expectTensorsClose(tfc.slice(y, [0, 2, 2, 0], [2, 2, 2, 3]), x);
        test_utils_1.expectTensorsClose(tfc.slice(y, [0, 0, 0, 0], [2, 2, 6, 3]), tfc.zeros([2, 2, 6, 3]));
        test_utils_1.expectTensorsClose(tfc.slice(y, [0, 4, 0, 0], [2, 2, 6, 3]), tfc.zeros([2, 2, 6, 3]));
        test_utils_1.expectTensorsClose(tfc.slice(y, [0, 0, 0, 0], [2, 6, 2, 3]), tfc.zeros([2, 6, 2, 3]));
        test_utils_1.expectTensorsClose(tfc.slice(y, [0, 0, 4, 0], [2, 6, 2, 3]), tfc.zeros([2, 6, 2, 3]));
    });
    it('Asymmetric padding 2-1-2-1, channelsLast', function () {
        var x = tfc.ones([2, 2, 2, 3]);
        var layer = tfl.layers.zeroPadding2d({ padding: [[2, 1], [2, 1]] });
        var y = layer.apply(x);
        expect(y.shape).toEqual([2, 5, 5, 3]);
        test_utils_1.expectTensorsClose(tfc.slice(y, [0, 2, 2, 0], [2, 2, 2, 3]), x);
        test_utils_1.expectTensorsClose(tfc.slice(y, [0, 0, 0, 0], [2, 2, 5, 3]), tfc.zeros([2, 2, 5, 3]));
        test_utils_1.expectTensorsClose(tfc.slice(y, [0, 4, 0, 0], [2, 1, 5, 3]), tfc.zeros([2, 1, 5, 3]));
        test_utils_1.expectTensorsClose(tfc.slice(y, [0, 0, 0, 0], [2, 5, 2, 3]), tfc.zeros([2, 5, 2, 3]));
        test_utils_1.expectTensorsClose(tfc.slice(y, [0, 0, 4, 0], [2, 5, 1, 3]), tfc.zeros([2, 5, 1, 3]));
    });
});
//# sourceMappingURL=padding_test.js.map