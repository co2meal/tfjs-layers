"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tfjs_core_1 = require("@tensorflow/tfjs-core");
var tfl = require("./index");
var metrics_1 = require("./metrics");
var test_utils_1 = require("./utils/test_utils");
test_utils_1.describeMathCPUAndGPU('binaryAccuracy', function () {
    it('1D exact', function () {
        var x = tfjs_core_1.tensor1d([1, 1, 1, 1, 0, 0, 0, 0]);
        var y = tfjs_core_1.tensor1d([1, 0, 1, 0, 0, 1, 0, 1]);
        var accuracy = tfl.metrics.binaryAccuracy(x, y);
        test_utils_1.expectTensorsClose(accuracy, tfjs_core_1.scalar(0.5));
    });
    it('2D thresholded', function () {
        var x = tfjs_core_1.tensor1d([1, 1, 1, 1, 0, 0, 0, 0]);
        var y = tfjs_core_1.tensor1d([0.2, 0.4, 0.6, 0.8, 0.2, 0.3, 0.4, 0.7]);
        var accuracy = tfl.metrics.binaryAccuracy(x, y);
        test_utils_1.expectTensorsClose(accuracy, tfjs_core_1.scalar(5 / 8));
    });
    it('2D exact', function () {
        var x = tfjs_core_1.tensor2d([[1, 1, 1, 1], [0, 0, 0, 0]], [2, 4]);
        var y = tfjs_core_1.tensor2d([[1, 0, 1, 0], [0, 0, 0, 1]], [2, 4]);
        var accuracy = tfl.metrics.binaryAccuracy(x, y);
        test_utils_1.expectTensorsClose(accuracy, tfjs_core_1.tensor1d([0.5, 0.75]));
    });
    it('2D thresholded', function () {
        var x = tfjs_core_1.tensor2d([[1, 1], [1, 1], [0, 0], [0, 0]], [4, 2]);
        var y = tfjs_core_1.tensor2d([[0.2, 0.4], [0.6, 0.8], [0.2, 0.3], [0.4, 0.7]], [4, 2]);
        var accuracy = tfl.metrics.binaryAccuracy(x, y);
        test_utils_1.expectTensorsClose(accuracy, tfjs_core_1.tensor1d([0, 1, 1, 0.5]));
    });
});
test_utils_1.describeMathCPUAndGPU('binaryCrossentropy', function () {
    it('2D single-value yTrue', function () {
        var x = tfjs_core_1.tensor2d([[0], [0], [0], [1], [1], [1]]);
        var y = tfjs_core_1.tensor2d([[0], [0.5], [1], [0], [0.5], [1]]);
        var accuracy = tfl.metrics.binaryCrossentropy(x, y);
        test_utils_1.expectTensorsClose(accuracy, tfjs_core_1.tensor1d([
            1.00000015e-07, 6.93147182e-01, 1.59423847e+01,
            1.61180954e+01, 6.93147182e-01, 1.19209332e-07
        ]));
    });
    it('2D one-hot binary yTrue', function () {
        var x = tfjs_core_1.tensor2d([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1], [0, 1]]);
        var y = tfjs_core_1.tensor2d([[1, 0], [0.5, 0.5], [0, 1], [1, 0], [0.5, 0.5], [0, 1]]);
        var accuracy = tfl.metrics.binaryCrossentropy(x, y);
        test_utils_1.expectTensorsClose(accuracy, tfjs_core_1.tensor1d([
            1.0960467e-07, 6.9314718e-01, 1.6030239e+01,
            1.6030239e+01, 6.9314718e-01, 1.0960467e-07
        ]));
    });
});
test_utils_1.describeMathCPUAndGPU('categoricalAccuracy', function () {
    it('1D', function () {
        var x = tfjs_core_1.tensor1d([0, 0, 0, 1]);
        var y = tfjs_core_1.tensor1d([0.1, 0.8, 0.05, 0.05]);
        var accuracy = tfl.metrics.categoricalAccuracy(x, y);
        expect(accuracy.dtype).toEqual('float32');
        expect(accuracy.shape).toEqual([]);
        expect(Array.from(accuracy.dataSync())).toEqual([0]);
    });
    it('2D', function () {
        var x = tfjs_core_1.tensor2d([[0, 0, 0, 1], [0, 0, 0, 1]], [2, 4]);
        var y = tfjs_core_1.tensor2d([[0.1, 0.8, 0.05, 0.05], [0.1, 0.05, 0.05, 0.8]], [2, 4]);
        var accuracy = tfl.metrics.categoricalAccuracy(x, y);
        expect(accuracy.dtype).toEqual('float32');
        expect(accuracy.shape).toEqual([2]);
        expect(Array.from(accuracy.dataSync())).toEqual([0, 1]);
    });
});
test_utils_1.describeMathCPUAndGPU('categoricalCrossentropy metric', function () {
    it('1D', function () {
        var x = tfjs_core_1.tensor1d([0, 0, 0, 1]);
        var y = tfjs_core_1.tensor1d([0.1, 0.8, 0.05, 0.05]);
        var accuracy = tfl.metrics.categoricalCrossentropy(x, y);
        expect(accuracy.dtype).toEqual('float32');
        test_utils_1.expectTensorsClose(accuracy, tfjs_core_1.scalar(2.995732));
    });
    it('2D', function () {
        var x = tfjs_core_1.tensor2d([[0, 0, 0, 1], [0, 0, 0, 1]]);
        var y = tfjs_core_1.tensor2d([[0.1, 0.8, 0.05, 0.05], [0.1, 0.05, 0.05, 0.8]]);
        var accuracy = tfl.metrics.categoricalCrossentropy(x, y);
        expect(accuracy.dtype).toEqual('float32');
        test_utils_1.expectTensorsClose(accuracy, tfjs_core_1.tensor1d([2.995732, 0.22314353]));
    });
});
describe('metrics.get', function () {
    it('valid name, not alias', function () {
        expect(metrics_1.get('binaryAccuracy') === metrics_1.get('categoricalAccuracy')).toEqual(false);
    });
    it('valid name, alias', function () {
        expect(metrics_1.get('mse') === metrics_1.get('MSE')).toEqual(true);
    });
    it('invalid name', function () {
        expect(function () { return metrics_1.get('InvalidMetricName'); }).toThrowError(/Unknown metric/);
    });
    it('LossOrMetricFn input', function () {
        expect(metrics_1.get(metrics_1.binaryAccuracy)).toEqual(metrics_1.binaryAccuracy);
        expect(metrics_1.get(metrics_1.categoricalAccuracy)).toEqual(metrics_1.categoricalAccuracy);
    });
});
//# sourceMappingURL=metrics_test.js.map