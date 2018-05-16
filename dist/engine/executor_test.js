"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tfjs_core_1 = require("@tensorflow/tfjs-core");
var tfl = require("../index");
var test_utils_1 = require("../utils/test_utils");
var executor_1 = require("./executor");
test_utils_1.describeMathCPU('FeedDict', function () {
    var x = tfl.input({ shape: [], name: 'x', dtype: 'float32' });
    var y = tfl.input({ shape: [], name: 'y', dtype: 'float32' });
    var xValue = tfjs_core_1.tensor1d([42]);
    var yValue = tfjs_core_1.tensor1d([21]);
    it('FeedDict from a single Feed', function () {
        var feedDict = new executor_1.FeedDict([{ key: x, value: xValue }]);
        expect(feedDict.hasKey(x)).toBe(true);
        expect(feedDict.hasKey(y)).toBe(false);
        expect(feedDict.getValue(x)).toEqual(xValue);
        expect(function () { return feedDict.getValue(y); }).toThrowError();
    });
    it('FeedDict from duplicate Feeds throws error', function () {
        var feed = { key: x, value: xValue };
        expect(function () { return new executor_1.FeedDict([feed, feed]); }).toThrowError(/Duplicate key/);
    });
    it('Add key and value', function () {
        var feedDict = new executor_1.FeedDict();
        expect(feedDict.hasKey(x)).toBe(false);
        expect(feedDict.hasKey(y)).toBe(false);
        expect(feedDict.add(x, xValue)).toEqual(feedDict);
        expect(feedDict.hasKey(x)).toBe(true);
        expect(feedDict.hasKey(y)).toBe(false);
        expect(feedDict.add(y, yValue)).toEqual(feedDict);
        expect(feedDict.hasKey(x)).toBe(true);
        expect(feedDict.hasKey(y)).toBe(true);
        expect(feedDict.getValue(x)).toEqual(xValue);
        expect(feedDict.getValue(y)).toEqual(yValue);
    });
    it('Copy constructor', function () {
        var feedDict1 = new executor_1.FeedDict().add(x, xValue);
        var feedDict2 = new executor_1.FeedDict(feedDict1);
        expect(feedDict2.hasKey(x)).toBe(true);
        expect(feedDict2.getValue(x)).toEqual(xValue);
        expect(feedDict2.hasKey(y)).toBe(false);
        feedDict2.add(y, yValue);
        expect(feedDict2.hasKey(y)).toBe(true);
        expect(feedDict2.getValue(y)).toEqual(yValue);
        expect(feedDict1.hasKey(y)).toBe(false);
    });
    it('Add duplicate key and value leads to error', function () {
        var feedDict = new executor_1.FeedDict();
        expect(feedDict.add(x, xValue)).toEqual(feedDict);
        expect(function () { return feedDict.add(x, xValue); }).toThrowError(/Duplicate key/);
    });
    it('Feeding compatible value with undetermined dimension works', function () {
        var s = tfl.input({ shape: [null, 4], name: 's', dtype: 'float32' });
        var sValue = tfjs_core_1.tensor3d([1, 3, 3, 7], [1, 1, 4]);
        var feedDict = new executor_1.FeedDict([{ key: s, value: sValue }]);
        expect(feedDict.getValue(s)).toEqual(sValue);
    });
    it('Feeding incompatible rank leads to error', function () {
        var s = tfl.input({ shape: [null, 4], name: 's', dtype: 'float32' });
        var sValue = tfjs_core_1.tensor2d([1, 3, 3, 7], [1, 4]);
        expect(function () { return new executor_1.FeedDict([{ key: s, value: sValue }]); })
            .toThrowError(/rank of feed .* does not match/);
    });
    it('Feeding incompatible dimension leads to error', function () {
        var s = tfl.input({ shape: [null, 4], name: 's', dtype: 'float32' });
        var sValue = tfjs_core_1.tensor3d([0, 0, 8], [1, 1, 3]);
        expect(function () { return new executor_1.FeedDict([{ key: s, value: sValue }]); })
            .toThrowError(/The 2-th dimension of the feed .* is incompatible/);
    });
});
test_utils_1.describeMathCPUAndGPU('Executor', function () {
    it('Linear Graph Topology', function () {
        var x = tfl.input({ shape: [2], name: 'fooInput', dtype: 'float32' });
        var denseLayer1 = tfl.layers.dense({ units: 5, activation: 'linear', kernelInitializer: 'ones' });
        var y = denseLayer1.apply(x);
        var u = tfl.input({ shape: [2], name: 'footInput', dtype: 'float32' });
        var denseLayer2 = tfl.layers.dense({ units: 5, activation: 'linear', kernelInitializer: 'ones' });
        var denseLayer3 = tfl.layers.dense({ units: 3, activation: 'linear', kernelInitializer: 'ones' });
        var v = denseLayer2.apply(u);
        var w = denseLayer3.apply(v);
        it('Execute Input directly', function () {
            var xValue = tfjs_core_1.ones([2, 2]);
            var feedDict = new executor_1.FeedDict().add(x, xValue);
            test_utils_1.expectTensorsClose(executor_1.execute(x, feedDict), tfjs_core_1.tensor2d([1, 1, 1, 1], [2, 2]));
        });
        it('Input to Dense', function () {
            var xValue = tfjs_core_1.ones([2, 2]);
            var feedDict = new executor_1.FeedDict([{ key: x, value: xValue }]);
            test_utils_1.expectTensorsClose(executor_1.execute(y, feedDict), tfjs_core_1.tensor2d([2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 5]));
        });
        it('Input to Dense1 to Dense2', function () {
            var uValue = tfjs_core_1.ones([2, 2]);
            var feedDict = new executor_1.FeedDict([{ key: u, value: uValue }]);
            test_utils_1.expectTensorsClose(executor_1.execute(w, feedDict), tfjs_core_1.tensor2d([10, 10, 10, 10, 10, 10], [2, 3]));
        });
        it('Feed value to intermediate layers is supported', function () {
            var vValue = tfjs_core_1.ones([3, 5]);
            var feedDict = new executor_1.FeedDict([{ key: v, value: vValue }]);
            test_utils_1.expectTensorsClose(executor_1.execute(w, feedDict), tfjs_core_1.tensor2d([5, 5, 5, 5, 5, 5, 5, 5, 5], [3, 3]));
        });
        it('Calling execute without all Input feeds available leads to error', function () {
            var feedDict = new executor_1.FeedDict();
            expect(function () { return executor_1.execute(y, feedDict); })
                .toThrowError(/Missing a feed value .* from InputLayer/);
        });
    });
    it('Diamond Graph Topology', function () {
        var x = tfl.input({ shape: [2], name: 'fooInput', dtype: 'float32' });
        var denseLayer1 = tfl.layers.dense({
            units: 5,
            activation: 'linear',
            kernelInitializer: 'ones',
            name: 'denseLayer1'
        });
        var y = denseLayer1.apply(x);
        var denseLayer2 = tfl.layers.dense({
            units: 4,
            activation: 'linear',
            kernelInitializer: 'ones',
            name: 'denseLayer2'
        });
        var denseLayer3 = tfl.layers.dense({
            units: 3,
            activation: 'linear',
            kernelInitializer: 'ones',
            name: 'denseLayer3'
        });
        var z1 = denseLayer2.apply(y);
        var z2 = denseLayer3.apply(y);
        it('Calling execute with two fetches and diamond graph works', function () {
            var xValue = tfjs_core_1.ones([2, 2]);
            var feedDict = new executor_1.FeedDict([{ key: x, value: xValue }]);
            var callCounter = 0;
            denseLayer1.setCallHook(function () {
                callCounter++;
            });
            var outputs = executor_1.execute([z1, z2], feedDict);
            test_utils_1.expectTensorsClose(outputs[0], tfjs_core_1.tensor2d([10, 10, 10, 10, 10, 10, 10, 10], [2, 4]));
            test_utils_1.expectTensorsClose(outputs[1], tfjs_core_1.tensor2d([10, 10, 10, 10, 10, 10], [2, 3]));
            expect(callCounter).toEqual(2);
        });
    });
});
//# sourceMappingURL=executor_test.js.map