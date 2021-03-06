"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tfjs_core_1 = require("@tensorflow/tfjs-core");
var K = require("../backend/tfjs_backend");
var tfl = require("../index");
var generic_utils_1 = require("../utils/generic_utils");
var math_utils_1 = require("../utils/math_utils");
var serialization_utils_1 = require("../utils/serialization_utils");
var test_utils_1 = require("../utils/test_utils");
describe('Dropout Layer: Symbolic', function () {
    var dropoutRates = [0, 0.5];
    var symbolicInputs = [
        new tfl.SymbolicTensor('float32', [10, 4], null, [], null),
        new tfl.SymbolicTensor('float32', [12, 10, 4], null, [], null),
        new tfl.SymbolicTensor('float32', [null, 4], null, [], null),
    ];
    var _loop_1 = function (rate) {
        var _loop_2 = function (symbolicInput) {
            var testTitle = "dropoutRate=" + rate + "; " +
                ("input shape=" + JSON.stringify(symbolicInput.shape));
            it(testTitle, function () {
                var dropoutLayer = tfl.layers.dropout({ rate: rate });
                var output = dropoutLayer.apply(symbolicInput);
                expect(output.dtype).toEqual(symbolicInput.dtype);
                expect(output.shape).toEqual(symbolicInput.shape);
                expect(output.sourceLayer).toEqual(dropoutLayer);
                expect(output.inputs).toEqual([symbolicInput]);
            });
        };
        for (var _i = 0, symbolicInputs_1 = symbolicInputs; _i < symbolicInputs_1.length; _i++) {
            var symbolicInput = symbolicInputs_1[_i];
            _loop_2(symbolicInput);
        }
    };
    for (var _i = 0, dropoutRates_1 = dropoutRates; _i < dropoutRates_1.length; _i++) {
        var rate = dropoutRates_1[_i];
        _loop_1(rate);
    }
});
test_utils_1.describeMathCPUAndGPU('Dropout Layer', function () {
    describe('tensor', function () {
        var inputShape = [2, 3, 4];
        var trainingValues = [false, true];
        var dropoutRates = [0, 0.5];
        var noiseShapes = [null, inputShape];
        var _loop_3 = function (training) {
            var _loop_4 = function (rate) {
                var _loop_5 = function (noiseShape) {
                    var testTitle = "training=" + training + ", dropoutRate=" + rate + ", " +
                        ("noiseShape=" + JSON.stringify(noiseShape));
                    it(testTitle, function () {
                        var x = tfjs_core_1.ones(inputShape);
                        var dropoutLayer = tfl.layers.dropout({ rate: rate, noiseShape: noiseShape });
                        var y = dropoutLayer.apply(x, { training: training });
                        expect(x.dtype).toEqual(y.dtype);
                        expect(x.shape).toEqual(y.shape);
                        var xValue = x.dataSync();
                        var yValue = y.dataSync();
                        var nKept = 0;
                        for (var i = 0; i < xValue.length; ++i) {
                            if (yValue[i] !== 0) {
                                nKept++;
                                if (training) {
                                    expect(yValue[i]).toBeCloseTo(1 / (1 - rate));
                                }
                                else {
                                    expect(yValue[i]).toBeCloseTo(1);
                                }
                            }
                        }
                        var numel = K.countParams(x);
                        if (rate === 0 || !training) {
                            expect(nKept).toEqual(numel);
                        }
                        else {
                            expect(nKept).toBeLessThan(numel);
                        }
                    });
                };
                for (var _i = 0, noiseShapes_1 = noiseShapes; _i < noiseShapes_1.length; _i++) {
                    var noiseShape = noiseShapes_1[_i];
                    _loop_5(noiseShape);
                }
            };
            for (var _i = 0, dropoutRates_2 = dropoutRates; _i < dropoutRates_2.length; _i++) {
                var rate = dropoutRates_2[_i];
                _loop_4(rate);
            }
        };
        for (var _i = 0, trainingValues_1 = trainingValues; _i < trainingValues_1.length; _i++) {
            var training = trainingValues_1[_i];
            _loop_3(training);
        }
    });
});
test_utils_1.describeMathCPU('Dense Layer: Symbolic', function () {
    var units = 3;
    var activations = [null, 'linear', 'relu', 'softmax'];
    var symbolicInputs = [
        new tfl.SymbolicTensor('float32', [10, 4], null, [], null),
        new tfl.SymbolicTensor('float32', [12, 10, 4], null, [], null),
        new tfl.SymbolicTensor('float32', [14, 12, 10, 4], null, [], null),
        new tfl.SymbolicTensor('float32', [null, 4], null, [], null),
        new tfl.SymbolicTensor('float32', [null, 10, 4], null, [], null),
        new tfl.SymbolicTensor('float32', [null, 12, 10, 4], null, [], null),
    ];
    var _loop_6 = function (activation) {
        var _loop_7 = function (symbolicInput) {
            it("Generates correct symbolic output: " +
                ("activation=" + activation + ", ") +
                ("input shape=" + JSON.stringify(symbolicInput.shape)), function () {
                var denseLayer = tfl.layers.dense({ units: units, activation: activation });
                var output = denseLayer.apply(symbolicInput);
                var expectedShape = symbolicInput.shape;
                expectedShape[expectedShape.length - 1] = units;
                expect(output.shape).toEqual(expectedShape);
                expect(output.sourceLayer).toEqual(denseLayer);
                expect(output.inputs).toEqual([symbolicInput]);
            });
        };
        for (var _i = 0, symbolicInputs_2 = symbolicInputs; _i < symbolicInputs_2.length; _i++) {
            var symbolicInput = symbolicInputs_2[_i];
            _loop_7(symbolicInput);
        }
    };
    for (var _i = 0, activations_1 = activations; _i < activations_1.length; _i++) {
        var activation = activations_1[_i];
        _loop_6(activation);
    }
    it('2D cascade: With undetermined dimension', function () {
        var input1 = new tfl.SymbolicTensor('float32', [null, 4], null, [], null);
        var denseLayer1 = tfl.layers.dense({ units: 3 });
        var output1 = denseLayer1.apply(input1);
        var denseLayer2 = tfl.layers.dense({ units: 6 });
        var output2 = denseLayer2.apply(output1);
        expect(output1.shape).toEqual([null, 3]);
        expect(output1.sourceLayer).toEqual(denseLayer1);
        expect(output1.inputs).toEqual([input1]);
        expect(output2.shape).toEqual([null, 6]);
        expect(output2.sourceLayer).toEqual(denseLayer2);
        expect(output2.inputs).toEqual([output1]);
    });
    it('Using 1D input leads to error', function () {
        var input = new tfl.SymbolicTensor('float32', [4], null, [], null);
        var denseLayer = tfl.layers.dense({ units: 3 });
        expect(function () { return denseLayer.apply(input); }).toThrowError();
    });
    it('Different rank but compatible shape works', function () {
        var denseLayer = tfl.layers.dense({ units: 3 });
        var input1 = new tfl.SymbolicTensor('float32', [null, 4], null, [], null);
        var input2 = new tfl.SymbolicTensor('float32', [null, 6, 4], null, [], null);
        var output1 = denseLayer.apply(input1);
        expect(output1.shape).toEqual([null, 3]);
        expect(output1.sourceLayer).toEqual(denseLayer);
        expect(output1.inputs).toEqual([input1]);
        var output2 = denseLayer.apply(input2);
        expect(output2.shape).toEqual([null, 6, 3]);
        expect(output2.sourceLayer).toEqual(denseLayer);
        expect(output2.inputs).toEqual([input2]);
    });
    it('2D incompatible shape leads to error', function () {
        var denseLayer = tfl.layers.dense({ units: 3 });
        var input1 = new tfl.SymbolicTensor('float32', [null, 4], null, [], null);
        var input2 = new tfl.SymbolicTensor('float32', [null, 5], null, [], null);
        var output1 = denseLayer.apply(input1);
        expect(output1.shape).toEqual([null, 3]);
        expect(output1.sourceLayer).toEqual(denseLayer);
        expect(output1.inputs).toEqual([input1]);
        expect(function () {
            denseLayer.apply(input2);
        }).toThrowError(/incompatible with layer .* axis -1/);
    });
    it('Invalid kernelInitializer', function () {
        expect(function () {
            tfl.layers.dense({ units: 4, kernelInitializer: 'invalid_initializer!' });
        }).toThrowError(/Unknown initializer/);
    });
    it('Invalid activation', function () {
        expect(function () {
            tfl.layers.dense({ units: 4, activation: 'invalid_ativation!' });
        }).toThrowError(/Unknown activation/);
    });
});
test_utils_1.describeMathCPUAndGPU('Dense Layer: Tensor', function () {
    var units = 6;
    var useBiases = [null, false, true];
    var biasInitializers = ['zeros', 'ones'];
    var activations = [null, 'linear', 'relu', 'softmax'];
    var inputLastDims = [5, 8];
    var _loop_8 = function (useBias) {
        var _loop_9 = function (biasInitializer) {
            var _loop_10 = function (activation) {
                var _loop_11 = function (inputLastDim) {
                    it("Call once: useBias=" + useBias + ", " +
                        ("biasInitializer=" + biasInitializer + ", ") +
                        ("activation=" + activation + ", ") +
                        ("inputLastDim=" + JSON.stringify(inputLastDim)), function () {
                        var input = tfjs_core_1.ones([2, inputLastDim]);
                        var denseLayer = tfl.layers.dense({
                            units: units,
                            useBias: useBias,
                            biasInitializer: biasInitializer,
                            activation: activation,
                            kernelInitializer: 'ones'
                        });
                        var expectedElementValue;
                        if (activation === 'softmax') {
                            expectedElementValue = 1 / units;
                        }
                        else {
                            expectedElementValue = input.shape[input.shape.length - 1];
                            if (useBias !== false && biasInitializer === 'ones') {
                                expectedElementValue += 1;
                            }
                        }
                        var expectedShape = input.shape.slice();
                        expectedShape[expectedShape.length - 1] = units;
                        var expectedOutput;
                        if (input.rank === 2) {
                            expectedOutput = tfjs_core_1.tensor2d(generic_utils_1.pyListRepeat(expectedElementValue, math_utils_1.arrayProd(expectedShape)), [expectedShape[0], expectedShape[1]]);
                        }
                        test_utils_1.expectTensorsClose(denseLayer.apply(input, null), expectedOutput);
                    });
                };
                for (var _i = 0, inputLastDims_1 = inputLastDims; _i < inputLastDims_1.length; _i++) {
                    var inputLastDim = inputLastDims_1[_i];
                    _loop_11(inputLastDim);
                }
            };
            for (var _i = 0, activations_2 = activations; _i < activations_2.length; _i++) {
                var activation = activations_2[_i];
                _loop_10(activation);
            }
        };
        for (var _i = 0, biasInitializers_1 = biasInitializers; _i < biasInitializers_1.length; _i++) {
            var biasInitializer = biasInitializers_1[_i];
            _loop_9(biasInitializer);
        }
    };
    for (var _i = 0, useBiases_1 = useBiases; _i < useBiases_1.length; _i++) {
        var useBias = useBiases_1[_i];
        _loop_8(useBias);
    }
    it('Calling apply again with incompatible shape leads to error', function () {
        var input1 = tfjs_core_1.ones([2, 2]);
        var input2 = tfjs_core_1.ones([3, 2]);
        var input3 = tfjs_core_1.ones([3, 3]);
        var denseLayer = tfl.layers.dense({ units: 4, kernelInitializer: 'ones' });
        test_utils_1.expectTensorsClose(denseLayer.apply(input1), tfjs_core_1.tensor2d([2, 2, 2, 2, 2, 2, 2, 2], [2, 4]));
        test_utils_1.expectTensorsClose(denseLayer.apply(input2), tfjs_core_1.tensor2d([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [3, 4]));
        expect(function () { return denseLayer.apply(input3); }).toThrowError();
    });
    it('Calling apply with compatible symbolic input after Tensor input works', function () {
        var concreteInput = tfjs_core_1.ones([2, 2]);
        var symbolicInput = new tfl.SymbolicTensor('float32', [2, 2], null, [], null);
        var denseLayer = tfl.layers.dense({ units: 4, kernelInitializer: 'ones' });
        test_utils_1.expectTensorsClose(denseLayer.apply(concreteInput), tfjs_core_1.tensor2d([2, 2, 2, 2, 2, 2, 2, 2], [2, 4]));
        var symbolicOuptut = denseLayer.apply(symbolicInput);
        expect(symbolicOuptut.shape).toEqual([2, 4]);
        expect(symbolicOuptut.sourceLayer).toEqual(denseLayer);
        expect(symbolicOuptut.inputs).toEqual([symbolicInput]);
    });
    it('Calling apply with incompatible symbolic input after Tensor', function () {
        var concreteInput = tfjs_core_1.ones([2, 2]);
        var symbolicInput = new tfl.SymbolicTensor('float32', [2, 3], null, [], null);
        var denseLayer = tfl.layers.dense({ units: 4, kernelInitializer: 'ones' });
        test_utils_1.expectTensorsClose(denseLayer.apply(concreteInput), tfjs_core_1.tensor2d([2, 2, 2, 2, 2, 2, 2, 2], [2, 4]));
        expect(function () {
            denseLayer.apply(symbolicInput);
        }).toThrowError(/incompatible with layer .* axis -1/);
    });
});
describe('Flatten Layer: Symbolic', function () {
    var symbolicInputs = [
        new tfl.SymbolicTensor('float32', [12, 10, 4], null, [], null),
        new tfl.SymbolicTensor('float32', [14, 12, 10, 4], null, [], null),
        new tfl.SymbolicTensor('float32', [null, 10, 4], null, [], null),
        new tfl.SymbolicTensor('float32', [null, 12, 10, 4], null, [], null),
    ];
    var _loop_12 = function (symbolicInput) {
        it("Generates correct symbolic output: no-arg constructor: " +
            ("input shape=" + JSON.stringify(symbolicInput.shape)), function () {
            var flattenLayer = tfl.layers.flatten();
            var output = flattenLayer.apply(symbolicInput);
            var expectedShape = [symbolicInput.shape[0], math_utils_1.arrayProd(symbolicInput.shape, 1)];
            expect(output.shape).toEqual(expectedShape);
            expect(output.sourceLayer).toEqual(flattenLayer);
            expect(output.inputs).toEqual([symbolicInput]);
        });
        it("Generates correct symbolic output: empty one-arg constructor: " +
            ("input shape=" + JSON.stringify(symbolicInput.shape)), function () {
            var flattenLayer = tfl.layers.flatten({});
            var output = flattenLayer.apply(symbolicInput);
            var expectedShape = [symbolicInput.shape[0], math_utils_1.arrayProd(symbolicInput.shape, 1)];
            expect(output.shape).toEqual(expectedShape);
            expect(output.sourceLayer).toEqual(flattenLayer);
            expect(output.inputs).toEqual([symbolicInput]);
        });
    };
    for (var _i = 0, symbolicInputs_3 = symbolicInputs; _i < symbolicInputs_3.length; _i++) {
        var symbolicInput = symbolicInputs_3[_i];
        _loop_12(symbolicInput);
    }
    it('2D tfl.SymbolicTensor leads to error', function () {
        var flattenLayer = tfl.layers.flatten();
        var x = new tfl.SymbolicTensor('float32', [null, 4], null, [], null);
        expect(function () { return flattenLayer.apply(x); }).toThrowError();
    });
    it('3D with undetermined input size leads to error', function () {
        var flattenLayer = tfl.layers.flatten({});
        var x = new tfl.SymbolicTensor('float32', [8, 4, null], null, [], null);
        expect(function () { return flattenLayer.apply(x); }).toThrowError(/not fully defined/);
    });
});
test_utils_1.describeMathCPUAndGPU('Flatten Layer: Tensor', function () {
    it('Attempt to apply on Tensor2D leads to error', function () {
        var flattenLayer = tfl.layers.flatten();
        var x = tfjs_core_1.tensor2d([[1, 3], [3, 3]], [2, 2]);
        expect(function () { return flattenLayer.apply(x); }).toThrowError();
    });
    it('Flattens Tensor3D', function () {
        var flattenLayer = tfl.layers.flatten();
        var x = tfjs_core_1.tensor3d([[[10, 20], [30, 40]], [[-10, -20], [-30, -40]]], [2, 2, 2]);
        var expectedOutput = tfjs_core_1.tensor2d([[10, 20, 30, 40], [-10, -20, -30, -40]], [2, 4]);
        test_utils_1.expectTensorsClose(flattenLayer.apply(x, null), expectedOutput);
    });
    it('Flattens Tensor4D', function () {
        var flattenLayer = tfl.layers.flatten();
        var x = tfjs_core_1.tensor4d([
            [[[10, 20], [30, 40]], [[-10, -20], [-30, -40]]],
            [[[1, 2], [3, 4]], [[-1, -2], [-3, -4]]]
        ], [2, 2, 2, 2]);
        var expectedOutput = tfjs_core_1.tensor2d([10, 20, 30, 40, -10, -20, -30, -40, 1, 2, 3, 4, -1, -2, -3, -4], [2, 8]);
        test_utils_1.expectTensorsClose(flattenLayer.apply(x, null), expectedOutput);
    });
});
test_utils_1.describeMathCPUAndGPU('Activation Layer: Tensor', function () {
    var inputShape = [1];
    it('linear', function () {
        var x = tfjs_core_1.mul(tfjs_core_1.scalar(10), tfjs_core_1.ones(inputShape));
        var layer = tfl.layers.activation({ activation: 'linear' });
        var output = layer.apply(x);
        test_utils_1.expectTensorsClose(output, x);
    });
    it('relu', function () {
        var x = tfjs_core_1.mul(tfjs_core_1.scalar(-5), tfjs_core_1.ones(inputShape));
        var expectedValue = tfjs_core_1.zeros(inputShape);
        var layer = tfl.layers.activation({ activation: 'relu' });
        var output = layer.apply(x);
        test_utils_1.expectTensorsClose(output, expectedValue);
    });
    it('sigmoid', function () {
        var val = 10;
        var x = tfjs_core_1.mul(tfjs_core_1.scalar(val), tfjs_core_1.ones(inputShape));
        var expectedValue = tfjs_core_1.mul(tfjs_core_1.scalar(1 / (1 + Math.exp(-1 * val))), tfjs_core_1.ones(inputShape));
        var layer = tfl.layers.activation({ activation: 'sigmoid' });
        var output = layer.apply(x);
        test_utils_1.expectTensorsClose(output, expectedValue);
    });
    it('softmax', function () {
        var x = tfjs_core_1.mul(tfjs_core_1.scalar(10), tfjs_core_1.ones(inputShape));
        var expectedValue = tfjs_core_1.ones(inputShape);
        var layer = tfl.layers.activation({ activation: 'softmax' });
        var output = layer.apply(x);
        test_utils_1.expectTensorsClose(output, expectedValue);
    });
    it('Serialization round trip', function () {
        var layer = tfl.layers.activation({ activation: 'relu' });
        var pythonicConfig = serialization_utils_1.convertTsToPythonic(layer.getConfig());
        var tsConfig = serialization_utils_1.convertPythonicToTs(pythonicConfig);
        var layerPrime = tfl.layers.activation(tsConfig);
        expect(layerPrime.getConfig().activation).toEqual('relu');
    });
});
describe('RepeatVector Layer: Symbolic', function () {
    it('All dimensions known.', function () {
        var symbolicInput = new tfl.SymbolicTensor('float32', [3, 4], null, [], null);
        var repeatVectorLayer = tfl.layers.repeatVector({ n: 2 });
        var output = repeatVectorLayer.apply(symbolicInput);
        expect(output.shape).toEqual([3, 2, 4]);
        expect(output.sourceLayer).toEqual(repeatVectorLayer);
        expect(output.inputs).toEqual([symbolicInput]);
    });
});
test_utils_1.describeMathCPUAndGPU('RepeatVector Layer: Tensor', function () {
    it('With 2D tensor', function () {
        var repeatVectorLayer = tfl.layers.repeatVector({ n: 3 });
        var x = tfjs_core_1.tensor2d([[10, 20], [30, 40]], [2, 2]);
        var expectedOutput = tfjs_core_1.tensor3d([[[10, 20], [10, 20], [10, 20]], [[30, 40], [30, 40], [30, 40]]], [2, 3, 2]);
        test_utils_1.expectTensorsClose(repeatVectorLayer.apply(x, null), expectedOutput);
    });
});
describe('Reshape Layer: Symbolic', function () {
    it('All dimensions known.', function () {
        var symbolicInput = new tfl.SymbolicTensor('float32', [12, 10, 4], null, [], null);
        var targetShape = [5, 8];
        var flattenLayer = tfl.layers.reshape({ targetShape: targetShape });
        var output = flattenLayer.apply(symbolicInput);
        expect(output.shape).toEqual([12, 5, 8]);
        expect(output.sourceLayer).toEqual(flattenLayer);
        expect(output.inputs).toEqual([symbolicInput]);
    });
    it('One unknown dimension.', function () {
        var symbolicInput = new tfl.SymbolicTensor('float32', [12, 10, 4], null, [], null);
        var targetShape = [5, null];
        var flattenLayer = tfl.layers.reshape({ targetShape: targetShape });
        var output = flattenLayer.apply(symbolicInput);
        expect(output.shape).toEqual([12, 5, 8]);
        expect(output.sourceLayer).toEqual(flattenLayer);
        expect(output.inputs).toEqual([symbolicInput]);
    });
    it('Incompatible size.', function () {
        var symbolicInput = new tfl.SymbolicTensor('float32', [12, 10, 4], null, [], null);
        var targetShape = [8, 8];
        var flattenLayer = tfl.layers.reshape({ targetShape: targetShape });
        expect(function () { return flattenLayer.apply(symbolicInput); })
            .toThrowError(/Total size of new array must be unchanged/);
    });
    it('Two unknown dimensions.', function () {
        var symbolicInput = new tfl.SymbolicTensor('float32', [12, 10, 4], null, [], null);
        var targetShape = [null, null];
        var flattenLayer = tfl.layers.reshape({ targetShape: targetShape });
        expect(function () { return flattenLayer.apply(symbolicInput); })
            .toThrowError(/Can only specifiy one unknown dimension/);
    });
    it('One unknown with indivisible size.', function () {
        var symbolicInput = new tfl.SymbolicTensor('float32', [12, 10, 4], null, [], null);
        var targetShape = [7, null];
        var flattenLayer = tfl.layers.reshape({ targetShape: targetShape });
        expect(function () { return flattenLayer.apply(symbolicInput); })
            .toThrowError(/Total size of new array must be unchanged/);
    });
    it('Serialization round-trip', function () {
        var layer = tfl.layers.reshape({ targetShape: [2, 3] });
        var pythonicConfig = serialization_utils_1.convertTsToPythonic(layer.getConfig());
        var tsConfig = serialization_utils_1.convertPythonicToTs(pythonicConfig);
        var layerPrime = tfl.layers.reshape(tsConfig);
        expect(layerPrime.getConfig().targetShape).toEqual([2, 3]);
    });
});
test_utils_1.describeMathCPUAndGPU('Reshape Layer: Tensor', function () {
    it('Reshape Tensor3D to Tensor3D: All dimensions known', function () {
        var reshapeLayer = tfl.layers.reshape({ targetShape: [4, 1] });
        var x = tfjs_core_1.tensor3d([[[10, 20], [30, 40]], [[-10, -20], [-30, -40]]], [2, 2, 2]);
        var expectedOutput = tfjs_core_1.tensor3d([10, 20, 30, 40, -10, -20, -30, -40], [2, 4, 1]);
        test_utils_1.expectTensorsClose(reshapeLayer.apply(x, null), expectedOutput);
    });
    it('Reshape Tensor3D to Tensor2D: All dimensions known', function () {
        var reshapeLayer = tfl.layers.reshape({ targetShape: [4] });
        var x = tfjs_core_1.tensor3d([[[10, 20], [30, 40]], [[-10, -20], [-30, -40]]], [2, 2, 2]);
        var expectedOutput = tfjs_core_1.tensor2d([10, 20, 30, 40, -10, -20, -30, -40], [2, 4]);
        test_utils_1.expectTensorsClose(reshapeLayer.apply(x, null), expectedOutput);
    });
    it('Reshape Tensor2D to Tensor3D: All dimensions known', function () {
        var reshapeLayer = tfl.layers.reshape({ targetShape: [3, 2] });
        var x = tfjs_core_1.tensor2d([[10, 20, 30, 40, 50, 60], [-10, -20, -30, -40, -50, -60]], [2, 6]);
        var expectedOutput = tfjs_core_1.tensor3d([10, 20, 30, 40, 50, 60, -10, -20, -30, -40, -50, -60], [2, 3, 2]);
        test_utils_1.expectTensorsClose(reshapeLayer.apply(x, null), expectedOutput);
    });
    var _loop_13 = function (unknownDim) {
        it("Reshape Tensor2D to Tensor3D: Last dimension unknown as " + unknownDim, function () {
            var reshapeLayer = tfl.layers.reshape({ targetShape: [3, unknownDim] });
            var x = tfjs_core_1.tensor2d([[10, 20, 30, 40, 50, 60], [-10, -20, -30, -40, -50, -60]], [2, 6]);
            var expectedOutput = tfjs_core_1.tensor3d([10, 20, 30, 40, 50, 60, -10, -20, -30, -40, -50, -60], [2, 3, 2]);
            test_utils_1.expectTensorsClose(reshapeLayer.apply(x, null), expectedOutput);
        });
        it("Reshape Tensor2D to Tensor3D: First dimension unknown as " + unknownDim, function () {
            var reshapeLayer = tfl.layers.reshape({ targetShape: [unknownDim, 3] });
            var x = tfjs_core_1.tensor2d([[10, 20, 30, 40, 50, 60], [-10, -20, -30, -40, -50, -60]], [2, 6]);
            var expectedOutput = tfjs_core_1.tensor3d([10, 20, 30, 40, 50, 60, -10, -20, -30, -40, -50, -60], [2, 2, 3]);
            test_utils_1.expectTensorsClose(reshapeLayer.apply(x, null), expectedOutput);
        });
    };
    for (var _i = 0, _a = [-1, null]; _i < _a.length; _i++) {
        var unknownDim = _a[_i];
        _loop_13(unknownDim);
    }
    it('Known but incompatible dimensions', function () {
        var reshapeLayer = tfl.layers.reshape({ targetShape: [3, 3] });
        var x = tfjs_core_1.tensor3d([[[10, 20], [30, 40]], [[-10, -20], [-30, -40]]], [2, 2, 2]);
        expect(function () { return reshapeLayer.apply(x, null); })
            .toThrowError(/Total size of new array must be unchanged/);
    });
    it('Unknown and incompatible dimensions', function () {
        var reshapeLayer = tfl.layers.reshape({ targetShape: [3, null] });
        var x = tfjs_core_1.tensor3d([[[10, 20], [30, 40]], [[-10, -20], [-30, -40]]], [2, 2, 2]);
        expect(function () { return reshapeLayer.apply(x, null); })
            .toThrowError(/Total size of new array must be unchanged/);
    });
    it('More than one unknown dimension.', function () {
        var reshapeLayer = tfl.layers.reshape({ targetShape: [null, null] });
        var x = tfjs_core_1.tensor3d([[[10, 20], [30, 40]], [[-10, -20], [-30, -40]]], [2, 2, 2]);
        expect(function () { return reshapeLayer.apply(x, null); })
            .toThrowError(/Can only specifiy one unknown dimension/);
    });
});
describe('Permute Layer: Symbolic', function () {
    it('1D Trivial', function () {
        var symbolicInput = new tfl.SymbolicTensor('float32', [null, 4], null, [], null);
        var dims = [1];
        var permuteLayer = tfl.layers.permute({ dims: dims });
        var output = permuteLayer.apply(symbolicInput);
        expect(output.shape).toEqual([null, 4]);
        expect(output.sourceLayer).toEqual(permuteLayer);
        expect(output.inputs).toEqual([symbolicInput]);
    });
    it('2D', function () {
        var symbolicInput = new tfl.SymbolicTensor('float32', [null, 4, 6], null, [], null);
        var dims = [2, 1];
        var permuteLayer = tfl.layers.permute({ dims: dims });
        var output = permuteLayer.apply(symbolicInput);
        expect(output.shape).toEqual([null, 6, 4]);
        expect(output.sourceLayer).toEqual(permuteLayer);
        expect(output.inputs).toEqual([symbolicInput]);
    });
    it('3D', function () {
        var symbolicInput = new tfl.SymbolicTensor('float32', [null, 4, 6, 8], null, [], null);
        var dims = [3, 1, 2];
        var permuteLayer = tfl.layers.permute({ dims: dims });
        var output = permuteLayer.apply(symbolicInput);
        expect(output.shape).toEqual([null, 8, 4, 6]);
        expect(output.sourceLayer).toEqual(permuteLayer);
        expect(output.inputs).toEqual([symbolicInput]);
    });
    it('Missing dims config leads to Error', function () {
        expect(function () { return tfl.layers.permute({}); }).toThrowError(/dims.* missing/);
    });
    it('Non-Array dims config leads to Error', function () {
        expect(function () { return tfl.layers.permute({ dims: 1 }); })
            .toThrowError(/requires.*dims.* to be an Array/);
    });
    it('Non-consecutive dims values leads to Error', function () {
        expect(function () { return tfl.layers.permute({ dims: [1, 3] }); })
            .toThrowError(/Invalid permutation .*dims/);
    });
    it('Repeating dims values leads to Error', function () {
        expect(function () { return tfl.layers.permute({ dims: [1, 1, 3] }); })
            .toThrowError(/Invalid permutation .*dims/);
    });
    it('Dims values containing 0 leads to Error', function () {
        expect(function () { return tfl.layers.permute({ dims: [0, 1, 2] }); })
            .toThrowError(/Invalid permutation .*dims/);
    });
    it('Serialization round-trip', function () {
        var layer = tfl.layers.permute({ dims: [1, 3, 2] });
        var pythonicConfig = serialization_utils_1.convertTsToPythonic(layer.getConfig());
        var tsConfig = serialization_utils_1.convertPythonicToTs(pythonicConfig);
        var layerPrime = tfl.layers.permute(tsConfig);
        expect(layerPrime.getConfig().dims).toEqual([1, 3, 2]);
    });
});
test_utils_1.describeMathCPUAndGPU('Permute Layer: Tensor', function () {
    it('2D', function () {
        var permuteLayer = tfl.layers.permute({ dims: [2, 1] });
        var x = tfjs_core_1.tensor3d([[[10, 20], [30, 40]], [[-10, -20], [-30, -40]]], [2, 2, 2]);
        var expectedOutput = tfjs_core_1.tensor3d([[[10, 30], [20, 40]], [[-10, -30], [-20, -40]]], [2, 2, 2]);
        test_utils_1.expectTensorsClose(permuteLayer.apply(x), expectedOutput);
    });
});
//# sourceMappingURL=core_test.js.map