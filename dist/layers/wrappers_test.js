"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tfjs_core_1 = require("@tensorflow/tfjs-core");
var tfl = require("../index");
var test_utils_1 = require("../utils/test_utils");
var core_1 = require("./core");
var recurrent_1 = require("./recurrent");
var wrappers_1 = require("./wrappers");
test_utils_1.describeMathCPU('TimeDistributed Layer: Symbolic', function () {
    it('3D input: Dense', function () {
        var input = new tfl.SymbolicTensor('float32', [10, 8, 2], null, [], null);
        var wrapper = tfl.layers.timeDistributed({ layer: new core_1.Dense({ units: 3 }) });
        var output = wrapper.apply(input);
        expect(wrapper.trainable).toEqual(true);
        expect(wrapper.getWeights().length).toEqual(2);
        expect(output.dtype).toEqual(input.dtype);
        expect(output.shape).toEqual([10, 8, 3]);
    });
    it('4D input: Reshape', function () {
        var input = new tfl.SymbolicTensor('float32', [10, 8, 2, 3], null, [], null);
        var wrapper = tfl.layers.timeDistributed({ layer: new core_1.Reshape({ targetShape: [6] }) });
        var output = wrapper.apply(input);
        expect(output.dtype).toEqual(input.dtype);
        expect(output.shape).toEqual([10, 8, 6]);
    });
    it('2D input leads to exception', function () {
        var input = new tfl.SymbolicTensor('float32', [10, 2], null, [], null);
        var wrapper = tfl.layers.timeDistributed({ layer: new core_1.Dense({ units: 3 }) });
        expect(function () { return wrapper.apply(input); })
            .toThrowError(/TimeDistributed .*expects an input shape >= 3D, .* \[10,.*2\]/);
    });
    it('getConfig and fromConfig: round trip', function () {
        var wrapper = tfl.layers.timeDistributed({ layer: new core_1.Dense({ units: 3 }) });
        var config = wrapper.getConfig();
        var wrapperPrime = wrappers_1.TimeDistributed.fromConfig(wrappers_1.TimeDistributed, config);
        expect(wrapperPrime.getConfig()).toEqual(wrapper.getConfig());
    });
});
test_utils_1.describeMathCPUAndGPU('TimeDistributed Layer: Tensor', function () {
    it('3D input: Dense', function () {
        var input = tfjs_core_1.tensor3d([
            [[1, 2], [3, 4], [5, 6], [7, 8]],
            [[-1, -2], [-3, -4], [-5, -6], [-7, -8]]
        ], [2, 4, 2]);
        var wrapper = tfl.layers.timeDistributed({
            layer: new core_1.Dense({ units: 1, kernelInitializer: 'ones', useBias: false })
        });
        var output = wrapper.apply(input);
        test_utils_1.expectTensorsClose(output, tfjs_core_1.tensor3d([[[3], [7], [11], [15]], [[-3], [-7], [-11], [-15]]], [2, 4, 1]));
    });
});
test_utils_1.describeMathCPU('Bidirectional Layer: Symbolic', function () {
    var mergeModes = [
        null,
        'concat',
        'ave',
        'mul',
        'sum',
    ];
    var returnStateValues = [false, true];
    var _loop_1 = function (mergeMode) {
        var _loop_2 = function (returnState) {
            var testTitle = "3D input: returnSequence=false, " +
                ("mergeMode=" + mergeMode + "; returnState=" + returnState);
            it(testTitle, function () {
                var input = new tfl.SymbolicTensor('float32', [10, 8, 2], null, [], null);
                var bidi = tfl.layers.bidirectional({
                    layer: new recurrent_1.SimpleRNN({ units: 3, recurrentInitializer: 'glorotNormal', returnState: returnState }),
                    mergeMode: mergeMode,
                });
                var outputs = bidi.apply(input);
                expect(bidi.trainable).toEqual(true);
                expect(bidi.getWeights().length).toEqual(6);
                if (!returnState) {
                    if (mergeMode === null) {
                        outputs = outputs;
                        expect(outputs.length).toEqual(2);
                        expect(outputs[0].shape).toEqual([10, 3]);
                        expect(outputs[1].shape).toEqual([10, 3]);
                    }
                    else if (mergeMode === 'concat') {
                        outputs = outputs;
                        expect(outputs.shape).toEqual([10, 6]);
                    }
                    else {
                        outputs = outputs;
                        expect(outputs.shape).toEqual([10, 3]);
                    }
                }
                else {
                    if (mergeMode === null) {
                        outputs = outputs;
                        expect(outputs.length).toEqual(4);
                        expect(outputs[0].shape).toEqual([10, 3]);
                        expect(outputs[1].shape).toEqual([10, 3]);
                        expect(outputs[2].shape).toEqual([10, 3]);
                        expect(outputs[3].shape).toEqual([10, 3]);
                    }
                    else if (mergeMode === 'concat') {
                        outputs = outputs;
                        expect(outputs.length).toEqual(3);
                        expect(outputs[0].shape).toEqual([10, 6]);
                        expect(outputs[1].shape).toEqual([10, 3]);
                        expect(outputs[2].shape).toEqual([10, 3]);
                    }
                    else {
                        outputs = outputs;
                        expect(outputs.length).toEqual(3);
                        expect(outputs[0].shape).toEqual([10, 3]);
                        expect(outputs[1].shape).toEqual([10, 3]);
                        expect(outputs[2].shape).toEqual([10, 3]);
                    }
                }
            });
        };
        for (var _i = 0, returnStateValues_1 = returnStateValues; _i < returnStateValues_1.length; _i++) {
            var returnState = returnStateValues_1[_i];
            _loop_2(returnState);
        }
    };
    for (var _i = 0, mergeModes_1 = mergeModes; _i < mergeModes_1.length; _i++) {
        var mergeMode = mergeModes_1[_i];
        _loop_1(mergeMode);
    }
    it('returnSequence=true', function () {
        var input = new tfl.SymbolicTensor('float32', [10, 8, 2], null, [], null);
        var bidi = tfl.layers.bidirectional({
            layer: new recurrent_1.SimpleRNN({
                units: 3,
                recurrentInitializer: 'glorotNormal',
                returnSequences: true,
                returnState: true
            }),
            mergeMode: 'ave'
        });
        var outputs = bidi.apply(input);
        expect(outputs.length).toEqual(3);
        expect(outputs[0].shape).toEqual([10, 8, 3]);
        expect(outputs[1].shape).toEqual([10, 3]);
        expect(outputs[2].shape).toEqual([10, 3]);
    });
});
describe('checkBidirectionalMergeMode', function () {
    it('Valid values', function () {
        var extendedValues = wrappers_1.VALID_BIDIRECTIONAL_MERGE_MODES.concat([undefined, null]);
        for (var _i = 0, extendedValues_1 = extendedValues; _i < extendedValues_1.length; _i++) {
            var validValue = extendedValues_1[_i];
            wrappers_1.checkBidirectionalMergeMode(validValue);
        }
    });
    it('Invalid values', function () {
        expect(function () { return wrappers_1.checkBidirectionalMergeMode('foo'); }).toThrowError(/foo/);
        try {
            wrappers_1.checkBidirectionalMergeMode('bad');
        }
        catch (e) {
            expect(e).toMatch('BidirectionalMergeMode');
            for (var _i = 0, VALID_BIDIRECTIONAL_MERGE_MODES_1 = wrappers_1.VALID_BIDIRECTIONAL_MERGE_MODES; _i < VALID_BIDIRECTIONAL_MERGE_MODES_1.length; _i++) {
                var validValue = VALID_BIDIRECTIONAL_MERGE_MODES_1[_i];
                expect(e).toMatch(validValue);
            }
        }
    });
});
test_utils_1.describeMathCPUAndGPU('Bidirectional Layer: Tensor', function () {
    var bidi;
    var x;
    function createLayerAndData(mergeMode, returnState) {
        var units = 3;
        bidi = tfl.layers.bidirectional({
            layer: new recurrent_1.SimpleRNN({
                units: units,
                kernelInitializer: 'ones',
                recurrentInitializer: 'ones',
                useBias: false,
                returnState: returnState
            }),
            mergeMode: mergeMode,
        });
        var timeSteps = 4;
        var inputSize = 2;
        x = tfjs_core_1.tensor3d([[[0.05, 0.05], [-0.05, -0.05], [0.1, 0.1], [-0.1, -0.1]]], [1, timeSteps, inputSize]);
    }
    var mergeModes = [null, 'concat', 'mul'];
    var _loop_3 = function (mergeMode) {
        it("No returnState, mergeMode=" + mergeMode, function () {
            createLayerAndData(mergeMode, false);
            var y = bidi.apply(x);
            if (mergeMode === null) {
                y = y;
                expect(y.length).toEqual(2);
                test_utils_1.expectTensorsClose(y[0], tfjs_core_1.tensor2d([[0.9440416, 0.9440416, 0.9440416]], [1, 3]));
                test_utils_1.expectTensorsClose(y[1], tfjs_core_1.tensor2d([[-0.9842659, -0.9842659, -0.9842659]], [1, 3]));
            }
            else if (mergeMode === 'concat') {
                y = y;
                test_utils_1.expectTensorsClose(y, tfjs_core_1.tensor2d([[
                        0.9440416, 0.9440416, 0.9440416, -0.9842659, -0.9842659,
                        -0.9842659
                    ]], [1, 6]));
            }
            else if (mergeMode === 'mul') {
                y = y;
                test_utils_1.expectTensorsClose(y, tfjs_core_1.tensor2d([[-0.929188, -0.929188, -0.929188]], [1, 3]));
            }
        });
    };
    for (var _i = 0, mergeModes_2 = mergeModes; _i < mergeModes_2.length; _i++) {
        var mergeMode = mergeModes_2[_i];
        _loop_3(mergeMode);
    }
    it('returnState', function () {
        createLayerAndData('ave', true);
        var y = bidi.apply(x);
        expect(y.length).toEqual(3);
        test_utils_1.expectTensorsClose(y[0], tfjs_core_1.tensor2d([[-0.02011216, -0.02011216, -0.02011216]], [1, 3]));
        test_utils_1.expectTensorsClose(y[1], tfjs_core_1.tensor2d([[0.9440416, 0.9440416, 0.9440416]], [1, 3]));
        test_utils_1.expectTensorsClose(y[2], tfjs_core_1.tensor2d([[-0.9842659, -0.9842659, -0.9842659]], [1, 3]));
    });
});
//# sourceMappingURL=wrappers_test.js.map