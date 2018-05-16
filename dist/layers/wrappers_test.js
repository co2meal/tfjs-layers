"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : new P(function (resolve) { resolve(result.value); }).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
var __generator = (this && this.__generator) || function (thisArg, body) {
    var _ = { label: 0, sent: function() { if (t[0] & 1) throw t[1]; return t[1]; }, trys: [], ops: [] }, f, y, t, g;
    return g = { next: verb(0), "throw": verb(1), "return": verb(2) }, typeof Symbol === "function" && (g[Symbol.iterator] = function() { return this; }), g;
    function verb(n) { return function (v) { return step([n, v]); }; }
    function step(op) {
        if (f) throw new TypeError("Generator is already executing.");
        while (_) try {
            if (f = 1, y && (t = y[op[0] & 2 ? "return" : op[0] ? "throw" : "next"]) && !(t = t.call(y, op[1])).done) return t;
            if (y = 0, t) op = [0, t.value];
            switch (op[0]) {
                case 0: case 1: t = op; break;
                case 4: _.label++; return { value: op[1], done: false };
                case 5: _.label++; y = op[1]; op = [0]; continue;
                case 7: op = _.ops.pop(); _.trys.pop(); continue;
                default:
                    if (!(t = _.trys, t = t.length > 0 && t[t.length - 1]) && (op[0] === 6 || op[0] === 2)) { _ = 0; continue; }
                    if (op[0] === 3 && (!t || (op[1] > t[0] && op[1] < t[3]))) { _.label = op[1]; break; }
                    if (op[0] === 6 && _.label < t[1]) { _.label = t[1]; t = op; break; }
                    if (t && _.label < t[2]) { _.label = t[2]; _.ops.push(op); break; }
                    if (t[2]) _.ops.pop();
                    _.trys.pop(); continue;
            }
            op = body.call(thisArg, _);
        } catch (e) { op = [6, e]; y = 0; } finally { f = t = 0; }
        if (op[0] & 5) throw op[1]; return { value: op[0] ? op[1] : void 0, done: true };
    }
};
var _this = this;
Object.defineProperty(exports, "__esModule", { value: true });
var tfjs_core_1 = require("@tensorflow/tfjs-core");
var tfl = require("../index");
var serialization_utils_1 = require("../utils/serialization_utils");
var test_utils_1 = require("../utils/test_utils");
var core_1 = require("./core");
var recurrent_1 = require("./recurrent");
var serialization_1 = require("./serialization");
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
    it('Serialization round trip', function () {
        var layer = tfl.layers.bidirectional({
            layer: new recurrent_1.SimpleRNN({ units: 3 }),
            mergeMode: 'concat',
            inputShape: [4, 4],
        });
        var model = tfl.sequential({ layers: [layer] });
        var unused = null;
        var modelJSON = model.toJSON(unused, false);
        var modelPrime = serialization_1.deserialize(serialization_utils_1.convertPythonicToTs(modelJSON));
        expect(modelPrime.layers[0].getConfig().mergeMode)
            .toEqual('concat');
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
    it('Backwards LSTM: predict and fit: concat', function () { return __awaiter(_this, void 0, void 0, function () {
        var lstm, bidi, model, x, y, history;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    lstm = tfl.layers.lstm({
                        units: 1,
                        kernelInitializer: 'ones',
                        recurrentInitializer: 'ones',
                        biasInitializer: 'ones',
                        goBackwards: true
                    });
                    bidi = tfl.layers.bidirectional({ layer: lstm, inputShape: [2, 2], mergeMode: 'concat' });
                    model = tfl.sequential({ layers: [bidi] });
                    model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });
                    x = tfjs_core_1.tensor3d([[[0.1, 0.2], [-0.1, 0.1]]]);
                    y = tfjs_core_1.tensor2d([[0.3, 0.5]]);
                    test_utils_1.expectTensorsClose(model.predict(x), tfjs_core_1.tensor2d([[0.69299805, 0.66088]]));
                    return [4, model.fit(x, y)];
                case 1:
                    history = _a.sent();
                    expect(history.history.loss[0]).toBeCloseTo(0.0901649);
                    test_utils_1.expectTensorsClose(model.predict(x), tfjs_core_1.tensor2d([[0.6927189, 0.66077083]]));
                    return [2];
            }
        });
    }); });
    it('Backwards LSTM: predict and fit: ave', function (done) {
        var lstm = tfl.layers.lstm({
            units: 1,
            kernelInitializer: 'ones',
            recurrentInitializer: 'ones',
            biasInitializer: 'ones',
            goBackwards: true
        });
        var bidi = tfl.layers.bidirectional({ layer: lstm, inputShape: [2, 2], mergeMode: 'ave' });
        var model = tfl.sequential({ layers: [bidi] });
        model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });
        var x = tfjs_core_1.tensor3d([[[0.1, 0.2], [-0.1, 0.1]]]);
        var y = tfjs_core_1.tensor2d([[0.4]]);
        test_utils_1.expectTensorsClose(model.predict(x), tfjs_core_1.tensor2d([[0.676939]]));
        model.fit(x, y)
            .then(function (history) {
            expect(history.history.loss[0]).toBeCloseTo(0.0766952);
            test_utils_1.expectTensorsClose(model.predict(x), tfjs_core_1.tensor2d([[0.6767467]]));
            done();
        })
            .catch(function (err) {
            done.fail(err.stack);
        });
    });
});
test_utils_1.describeMathCPUAndGPU('Bidirectional with initial state', function () {
    var sequenceLength = 4;
    var recurrentUnits = 3;
    var inputDim = 2;
    function createLayerAndTensors() {
        var initState1 = tfl.input({ shape: [recurrentUnits] });
        var initState2 = tfl.input({ shape: [recurrentUnits] });
        var inputTensor = tfl.input({ shape: [sequenceLength, inputDim] });
        var bidiLayer = tfl.layers.bidirectional({
            layer: tfl.layers.gru({
                units: recurrentUnits,
                kernelInitializer: 'zeros',
                recurrentInitializer: 'zeros',
                biasInitializer: 'ones'
            }),
            mergeMode: null,
        });
        var outputTensors = bidiLayer.apply(inputTensor, { initialState: [initState1, initState2] });
        return { bidiLayer: bidiLayer, initState1: initState1, initState2: initState2, inputTensor: inputTensor, outputTensors: outputTensors };
    }
    it('Correct shapes', function () {
        var layerAndTensors = createLayerAndTensors();
        expect(layerAndTensors.outputTensors.length).toEqual(2);
        expect(layerAndTensors.outputTensors[0].shape).toEqual([
            null, recurrentUnits
        ]);
        expect(layerAndTensors.outputTensors[1].shape).toEqual([
            null, recurrentUnits
        ]);
    });
    it('apply() with concrete tensors', function () {
        var layerAndTensors = createLayerAndTensors();
        var xVal = tfjs_core_1.ones([1, sequenceLength, inputDim]);
        var initState1Val = tfjs_core_1.ones([1, recurrentUnits]).mul(tfjs_core_1.scalar(-1));
        var initState2Val = tfjs_core_1.ones([1, recurrentUnits]);
        var yVals = layerAndTensors.bidiLayer.apply(xVal, { initialState: [initState1Val, initState2Val] });
        expect(yVals.length).toEqual(2);
        test_utils_1.expectTensorsClose(yVals[0], tfjs_core_1.tensor2d([[0.33863544, 0.33863544, 0.33863544]]));
        test_utils_1.expectTensorsClose(yVals[1], tfjs_core_1.tensor2d([[0.8188354, 0.8188354, 0.8188354]]));
    });
    it('Model predict', function () {
        var layerAndTensors = createLayerAndTensors();
        var model = tfl.model({
            inputs: [
                layerAndTensors.inputTensor, layerAndTensors.initState1,
                layerAndTensors.initState2
            ],
            outputs: layerAndTensors.outputTensors
        });
        var xVal = tfjs_core_1.ones([1, sequenceLength, inputDim]);
        var initState1Val = tfjs_core_1.ones([1, recurrentUnits]).mul(tfjs_core_1.scalar(-1));
        var initState2Val = tfjs_core_1.ones([1, recurrentUnits]);
        var yVals = model.predict([xVal, initState1Val, initState2Val]);
        expect(yVals.length).toEqual(2);
        test_utils_1.expectTensorsClose(yVals[0], tfjs_core_1.tensor2d([[0.33863544, 0.33863544, 0.33863544]]));
        test_utils_1.expectTensorsClose(yVals[1], tfjs_core_1.tensor2d([[0.8188354, 0.8188354, 0.8188354]]));
    });
    it('Model serialization round trip', function () {
        spyOn(console, 'warn');
        var layerAndTensors = createLayerAndTensors();
        var model = tfl.model({
            inputs: [
                layerAndTensors.inputTensor, layerAndTensors.initState1,
                layerAndTensors.initState2
            ],
            outputs: layerAndTensors.outputTensors
        });
        var json1 = model.toJSON(null, false);
        var model2 = serialization_1.deserialize(serialization_utils_1.convertPythonicToTs(json1));
        var json2 = model2.toJSON(null, false);
        expect(json2).toEqual(json1);
    });
    it('Incorrect number of initial-state tensors leads to error', function () {
        var initState1 = tfl.input({ shape: [recurrentUnits] });
        var x = tfl.input({ shape: [sequenceLength, inputDim] });
        var bidi = tfl.layers.bidirectional({
            layer: tfl.layers.gru({
                units: recurrentUnits,
            }),
            mergeMode: null,
        });
        expect(function () { return bidi.apply(x, {
            initialState: [initState1]
        }); }).toThrowError(/the state should be .*RNNs/);
    });
});
//# sourceMappingURL=wrappers_test.js.map