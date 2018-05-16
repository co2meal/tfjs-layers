"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var tfjs_core_1 = require("@tensorflow/tfjs-core");
var tfl = require("../index");
var initializers = require("../initializers");
var test_utils_1 = require("../utils/test_utils");
var variables_1 = require("../variables");
var container_1 = require("./container");
var topology_1 = require("./topology");
var LayerForTest = (function (_super) {
    __extends(LayerForTest, _super);
    function LayerForTest(config) {
        return _super.call(this, config) || this;
    }
    LayerForTest.className = 'LayerForTest';
    return LayerForTest;
}(tfl.layers.Layer));
describe('InputSpec', function () {
    it('initializes with expected default values.', function () {
        var inputSpec = new topology_1.InputSpec({});
        expect(inputSpec.dtype).toBeUndefined();
        expect(inputSpec.shape).toBeUndefined();
        expect(inputSpec.ndim).toBeUndefined();
        expect(inputSpec.maxNDim).toBeUndefined();
        expect(inputSpec.minNDim).toBeUndefined();
        expect(inputSpec.axes).toEqual({});
    });
    it('initializes with inputSpec.ndim = shape.length when shape is specified.', function () {
        var shape = [1, 2, 3];
        var expectedValue = shape.length;
        var inputSpec = new topology_1.InputSpec({ shape: [1, 2, 3], ndim: -1 });
        expect(inputSpec.ndim).toEqual(expectedValue);
    });
    it('initializes inputSpec.axes when axes specified.', function () {
        var expectedValue = { 1: 2 };
        var inputSpec = new topology_1.InputSpec({ axes: expectedValue });
        expect(inputSpec.axes).toEqual(expectedValue);
    });
});
describe('Node', function () {
    var outboundLayerName = 'outboundLayer';
    var inboundLayerName = 'inboundLayer';
    var outboundLayer = new LayerForTest({ name: outboundLayerName });
    var inboundLayers = [new LayerForTest({ name: inboundLayerName })];
    var nodeIndices = [0];
    var tensorIndices = [0];
    var inputTensors = [new tfl.SymbolicTensor('float32', [1], null, [], {})];
    var outputTensors = [new tfl.SymbolicTensor('float32', [2, 2], null, [], {})];
    var inputMasks = [tfjs_core_1.zeros([1])];
    var outputMasks = [tfjs_core_1.zeros([1])];
    var inputShapes = [[1]];
    var outputShapes = [[1], [1]];
    var callArgs = { mask: tfjs_core_1.zeros([1]) };
    var node = new topology_1.Node({
        outboundLayer: outboundLayer,
        inboundLayers: inboundLayers,
        nodeIndices: nodeIndices,
        tensorIndices: tensorIndices,
        inputTensors: inputTensors,
        outputTensors: outputTensors,
        inputMasks: inputMasks,
        outputMasks: outputMasks,
        inputShapes: inputShapes,
        outputShapes: outputShapes
    }, callArgs);
    it('initializes object as expected.', function () {
        expect(node.outboundLayer).toEqual(outboundLayer);
        expect(node.inboundLayers).toEqual(inboundLayers);
        expect(node.nodeIndices).toEqual(nodeIndices);
        expect(node.tensorIndices).toEqual(tensorIndices);
        expect(node.inputTensors).toEqual(inputTensors);
        expect(node.outputTensors).toEqual(outputTensors);
        expect(node.inputMasks).toEqual(inputMasks);
        expect(node.outputMasks).toEqual(outputMasks);
        expect(node.inputShapes).toEqual(inputShapes);
        expect(node.outputShapes).toEqual(outputShapes);
        expect(node.callArgs).toEqual(callArgs);
        expect(inboundLayers[0].outboundNodes).toEqual([node]);
        expect(node.outboundLayer.inboundNodes).toEqual([node]);
    });
    it('generates expected SerializableNodeConfig.', function () {
        var nodeConfig = node.getConfig();
        expect(nodeConfig.outboundLayer).toEqual(outboundLayerName);
        expect(nodeConfig.inboundLayers).toEqual([inboundLayerName]);
        expect(nodeConfig.nodeIndices).toEqual(nodeIndices);
        expect(nodeConfig.tensorIndices).toEqual(tensorIndices);
    });
    it('generates unique IDs', function () {
        var secondNode = new topology_1.Node({
            outboundLayer: outboundLayer,
            inboundLayers: inboundLayers,
            nodeIndices: nodeIndices,
            tensorIndices: tensorIndices,
            inputTensors: inputTensors,
            outputTensors: outputTensors,
            inputMasks: inputMasks,
            outputMasks: outputMasks,
            inputShapes: inputShapes,
            outputShapes: outputShapes
        }, callArgs);
        expect(secondNode.id).not.toEqual(node.id);
    });
});
test_utils_1.describeMathCPU('Layer', function () {
    describe('initialized to its defaults', function () {
        var defaultLayer;
        beforeEach(function () {
            defaultLayer = new LayerForTest({});
        });
        it('has a default layer name of layer_....', function () {
            expect(defaultLayer.name).toMatch(/^layer_.+$/);
        });
        it('has null inputSpecs.', function () {
            expect(defaultLayer.inputSpec).toBeNull();
        });
        it('does not support masking (supportsMasking == false).', function () {
            expect(defaultLayer.supportsMasking).toEqual(false);
        });
        it('is trainable.', function () {
            expect(defaultLayer.trainable).toEqual(true);
        });
        it('has an undefined batchInputShape.', function () {
            expect(defaultLayer.batchInputShape).toBeUndefined();
        });
        it('has an undefined dtype.', function () {
            expect(defaultLayer.dtype).toBeUndefined();
        });
        it('has null initialWeights.', function () {
            expect(defaultLayer.initialWeights).toBeNull();
        });
        it('has an empty inboundNodes list.', function () {
            expect(defaultLayer.inboundNodes).toEqual([]);
        });
        it('has an empty outboundNodes list.', function () {
            expect(defaultLayer.outboundNodes).toEqual([]);
        });
        it('has an empty losses list.', function () {
            expect(defaultLayer.losses).toEqual([]);
        });
        it('has an empty updates list.', function () {
            expect(defaultLayer.updates).toEqual([]);
        });
        it('is not built (built == false).', function () {
            expect(defaultLayer.built).toEqual(false);
        });
        it('has an empty trainableWeights list.', function () {
            expect(defaultLayer.trainableWeights).toEqual([]);
        });
        it('has an empty nonTrainableWeights list.', function () {
            expect(defaultLayer.nonTrainableWeights).toEqual([]);
        });
        it('has an empty weights list.', function () {
            expect(defaultLayer.weights).toEqual([]);
        });
        it('produces a unique ID', function () {
            var secondLayer = new LayerForTest({});
            expect(defaultLayer.id).not.toEqual(secondLayer.id);
        });
        it('stateful is false by default', function () {
            var layer = new LayerForTest({});
            expect(layer.stateful).toBe(false);
        });
        it('returns null if it doesn`t support masking and no mask is passed in.', function () {
            expect(defaultLayer.computeMask([], null)).toBeNull();
        });
        it('throws exception if it doesn`t support masking and a ' +
            'mask is passed in.', function () {
            var mask = tfjs_core_1.ones([1]);
            expect(function () { return defaultLayer.computeMask([], mask); })
                .toThrowError(/does not support masking/);
        });
        it('returns the same mask passed in if it supports masking', function () {
            var mask = tfjs_core_1.ones([1]);
            defaultLayer.supportsMasking = true;
            expect(defaultLayer.computeMask([], mask)).toEqual(mask);
        });
        it('correctly generates a config for serialization', function () {
            var config = defaultLayer.getConfig();
            expect(config.name).toEqual(defaultLayer.name);
            expect(config.trainable).toEqual(defaultLayer.trainable);
            expect(config.batchInputShape).toBeUndefined();
            expect(config.dtype).toBeUndefined();
        });
    });
    describe('A layer with non-default arguments', function () {
        it('initializes layer with given name.', function () {
            var name = 'layer name';
            var layer = new LayerForTest({ name: name });
            expect(layer.name).toMatch(name);
            var config = layer.getConfig();
            expect(config.name).toEqual(layer.name);
        });
        var _loop_1 = function (trainable) {
            it('initializes layer as trainable, if specified.', function () {
                var layer = new LayerForTest({ trainable: trainable });
                expect(layer.trainable).toEqual(trainable);
                var config = layer.getConfig();
                expect(config.trainable).toEqual(layer.trainable);
            });
        };
        for (var _i = 0, _a = [true, false]; _i < _a.length; _i++) {
            var trainable = _a[_i];
            _loop_1(trainable);
        }
        var _loop_2 = function (batchInputShape) {
            it('initializes batchInputShape to layerConfig.batchInputShape.', function () {
                var layer = new LayerForTest({ batchInputShape: batchInputShape });
                expect(layer.batchInputShape).toEqual(batchInputShape);
                var config = layer.getConfig();
                expect(config.batchInputShape).toEqual(layer.batchInputShape);
            });
        };
        for (var _b = 0, _c = [[], [1]]; _b < _c.length; _b++) {
            var batchInputShape = _c[_b];
            _loop_2(batchInputShape);
        }
        it('initializes batchInputShape to layerConfig.batchInputShape even if ' +
            'layerConfig.inputShape is defined.', function () {
            var batchInputShape = [1];
            var inputShape = [2, 3];
            var layer = new LayerForTest({ batchInputShape: batchInputShape, inputShape: inputShape });
            expect(layer.batchInputShape).toEqual(batchInputShape);
        });
        var _loop_3 = function (batchSize, inputShape, expectedBatchInputShape) {
            it('initializes batchInputShape to layerConfig.inputShape.', function () {
                var layer = new LayerForTest({ batchSize: batchSize, inputShape: inputShape });
                expect(layer.batchInputShape).toEqual(expectedBatchInputShape);
            });
        };
        for (var _d = 0, _e = [
            [null, [], [null]], [null, [1], [null, 1]], [3, [], [3]],
            [3, [1], [3, 1]]
        ]; _d < _e.length; _d++) {
            var _f = _e[_d], batchSize = _f[0], inputShape = _f[1], expectedBatchInputShape = _f[2];
            _loop_3(batchSize, inputShape, expectedBatchInputShape);
        }
        it('initializes dtype to float32 if layerConfig.inputShape is set.', function () {
            var layer = new LayerForTest({ inputShape: [] });
            expect(layer.dtype).toEqual('float32');
            var config = layer.getConfig();
            expect(config.dtype).toEqual(layer.dtype);
        });
        it('initializes dtype to float32 if layerConfig.batchInputShape is set.', function () {
            var layer = new LayerForTest({ batchInputShape: [] });
            expect(layer.dtype).toEqual('float32');
        });
        it('initializes initialWeights if present.', function () {
            var weights = [tfjs_core_1.zeros([1])];
            var layer = new LayerForTest({ weights: weights });
            expect(layer.initialWeights).toEqual(weights);
        });
        it('Layer with duplicate weight names throws error', function () {
            var LayerForTest = (function (_super) {
                __extends(LayerForTest, _super);
                function LayerForTest(config) {
                    var _this = _super.call(this, config) || this;
                    _this.addWeight('foo', [1, 2], 'float32', initializers.getInitializer('zeros'));
                    _this.addWeight('foo', [2, 3], 'float32', initializers.getInitializer('zeros'));
                    return _this;
                }
                LayerForTest.className = 'LayerForTest';
                return LayerForTest;
            }(tfl.layers.Layer));
            expect(function () { return new LayerForTest({}); })
                .toThrowError(/[Dd]uplicate weight name/);
        });
    });
    it('can be set to built.', function () {
        var layer = new LayerForTest({});
        layer.built = true;
        expect(layer.built).toEqual(true);
    });
    var trainableWeights = [variables_1.zerosVariable([1])];
    var nonTrainableWeights = [variables_1.onesVariable([1])];
    it('can set trainableWeights.', function () {
        var layer = new LayerForTest({});
        layer.trainableWeights = trainableWeights;
        expect(layer.trainableWeights).toEqual(trainableWeights);
    });
    it('doesn\'t return trainableWeights if layer is not trainable, even ' +
        'if they exist', function () {
        var layer = new LayerForTest({ trainable: false });
        layer.trainableWeights = trainableWeights;
        expect(layer.trainableWeights).toEqual([]);
    });
    it('can set nonTrainableWeights.', function () {
        var layer = new LayerForTest({});
        layer.nonTrainableWeights = nonTrainableWeights;
        expect(layer.nonTrainableWeights).toEqual(nonTrainableWeights);
    });
    it('only returns nonTrainableWeights for nonTrainableWeights if the layer ' +
        'is trainable.', function () {
        var layer = new LayerForTest({ trainable: true });
        layer.trainableWeights = trainableWeights;
        layer.nonTrainableWeights = nonTrainableWeights;
        expect(layer.nonTrainableWeights).toEqual(nonTrainableWeights);
    });
    it('concats trainable and nonTrainableWeights for nonTrainableWeights if ' +
        'not trainable.', function () {
        var layer = new LayerForTest({ trainable: false });
        var expectedWeights = trainableWeights.concat(nonTrainableWeights);
        layer.trainableWeights = trainableWeights;
        layer.nonTrainableWeights = nonTrainableWeights;
        expect(layer.nonTrainableWeights).toEqual(expectedWeights);
    });
    var _loop_4 = function (trainable) {
        it('concats trainable and nonTrainableWeights for weights regardless of ' +
            'whether the layer is trainable trainable.', function () {
            var layer = new LayerForTest({ trainable: trainable });
            var expectedWeights = trainableWeights.concat(nonTrainableWeights);
            layer.trainableWeights = trainableWeights;
            layer.nonTrainableWeights = nonTrainableWeights;
            expect(layer.weights).toEqual(expectedWeights);
        });
    };
    for (var _i = 0, _a = [true, false]; _i < _a.length; _i++) {
        var trainable = _a[_i];
        _loop_4(trainable);
    }
    describe('assertInputCompatibility()', function () {
        function runAssert(layer, inputs) {
            layer.assertInputCompatibility(inputs);
        }
        var testInputs = [
            function () { return tfjs_core_1.ones([1]); }, function () { return [tfjs_core_1.ones([1])]; },
            function () { return new tfl.SymbolicTensor('float32', [1], null, [], {}); },
            function () { return [new tfl.SymbolicTensor('float32', [1], null, [], {})]; }
        ];
        var _loop_5 = function (inputs) {
            it('doesn\'t raise an exception if no inputSpec is provided.', function () {
                var layer = new LayerForTest({});
                runAssert(layer, inputs());
            });
            it('doesn\'t raise exception if number of inputs == number of ' +
                'inputSpecs.', function () {
                var inputSpecs = [new topology_1.InputSpec({})];
                var layer = new LayerForTest({});
                layer.inputSpec = inputSpecs;
                expect(function () { return runAssert(layer, inputs()); }).not.toThrowError();
            });
            it('throws exception if number of inputs != number of inputSpecs.', function () {
                var inputSpecs = [new topology_1.InputSpec({}), new topology_1.InputSpec({})];
                var layer = new LayerForTest({});
                layer.inputSpec = inputSpecs;
                expect(function () { return runAssert(layer, inputs()); })
                    .toThrowError(/expects [0-9]+ inputs/);
            });
            it('doesn\'t raise exception if inputs\' ndim == inputSpecs.ndim.', function () {
                var inputSpecs = [new topology_1.InputSpec({ ndim: 1 })];
                var layer = new LayerForTest({});
                layer.inputSpec = inputSpecs;
                expect(function () { return runAssert(layer, inputs()); }).not.toThrowError();
            });
            it('throws exception if inputs\' ndim != inputSpecs.ndim.', function () {
                var inputSpecs = [new topology_1.InputSpec({ ndim: 2 })];
                var layer = new LayerForTest({});
                layer.inputSpec = inputSpecs;
                expect(function () { return runAssert(layer, inputs()); }).toThrowError(/expected ndim=/);
            });
            it('doesn\'t raise exception if inputs\' ndim <= inputSpecs.maxNdim.', function () {
                var inputSpecs = [new topology_1.InputSpec({ maxNDim: 1 })];
                var layer = new LayerForTest({});
                layer.inputSpec = inputSpecs;
                expect(function () { return runAssert(layer, inputs()); }).not.toThrowError();
            });
            it('throws exception if inputs\' ndim > inputSpecs.maxNdim.', function () {
                var inputSpecs = [new topology_1.InputSpec({ maxNDim: 0 })];
                var layer = new LayerForTest({});
                layer.inputSpec = inputSpecs;
                expect(function () { return runAssert(layer, inputs()); })
                    .toThrowError(/expected max_ndim=/);
            });
            it('doesn\'t raise exception if inputs\' ndim >= inputSpecs.minNdim.', function () {
                var inputSpecs = [new topology_1.InputSpec({ minNDim: 1 })];
                var layer = new LayerForTest({});
                layer.inputSpec = inputSpecs;
                expect(function () { return runAssert(layer, inputs()); }).not.toThrowError();
            });
            it('throws exception if inputs\' ndim < inputSpecs.minNdim.', function () {
                var inputSpecs = [new topology_1.InputSpec({ minNDim: 2 })];
                var layer = new LayerForTest({});
                layer.inputSpec = inputSpecs;
                expect(function () { return runAssert(layer, inputs()); })
                    .toThrowError(/expected min_ndim=/);
            });
            it('doesn\'t raise exception if inputs\' dtype == inputSpecs.dtype.', function () {
                var inputSpecs = [new topology_1.InputSpec({ dtype: 'float32' })];
                var layer = new LayerForTest({});
                layer.inputSpec = inputSpecs;
                expect(function () { return runAssert(layer, inputs()); }).not.toThrowError();
            });
            it('doesn\'t raise exception if inputs\' dimensions == inputSpecs.axes.', function () {
                var inputSpecs = [new topology_1.InputSpec({ axes: { 0: 1 } })];
                var layer = new LayerForTest({});
                layer.inputSpec = inputSpecs;
                expect(function () { return runAssert(layer, inputs()); }).not.toThrowError();
            });
            it('throws exception if inputs\' dimensions != inputSpecs.axes.', function () {
                var inputSpecs = [new topology_1.InputSpec({ axes: { 0: 2 } })];
                var layer = new LayerForTest({});
                layer.inputSpec = inputSpecs;
                expect(function () { return runAssert(layer, inputs()); }).toThrowError(/expected axis/);
            });
            it('throws exception if inputs\' dimensions don\'t have the same ' +
                'number of inputSpecs.axes.', function () {
                var inputSpecs = [new topology_1.InputSpec({ axes: { 0: 1, 2: 1 } })];
                var layer = new LayerForTest({});
                layer.inputSpec = inputSpecs;
                expect(function () { return runAssert(layer, inputs()); })
                    .toThrowError(/expected axis/);
            });
            it('doesn\'t raise exception if inputs\' shape == inputSpecs.shape.', function () {
                var inputSpecs = [new topology_1.InputSpec({ shape: [1] })];
                var layer = new LayerForTest({});
                layer.inputSpec = inputSpecs;
                expect(function () { return runAssert(layer, inputs()); }).not.toThrowError();
            });
            it('throws exception if inputs\' shape != inputSpecs.shape.', function () {
                var inputSpecs = [new topology_1.InputSpec({ shape: [2] })];
                var layer = new LayerForTest({});
                layer.inputSpec = inputSpecs;
                expect(function () { return runAssert(layer, inputs()); }).toThrowError(/expected shape/);
            });
        };
        for (var _i = 0, testInputs_1 = testInputs; _i < testInputs_1.length; _i++) {
            var inputs = testInputs_1[_i];
            _loop_5(inputs);
        }
    });
    describe('apply() passed 1 SymbolicTensor', function () {
        var firstLayer = new LayerForTest({ name: 'firstLayer' });
        var secondLayer = new LayerForTest({ name: 'secondLayer' });
        var callArgs = { a: 1 };
        var singleSymbolicTensor = new tfl.SymbolicTensor('float32', [1], firstLayer, [], {});
        var returnedTensor = secondLayer.apply(singleSymbolicTensor, callArgs);
        it('returns a SymbolicTensor.', function () {
            expect(returnedTensor instanceof tfl.SymbolicTensor).toBe(true);
        });
        it('returns a SymbolicTensor with a reference to the source layer.', function () {
            expect(returnedTensor.sourceLayer).toEqual(secondLayer);
        });
        it('returns a SymbolicTensor with a reference to the inputs passed ' +
            'to apply().', function () {
            expect(returnedTensor.inputs).toEqual([singleSymbolicTensor]);
            expect(returnedTensor.callArgs).toEqual(callArgs);
        });
        it('returns a SymbolicTensor with nodeIndex and tensorIndex set.', function () {
            expect(returnedTensor.nodeIndex).toBeDefined();
            expect(returnedTensor.tensorIndex).toBeDefined();
        });
        it('returns a SymbolicTensor with the name set.', function () {
            expect(returnedTensor.name).toMatch(/secondLayer/);
        });
        it('is built.', function () {
            expect(secondLayer.built).toBe(true);
        });
        it('Incompatible inputShape leads to warning', function () {
            var recordedWarnMessage;
            spyOn(console, 'warn')
                .and.callFake(function (message) { return recordedWarnMessage = message; });
            var layer1 = tfl.layers.dense({ units: 2, inputShape: [5] });
            layer1.apply(tfl.input({ shape: [4] }));
            expect(recordedWarnMessage)
                .toMatch(/shape of the input tensor .*null,4.* not match .*null,5.*/);
        });
        it('Incompatible inputShape leads to warning: batchInputShape', function () {
            var recordedWarnMessage;
            spyOn(console, 'warn')
                .and.callFake(function (message) { return recordedWarnMessage = message; });
            var layer1 = tfl.layers.dense({ units: 2, batchInputShape: [2, 3, 5] });
            layer1.apply(tfl.input({ shape: [4, 5] }));
            expect(recordedWarnMessage)
                .toMatch(/shape of the input tensor .*null,4,5.* not match .*2,3,5.*/);
        });
        it('Incompatible inputShape rank leads to warning', function () {
            var recordedWarnMessage;
            spyOn(console, 'warn')
                .and.callFake(function (message) { return recordedWarnMessage = message; });
            var layer1 = tfl.layers.dense({ units: 2, inputShape: [5] });
            layer1.apply(tfl.input({ shape: [4, 3] }));
            expect(recordedWarnMessage)
                .toMatch(/rank .*null,4,3.* does not match .*null,5.*/);
        });
        it('Incompatible inputShape rank leads to warning: batchInputShape', function () {
            var recordedWarnMessage;
            spyOn(console, 'warn')
                .and.callFake(function (message) { return recordedWarnMessage = message; });
            var layer1 = tfl.layers.dense({ units: 2, batchInputShape: [3, 5] });
            layer1.apply(tfl.input({ shape: [4, 3] }));
            expect(recordedWarnMessage)
                .toMatch(/rank .*null,4,3.* does not match .*3,5.*/);
        });
        it('Compatible inputShape leads to NO warning', function () {
            var recordedWarnMessage;
            spyOn(console, 'warn')
                .and.callFake(function (message) { return recordedWarnMessage = message; });
            var layer1 = tfl.layers.dense({ units: 2, inputShape: [5] });
            layer1.apply(tfl.input({ shape: [5] }));
            expect(recordedWarnMessage).toEqual(undefined);
        });
    });
    describe('apply() passed >1 SymbolicTensor', function () {
        it('throws an exception for multiple symbolic inputs.', function () {
            var firstLayer = new LayerForTest({ name: 'first layer' });
            var secondLayer = new LayerForTest({ name: 'second layer' });
            var symbolicTensorList = [
                new tfl.SymbolicTensor('float32', [1], firstLayer, [], {}, 'first_symbolic_tensor'),
                new tfl.SymbolicTensor('float32', [1], firstLayer, [], {}, 'second_symbolic_tensor')
            ];
            expect(function () { return secondLayer.apply(symbolicTensorList); }).toThrowError();
        });
    });
    describe('apply() passed SymbolicTensor and Tensor', function () {
        it('throws an exception.', function () {
            var layer = new LayerForTest({});
            var inputs = [
                new tfl.SymbolicTensor('float32', [1], null, [], {}, 'first_symbolic_tensor'),
                tfjs_core_1.ones([1])
            ];
            expect(function () { return layer.apply(inputs); })
                .toThrowError(/must be all SymbolicTensors or all Tensors/);
        });
    });
    it('apply() returns multiple symbolic tensors for multiple ' +
        'output shapes', function () {
        var layer = new LayerForTest({});
        var outputShapes = [[1], [2, 3]];
        var input = new tfl.SymbolicTensor('float32', [1], null, [], {});
        spyOn(layer, 'computeOutputShape').and.callFake(function () {
            return outputShapes;
        });
        var results = layer.apply(input);
        expect(results.length).toEqual(2);
        expect(results.map(function (x) { return x.shape; })).toEqual(outputShapes);
        expect(results.map(function (x) { return x.outputTensorIndex; })).toEqual([0, 1]);
    });
    describe('apply() passed 1+ Tensors', function () {
        it('returns new values for output if the same as the input.', function () {
            var anArray = tfjs_core_1.ones([1]);
            for (var _i = 0, _a = [anArray, [anArray, anArray]]; _i < _a.length; _i++) {
                var inputs = _a[_i];
                var layer = new LayerForTest({});
                var result = layer.apply(inputs);
                expect(result instanceof tfjs_core_1.Tensor || (result[0] instanceof tfjs_core_1.Tensor))
                    .toBe(true);
                expect(layer.built).toBe(true);
                if (result instanceof Array) {
                    var inputArray = inputs;
                    for (var i = 0; i < result.length; i++) {
                        test_utils_1.expectTensorsClose(result[i], inputArray[i]);
                    }
                }
                else {
                    test_utils_1.expectTensorsClose(result, inputs);
                }
                expect(result === inputs).toBe(false);
            }
        });
    });
    describe('initialized with weights at construction time', function () {
        it('sets those weights after calling apply().', function () {
            var initialWeights = tfjs_core_1.eye(2);
            var arrayInput = tfjs_core_1.zeros([1]);
            var symbolicInput = new tfl.SymbolicTensor('float32', [1], null, [], {});
            var _loop_6 = function (inputs) {
                var layer = new LayerForTest({ weights: [initialWeights] });
                spyOn(layer, 'build').and.callFake(function () {
                    layer.built = true;
                    layer.trainableWeights = [new variables_1.LayerVariable(tfjs_core_1.zeros([2, 2]))];
                });
                expect(layer.weights.length).toEqual(0);
                layer.apply(inputs);
                expect(layer.weights.length).toEqual(1);
                test_utils_1.expectTensorsClose(layer.weights[0].read(), initialWeights);
            };
            for (var _i = 0, _a = [arrayInput, symbolicInput]; _i < _a.length; _i++) {
                var inputs = _a[_i];
                _loop_6(inputs);
            }
        });
    });
    describe('apply() (nodes)', function () {
        it('doesn\'t change inboundNodes or outboundNodes when called with ' +
            'concrete input', function () {
            var layer = new LayerForTest({});
            expect(layer.inboundNodes.length).toEqual(0);
            expect(layer.outboundNodes.length).toEqual(0);
            layer.apply(tfjs_core_1.eye(1));
            expect(layer.inboundNodes.length).toEqual(0);
            expect(layer.outboundNodes.length).toEqual(0);
        });
        it('changes inboundNodes and outboundNodes when called with ' +
            'symbolic input', function () {
            var layer = new LayerForTest({});
            var input = new tfl.SymbolicTensor('float32', [1], null, [], {});
            expect(layer.inboundNodes.length).toEqual(0);
            expect(layer.outboundNodes.length).toEqual(0);
            layer.apply(input);
            expect(layer.inboundNodes.length).toEqual(1);
            expect(layer.outboundNodes.length).toEqual(0);
            expect(layer.inboundNodes[0].outboundLayer).toEqual(layer);
        });
        it('updates inbound and outboundNodes when there are multiple layers', function () {
            var firstLayer = new LayerForTest({ name: 'first_layer' });
            var secondLayer = new LayerForTest({ name: 'second_layer' });
            var initialInput = new tfl.SymbolicTensor('float32', [1], null, [], {});
            var firstOutput = firstLayer.apply(initialInput);
            secondLayer.apply(firstOutput);
            expect(firstLayer.inboundNodes.length).toEqual(1);
            expect(firstLayer.outboundNodes.length).toEqual(1);
            expect(secondLayer.inboundNodes.length).toEqual(1);
            expect(secondLayer.outboundNodes.length).toEqual(0);
            expect(firstLayer.outboundNodes[0].outboundLayer).toEqual(secondLayer);
        });
    });
    describe('Layer.outputShape', function () {
        it('Layers with one output', function () {
            var layer = tfl.layers.dense({ units: 3 });
            layer.apply(new tfl.SymbolicTensor('float32', [null, 4], null, [], {}));
            expect(layer.outputShape).toEqual([null, 3]);
        });
        it('Layers with two outputs', function () {
            var layer = tfl.layers.simpleRNN({ units: 3, returnState: true });
            layer.apply(new tfl.SymbolicTensor('float32', [null, 4, 5], null, [], {}));
            expect(layer.outputShape).toEqual([[null, 3], [null, 3]]);
        });
        it('Layers with two inboundNodes of the same outputShape', function () {
            var layer = tfl.layers.dense({ units: 3 });
            layer.apply(new tfl.SymbolicTensor('float32', [null, 4], null, [], {}));
            layer.apply(new tfl.SymbolicTensor('float32', [null, 4], null, [], {}));
            expect(layer.inboundNodes.length).toEqual(2);
            expect(layer.outputShape).toEqual([null, 3]);
        });
        it('Layers with two inboundNodes of different outputShapes', function () {
            var layer = tfl.layers.dense({ units: 3 });
            layer.apply(new tfl.SymbolicTensor('float32', [null, 5, 4], null, [], {}));
            layer.apply(new tfl.SymbolicTensor('float32', [null, 6, 4], null, [], {}));
            expect(layer.inboundNodes.length).toEqual(2);
            expect(function () { return layer.outputShape; })
                .toThrowError(/has multiple inbound nodes/);
        });
        it('Unbuilt layer throws Error', function () {
            var layer = tfl.layers.dense({ units: 3 });
            expect(function () { return layer.outputShape; }).toThrowError(/has never been called/);
        });
    });
    describe('Layer.countParams', function () {
        it('Layers with weights', function () {
            var units = 3;
            var inputSize = 4;
            var layer = tfl.layers.dense({ units: units });
            layer.apply(tfjs_core_1.zeros([1, inputSize]));
            var numParams = layer.countParams();
            expect(numParams).toEqual(units * inputSize + units);
        });
        it('Layer without weights', function () {
            var layer = tfl.layers.flatten();
            layer.apply(tfjs_core_1.zeros([2, 2, 2]));
            var numParams = layer.countParams();
            expect(numParams).toEqual(0);
        });
    });
    describe('setWeights', function () {
        it('throws exception if weights are not the same length ' +
            'as existing weights', function () {
            var layer = new LayerForTest({});
            layer.trainableWeights = [new variables_1.LayerVariable(tfjs_core_1.zeros([2, 2]))];
            var onesTensor = tfjs_core_1.ones([1]);
            expect(function () { return layer.setWeights([
                onesTensor, onesTensor
            ]); }).toThrowError(/with a weight list of length/);
        });
        it('throws exception if weights are not the same shape ' +
            'as existing weights', function () {
            var layer = new LayerForTest({});
            var onesTensor = tfjs_core_1.ones([1]);
            layer.trainableWeights = [new variables_1.LayerVariable(tfjs_core_1.zeros([2, 2]))];
            expect(function () { return layer.setWeights([onesTensor]); })
                .toThrowError(/not compatible with provided weight shape/);
        });
        it('updates weights.', function () {
            var layer = new LayerForTest({});
            var onesTensor = tfjs_core_1.ones([1]);
            layer.trainableWeights = [new variables_1.LayerVariable(tfjs_core_1.zeros([1]))];
            layer.setWeights([onesTensor]);
            test_utils_1.expectTensorsClose(layer.trainableWeights[0].read(), onesTensor);
        });
    });
    describe('computeOutputShape()', function () {
        it('returns the inputShape in the base class', function () {
            var layer = new LayerForTest({});
            var shape = [1];
            expect(layer.computeOutputShape(shape)).toEqual(shape);
        });
    });
    describe('input and output properties: ', function () {
        var input;
        var layer;
        var output;
        beforeEach(function () {
            input =
                new tfl.SymbolicTensor('float32', [1], null, [], {}, 'firstInput');
            layer = new LayerForTest({});
            output = layer.apply(input);
        });
        it('input retrieves layer\'s inputs.', function () {
            expect(layer.input).toEqual(input);
        });
        it('input retrieves layer\'s outputs.', function () {
            expect(layer.output).toEqual(output);
        });
        it('input throws exception if there is more than one input', function () {
            var secondInput = new tfl.SymbolicTensor('float32', [1], null, [], {}, 'secondInput');
            layer.apply(secondInput);
            expect(function () { return layer.input; }).toThrowError(/"layer input" is ill-defined/);
        });
        it('output throws exception if there is more than one output', function () {
            var secondInput = new tfl.SymbolicTensor('float32', [1], null, [], {}, 'secondInput');
            layer.apply(secondInput);
            expect(function () { return layer.output; }).toThrowError(/"layer output" is ill-defined/);
        });
    });
    describe('getInputAt and getOutputAt: ', function () {
        var input;
        var layer;
        var output;
        beforeEach(function () {
            input =
                new tfl.SymbolicTensor('float32', [1], null, [], {}, 'firstInput');
            layer = new LayerForTest({});
            output = layer.apply(input);
        });
        it('getInputAt() retrieves layer\'s inputs.', function () {
            expect(layer.getInputAt(0)).toEqual(input);
        });
        it('getOutputAt() retrieves layer\'s outputs.', function () {
            expect(layer.getOutputAt(0)).toEqual(output);
        });
        it('getInputAt() throws exception ask for incorrect index.', function () {
            expect(function () { return layer.getInputAt(1); })
                .toThrowError(/Asked to get input at node 1, but/);
        });
        it('getOutputAt() throws exception ask for incorrect index.', function () {
            expect(function () { return layer.getOutputAt(1); })
                .toThrowError(/Asked to get output at node 1, but/);
        });
    });
});
test_utils_1.describeMathCPUAndGPU('Layer-dispose', function () {
    it('Dispose Dense Layer before build leads to Error', function () {
        var dense = tfl.layers.dense({ units: 1, inputShape: [4] });
        expect(function () { return dense.dispose(); }).toThrowError(/has not been built/);
    });
    it('Dispose Dense Layer after one tensor call frees memory', function () {
        var dense = tfl.layers.dense({ units: 1, inputShape: [4] });
        dense.apply(tfjs_core_1.zeros([2, 4]));
        var numTensors0 = tfjs_core_1.memory().numTensors;
        var result = dense.dispose();
        expect(result.refCountAfterDispose).toEqual(0);
        expect(result.numDisposedVariables).toEqual(2);
        expect(tfjs_core_1.memory().numTensors).toEqual(numTensors0 - 2);
    });
    it('Symbolic apply() call after Dense disposal leads to Error', function () {
        var dense = tfl.layers.dense({ units: 1, inputShape: [4] });
        dense.apply(tfjs_core_1.zeros([2, 4]));
        var result = dense.dispose();
        expect(result.refCountAfterDispose).toEqual(0);
        expect(result.numDisposedVariables).toEqual(2);
        expect(function () { return dense.apply(new tfl.SymbolicTensor('float32', [2, 4], null, [], {})); })
            .toThrowError(/Layer .* is already disposed/);
    });
    it('Non-symbolic apply() call after Dense disposal leads to Error', function () {
        var dense = tfl.layers.dense({ units: 1, inputShape: [4] });
        dense.apply(tfjs_core_1.zeros([2, 4]));
        dense.dispose();
        expect(function () { return dense.apply(tfjs_core_1.ones([2, 4])); })
            .toThrowError(/Layer .* is already disposed/);
    });
    it('Calling defRec() repeatedly for two-Node Layer frees memory', function () {
        var dense = tfl.layers.dense({ units: 1, inputShape: [4] });
        dense.apply(new tfl.SymbolicTensor('float32', [2, 4], null, [], {}));
        dense.apply(new tfl.SymbolicTensor('float32', [2, 4], null, [], {}));
        var numTensors0 = tfjs_core_1.memory().numTensors;
        var result1 = dense.dispose();
        expect(tfjs_core_1.memory().numTensors).toEqual(numTensors0);
        expect(result1.refCountAfterDispose).toEqual(1);
        expect(result1.numDisposedVariables).toEqual(0);
        var result2 = dense.dispose();
        expect(tfjs_core_1.memory().numTensors).toEqual(numTensors0 - 2);
        expect(result2.refCountAfterDispose).toEqual(0);
        expect(result2.numDisposedVariables).toEqual(2);
    });
    it('Calling dispose on already-disposed Layer leads to Error', function () {
        var dense = tfl.layers.dense({ units: 1, inputShape: [4] });
        dense.apply(tfjs_core_1.zeros([2, 4]));
        dense.dispose();
        expect(function () { return dense.dispose(); }).toThrowError(/Layer .* is already disposed/);
    });
    it('Symbolic apply() call after Flatten disposal leads to Error', function () {
        var dense = tfl.layers.flatten();
        dense.apply(tfjs_core_1.zeros([2, 3, 4]));
        dense.dispose();
        expect(function () { return dense.apply(new tfl.SymbolicTensor('float32', [2, 4], null, [], {})); })
            .toThrowError(/Layer .* is already disposed/);
    });
    it('Non-symbolic apply() call after Flatten disposal leads to Error', function () {
        var dense = tfl.layers.flatten();
        dense.apply(tfjs_core_1.zeros([2, 3, 4]));
        dense.dispose();
        expect(function () { return dense.apply(tfjs_core_1.zeros([2, 3, 4])); })
            .toThrowError(/Layer .* is already disposed/);
    });
    it('dispose() call works on Input Layer', function () {
        var input = tfl.layers.input({ shape: [2, 3] });
        var output = tfl.layers.reshape({ targetShape: [3, 2] }).apply(input);
        var model = tfl.model({ inputs: [input], outputs: [output] });
        var result = model.dispose();
        expect(result.numDisposedVariables).toEqual(0);
        expect(function () { return model.predict(tfjs_core_1.zeros([1, 2, 3])); })
            .toThrowError(/already disposed/);
    });
});
test_utils_1.describeMathCPUAndGPU('loadWeightsFromJson', function () {
    var inputTensor = tfl.layers.input({ shape: [3], name: 'inputLayer', dtype: 'float32' });
    it('One layer', function () {
        var denseLayer = tfl.layers.dense({ units: 2, useBias: true, name: 'denseLayer' });
        denseLayer.apply(inputTensor);
        var weightsJSON = {
            'keras_version': '2.1.2',
            'backend': 'tensorflow',
            'weights': {
                'denseLayer': [
                    {
                        'name': 'denseLayer/kernel:0',
                        'dtype': 'float32',
                        'shape': [3, 2],
                        'value': [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
                    },
                    {
                        'name': 'denseLayer/bias:0',
                        'dtype': 'float32',
                        'shape': [2],
                        'value': [-0.1, -0.2],
                    },
                ],
            },
        };
        container_1.loadWeightsFromJson(weightsJSON, [denseLayer]);
        test_utils_1.expectTensorsClose(denseLayer.apply(tfjs_core_1.tensor2d([[1, 1, 1]], [1, 3])), tfjs_core_1.tensor2d([[0.8, 1.0]], [1, 2]));
    });
    it('Two layers', function () {
        var denseLayer1 = tfl.layers.dense({ units: 2, useBias: true, name: 'denseLayer1' });
        var denseLayer2 = tfl.layers.dense({ units: 1, useBias: false, name: 'denseLayer2' });
        denseLayer2.apply(denseLayer1.apply(inputTensor));
        var weightsJSON = {
            'keras_version': '2.1.2',
            'backend': 'tensorflow',
            'weights': {
                'denseLayer1': [
                    {
                        'name': 'denseLayer1/kernel:0',
                        'dtype': 'float32',
                        'shape': [3, 2],
                        'value': [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
                    },
                    {
                        'name': 'denseLayer1/bias:0',
                        'dtype': 'float32',
                        'shape': [2],
                        'value': [-0.1, -0.2],
                    },
                ],
                'denseLayer2': [
                    {
                        'name': 'denseLayer2/kernel:0',
                        'dtype': 'float32',
                        'shape': [2, 1],
                        'value': [[1.2], [1.3]],
                    },
                ],
            },
        };
        container_1.loadWeightsFromJson(weightsJSON, [denseLayer1, denseLayer2]);
        test_utils_1.expectTensorsClose(denseLayer2.apply(denseLayer1.apply(tfjs_core_1.tensor2d([[1, 1, 1]], [1, 3]))), tfjs_core_1.tensor2d([[2.26]], [1, 1]));
    });
    it('Missing weights for a layer', function () {
        var denseLayer = tfl.layers.dense({ units: 2, useBias: true, name: 'denseLayer' });
        denseLayer.apply(inputTensor);
        var weightsJSON = {
            'keras_version': '2.1.2',
            'backend': 'tensorflow',
            'weights': {},
        };
        expect(function () {
            container_1.loadWeightsFromJson(weightsJSON, [denseLayer]);
        })
            .toThrowError(/Layer.*denseLayer.*expects 2 weight.*but.*have 0 element.*/);
    });
    it('Missing a single weight', function () {
        var denseLayer = tfl.layers.dense({ units: 2, useBias: true, name: 'denseLayer' });
        denseLayer.apply(inputTensor);
        var weightsJSON = {
            'keras_version': '2.1.2',
            'backend': 'tensorflow',
            'weights': {
                'denseLayer': [
                    {
                        'name': 'denseLayer1/kernel:0',
                        'dtype': 'float32',
                        'shape': [3, 2],
                        'value': [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
                    },
                    {
                        'name': 'denseLayer1/bias:0',
                        'dtype': 'float32',
                        'shape': [1],
                        'value': [-0.1],
                    },
                ],
            }
        };
        expect(function () {
            container_1.loadWeightsFromJson(weightsJSON, [denseLayer]);
        }).toThrowError(/Shape mismatch.*\[2\] vs\. \[1\].*/);
    });
    it('Shape mismatch in a single weight', function () {
        var denseLayer = tfl.layers.dense({ units: 2, useBias: true, name: 'denseLayer' });
        denseLayer.apply(inputTensor);
        var weightsJSON = {
            'keras_version': '2.1.2',
            'backend': 'tensorflow',
            'weights': {
                'denseLayer': [
                    {
                        'name': 'denseLayer1/kernel:0',
                        'dtype': 'float32',
                        'shape': [3, 2],
                        'value': [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
                    },
                ],
            }
        };
        expect(function () {
            container_1.loadWeightsFromJson(weightsJSON, [denseLayer]);
        })
            .toThrowError(/Layer.*denseLayer.*expects 2 weight.*but.*have 1 element.*/);
    });
    it('skipMismatch=true tolerates a single missing weight', function () {
        var denseLayer = tfl.layers.dense({ units: 2, useBias: true, name: 'denseLayer' });
        denseLayer.apply(inputTensor);
        var weightsJSON = {
            'keras_version': '2.1.2',
            'backend': 'tensorflow',
            'weights': {
                'denseLayer': [
                    {
                        'name': 'denseLayer1/kernel:0',
                        'dtype': 'float32',
                        'shape': [3, 2],
                        'value': [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
                    },
                ],
            }
        };
        spyOn(console, 'warn');
        container_1.loadWeightsFromJson(weightsJSON, [denseLayer], true);
        expect(console.warn).toHaveBeenCalled();
        test_utils_1.expectTensorsClose(denseLayer.apply(tfjs_core_1.tensor2d([[1, 1, 1]], [1, 3])), tfjs_core_1.tensor2d([[0.9, 1.2]], [1, 2]));
    });
});
test_utils_1.describeMathCPUAndGPU('loadWeightsFromNamedTensorMap', function () {
    var inputTensor = tfl.layers.input({ shape: [3], name: 'inputLayer', dtype: 'float32' });
    it('One layer', function () {
        var denseLayer = tfl.layers.dense({ units: 2, useBias: true, name: 'dense_layer' });
        denseLayer.apply(inputTensor);
        var namedWeightsMap = {};
        namedWeightsMap[denseLayer.weights[0].originalName] =
            tfjs_core_1.tensor2d([1, 2, 3, 4, 5, 6], [3, 2]);
        namedWeightsMap[denseLayer.weights[1].originalName] = tfjs_core_1.tensor1d([10, 20]);
        container_1.loadWeightsFromNamedTensorMap(namedWeightsMap, [denseLayer]);
        test_utils_1.expectTensorsClose(denseLayer.weights[0].read(), tfjs_core_1.tensor2d([1, 2, 3, 4, 5, 6], [3, 2]));
        test_utils_1.expectTensorsClose(denseLayer.weights[1].read(), tfjs_core_1.tensor1d([10, 20]));
    });
    it('Mismatching shape throws an error even in non-strict mode', function () {
        var denseLayer = tfl.layers.dense({ units: 2, useBias: true, name: 'dense_layer' });
        denseLayer.apply(inputTensor);
        var namedWeightsMap = {};
        namedWeightsMap[denseLayer.weights[0].originalName] =
            tfjs_core_1.tensor2d([1, 2, 3, 4, 5, 6, 7, 8], [4, 2]);
        namedWeightsMap[denseLayer.weights[1].originalName] = tfjs_core_1.tensor1d([10, 20]);
        expect(function () {
            return container_1.loadWeightsFromNamedTensorMap(namedWeightsMap, [denseLayer], false);
        })
            .toThrowError('Shape mismatch: [3,2] vs. [4,2]');
    });
    it('Extra weights leads to error', function () {
        var denseLayer = tfl.layers.dense({ units: 2, useBias: true, name: 'dense_layer' });
        denseLayer.apply(inputTensor);
        var namedWeightsMap = {};
        namedWeightsMap[denseLayer.weights[0].originalName] =
            tfjs_core_1.tensor2d([1, 2, 3, 4, 5, 6], [3, 2]);
        namedWeightsMap[denseLayer.weights[1].originalName] = tfjs_core_1.tensor1d([10, 20]);
        namedWeightsMap['extra'] = tfjs_core_1.tensor1d([10, 20]);
        expect(function () { return container_1.loadWeightsFromNamedTensorMap(namedWeightsMap, [denseLayer]); })
            .toThrowError(/Provided weight data has no target variable: extra/);
    });
    it('Extra weights are allowed in non-strict mode', function () {
        var denseLayer = tfl.layers.dense({ units: 2, useBias: true, name: 'dense_layer' });
        denseLayer.apply(inputTensor);
        var namedWeightsMap = {};
        namedWeightsMap[denseLayer.weights[0].originalName] =
            tfjs_core_1.tensor2d([1, 2, 3, 4, 5, 6], [3, 2]);
        namedWeightsMap[denseLayer.weights[1].originalName] = tfjs_core_1.tensor1d([10, 20]);
        namedWeightsMap['extra'] = tfjs_core_1.tensor1d([10, 20]);
        container_1.loadWeightsFromNamedTensorMap(namedWeightsMap, [denseLayer], false);
        test_utils_1.expectTensorsClose(denseLayer.weights[0].read(), tfjs_core_1.tensor2d([1, 2, 3, 4, 5, 6], [3, 2]));
        test_utils_1.expectTensorsClose(denseLayer.weights[1].read(), tfjs_core_1.tensor1d([10, 20]));
    });
    it('Unset weights leads to error', function () {
        var denseLayer = tfl.layers.dense({ units: 2, useBias: true, name: 'dense_layer' });
        denseLayer.apply(inputTensor);
        var namedWeightsMap = {};
        namedWeightsMap[denseLayer.weights[0].originalName] =
            tfjs_core_1.tensor2d([1, 2, 3, 4, 5, 6], [3, 2]);
        expect(function () { return container_1.loadWeightsFromNamedTensorMap(namedWeightsMap, [denseLayer]); })
            .toThrowError(/1 of 2 weights are not set: .*bias.*/);
    });
    it('Unset weights are allowed in non-strict mode', function () {
        var denseLayer = tfl.layers.dense({ units: 2, useBias: true, name: 'dense_layer' });
        denseLayer.apply(inputTensor);
        var namedWeightsMap = {};
        namedWeightsMap[denseLayer.weights[0].originalName] =
            tfjs_core_1.tensor2d([1, 2, 3, 4, 5, 6], [3, 2]);
        container_1.loadWeightsFromNamedTensorMap(namedWeightsMap, [denseLayer], false);
    });
});
//# sourceMappingURL=topology_test.js.map