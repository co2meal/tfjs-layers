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
var test_utils_1 = require("../utils/test_utils");
var input_layer_1 = require("./input_layer");
test_utils_1.describeMathCPU('InputLayer', function () {
    it('when initialized to its defaults throws an exception', function () {
        expect(function () { return tfl.layers.inputLayer({}); })
            .toThrowError(/InputLayer should be passed either/);
    });
    describe('initialized with only an inputShape', function () {
        var inputShape = [1];
        var inputLayer = tfl.layers.inputLayer({ inputShape: inputShape });
        it('is not trainable.', function () {
            expect(inputLayer.trainable).toBe(false);
        });
        it('is built.', function () {
            expect(inputLayer.built).toBe(true);
        });
        it('is not sparse.', function () {
            expect(inputLayer.sparse).toBe(false);
        });
        it('automatically assigns a name.', function () {
            expect(inputLayer.name).toMatch(/^input.*$/);
        });
        it('creates a batchInputShape of [null].concat(inputShape).', function () {
            expect(inputLayer.batchInputShape).toEqual([null].concat(inputShape));
        });
        it('has no outboundNodes', function () {
            expect(inputLayer.outboundNodes.length).toEqual(0);
        });
        it('has one inboundNode', function () {
            expect(inputLayer.inboundNodes.length).toEqual(1);
        });
        describe('creates an inbound Node', function () {
            var inboundNode = inputLayer.inboundNodes[0];
            it('with no inboundLayers, nodeIndices, or tensorIndices', function () {
                expect(inboundNode.inboundLayers.length).toEqual(0);
                expect(inboundNode.nodeIndices.length).toEqual(0);
                expect(inboundNode.tensorIndices.length).toEqual(0);
            });
            it('with [null] inputMasks and outputMasks', function () {
                expect(inboundNode.inputMasks).toEqual([null]);
                expect(inboundNode.outputMasks).toEqual([null]);
            });
            it('with equal inputShapes and outputShapes', function () {
                expect(inboundNode.inputShapes).toEqual(inboundNode.outputShapes);
                expect(inboundNode.inputShapes).toEqual([[null].concat(inputShape)]);
            });
            describe('with a SymbolicTensor', function () {
                var symbolicTensor = inboundNode.inputTensors[0];
                it('that is defined.', function () {
                    expect(symbolicTensor instanceof tfl.SymbolicTensor).toBe(true);
                });
                it('assigned to both the input and outputTensors.', function () {
                    expect(inboundNode.inputTensors.length).toEqual(1);
                    expect(inboundNode.outputTensors.length).toEqual(1);
                    expect(inboundNode.inputTensors).toEqual(inboundNode.outputTensors);
                });
                it('with a node and tensorIndex of 0.', function () {
                    expect(symbolicTensor.nodeIndex).toEqual(0);
                    expect(symbolicTensor.tensorIndex).toEqual(0);
                });
                it('with a sourceLayer of the inputLayer.', function () {
                    expect(symbolicTensor.sourceLayer).toEqual(inputLayer);
                });
                it('with a name matching the inputLayer name.', function () {
                    expect(symbolicTensor.name).toEqual(inputLayer.name);
                });
                it('with a dtype equal to the inputLayer.', function () {
                    expect(symbolicTensor.dtype).toEqual(inputLayer.dtype);
                });
                it('with a shape matching the inputLayer.batchInputShape', function () {
                    expect(symbolicTensor.shape).toEqual(inputLayer.batchInputShape);
                });
            });
        });
    });
    it('throws an exception if both inputShape and batchInputShape ' +
        'are specified during initialization.', function () {
        expect(function () { return tfl.layers.inputLayer({ inputShape: [1], batchInputShape: [1] }); })
            .toThrowError(/Only provide the inputShape OR batchInputShape/);
    });
    var _loop_1 = function (batchSize) {
        it('initializes with batchSize when inputShape specified', function () {
            var inputShape = [1];
            var inputLayer = tfl.layers.inputLayer({ inputShape: inputShape, batchSize: batchSize });
            expect(inputLayer.batchInputShape).toEqual([
                batchSize
            ].concat(inputShape));
        });
    };
    for (var _i = 0, _a = [null, 5]; _i < _a.length; _i++) {
        var batchSize = _a[_i];
        _loop_1(batchSize);
    }
    it('initializes with batchInputShape if specified.', function () {
        var batchInputShape = [1, 2];
        var inputLayer = tfl.layers.inputLayer({ batchInputShape: batchInputShape });
        expect(inputLayer.batchInputShape).toEqual(batchInputShape);
    });
    it('initializes with batchInputShape if null specified for the batch size.', function () {
        var batchInputShape = [1, 2];
        var inputLayer = tfl.layers.inputLayer({ batchInputShape: batchInputShape });
        expect(inputLayer.batchInputShape).toEqual(batchInputShape);
    });
    it('throws exception if batchSize and batchInputShape are specified.', function () {
        expect(function () { return tfl.layers.inputLayer({ batchInputShape: [1], batchSize: 5 }); })
            .toThrowError(/Cannot specify batchSize if batchInputShape/);
    });
    var _loop_2 = function (sparse) {
        it('uses config.sparse during initialization.', function () {
            var inputLayer = tfl.layers.inputLayer({ inputShape: [1], sparse: sparse });
            expect(inputLayer.sparse).toEqual(sparse);
        });
    };
    for (var _b = 0, _c = [true, false]; _b < _c.length; _b++) {
        var sparse = _c[_b];
        _loop_2(sparse);
    }
    it('use config.dtype during initialization.', function () {
        var dtype = 'float32';
        var inputLayer = tfl.layers.inputLayer({ inputShape: [1], dtype: dtype });
        expect(inputLayer.dtype).toEqual(dtype);
    });
    it('use config.name during initialization.', function () {
        var name = 'abc';
        var inputLayer = tfl.layers.inputLayer({ inputShape: [1], name: name });
        expect(inputLayer.name).toEqual(name);
    });
    it('throws an exception if apply() is called with any input.', function () {
        var inputLayer = tfl.layers.inputLayer({ inputShape: [1] });
        var symbolicTensor = new tfl.SymbolicTensor('float32', [2], null, [], {});
        expect(function () { return inputLayer.apply(symbolicTensor); })
            .toThrowError(/Cannot pass any input to an InputLayer's apply/);
    });
    it('throws an exception if its inputs differ in shape to what it ' +
        'was initialized to.', function () {
        var inputLayer = tfl.layers.inputLayer({ inputShape: [1] });
        var inputs = tfjs_core_1.ones([2, 2]);
        expect(function () { return inputLayer.apply(inputs); }).toThrowError();
    });
    it('returns a serializable config.', function () {
        var batchInputShape = [1];
        var dtype = 'float32';
        var sparse = true;
        var name = 'my_name';
        var inputLayer = tfl.layers.inputLayer({ batchInputShape: batchInputShape, dtype: dtype, sparse: sparse, name: name });
        expect(inputLayer.getConfig())
            .toEqual({ batchInputShape: batchInputShape, dtype: dtype, sparse: sparse, name: name });
    });
});
var LayerForTest = (function (_super) {
    __extends(LayerForTest, _super);
    function LayerForTest(config) {
        return _super.call(this, config) || this;
    }
    LayerForTest.className = 'LayerForTest';
    return LayerForTest;
}(tfl.layers.Layer));
describe('Input()', function () {
    it('throws an exception if neither shape nor batchShape are specified', function () {
        expect(function () { return tfl.layers.input({}); })
            .toThrowError(/Please provide to Input either/);
    });
    var shape = [1];
    var batchShape = [2, 2];
    var name = 'abc';
    var dtype = 'float32';
    it('returns an initialized symbolicTensor given a shape.', function () {
        var symbolicTensor = tfl.layers.input({ shape: shape, name: name, dtype: dtype });
        expect(symbolicTensor instanceof tfl.SymbolicTensor).toBe(true);
        expect(symbolicTensor.shape).toEqual([null].concat(shape));
        expect(symbolicTensor.name).toMatch(/abc/);
        expect(symbolicTensor.dtype).toEqual(dtype);
    });
    it('returns a SymbolicTensor given a batchShape', function () {
        var symbolicTensor = tfl.layers.input({ batchShape: batchShape });
        expect(symbolicTensor.shape).toEqual(batchShape);
    });
    it('throws exception if both shape and batchShape are specified.', function () {
        expect(function () { return tfl.layers.input({ shape: shape, batchShape: batchShape }); })
            .toThrowError(/Please provide either a `shape`/);
    });
    it('produces output that can feed into a Layer.', function () {
        var inputTensor = input_layer_1.Input({ shape: shape, name: name });
        var otherLayer = new LayerForTest({ name: 'firstLayer' });
        var output = otherLayer.apply(inputTensor);
        expect(output instanceof tfl.SymbolicTensor).toBe(true);
        expect(output.name).toEqual('firstLayer/firstLayer');
    });
});
//# sourceMappingURL=input_layer_test.js.map