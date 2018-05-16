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
var test_utils_1 = require("../utils/test_utils");
var container_1 = require("./container");
var executor_1 = require("./executor");
var topology_1 = require("./topology");
var LayerForTest = (function (_super) {
    __extends(LayerForTest, _super);
    function LayerForTest(config) {
        return _super.call(this, config) || this;
    }
    LayerForTest.className = 'LayerForTest';
    return LayerForTest;
}(tfl.layers.Layer));
var ContainerForTest = (function (_super) {
    __extends(ContainerForTest, _super);
    function ContainerForTest(config) {
        return _super.call(this, config) || this;
    }
    ContainerForTest.className = 'ContainerForTest';
    return ContainerForTest;
}(container_1.Container));
test_utils_1.describeMathCPUAndGPU('Container.fromConfig', function () {
    it('creates a minimal Container from simplest config', function () {
        var config = {
            name: 'test',
            layers: [],
            inputLayers: [],
            outputLayers: []
        };
        var container = container_1.Container.fromConfig(ContainerForTest, config);
        expect(container.name).toEqual('test');
    });
    it('creates a simple network', function () {
        var config = {
            inputLayers: [['input_2', 0, 0]],
            layers: [
                {
                    className: 'InputLayer',
                    config: {
                        batchInputShape: [null, 32],
                        dtype: 'float32',
                        name: 'input_2',
                        sparse: false
                    },
                    inboundNodes: [],
                    name: 'input_2'
                },
                {
                    className: 'Dense',
                    config: {
                        activation: 'linear',
                        activityRegularizer: null,
                        biasConstraint: null,
                        biasInitializer: { className: 'Zeros', config: {} },
                        biasRegularizer: null,
                        kernelConstraint: null,
                        kernelInitializer: {
                            className: 'VarianceScaling',
                            config: {
                                distribution: 'uniform',
                                mode: 'fanAvg',
                                scale: 1.0,
                                seed: null
                            }
                        },
                        kernelRegularizer: null,
                        name: 'dense_2',
                        trainable: null,
                        units: 32,
                        use_bias: true
                    },
                    inboundNodes: [[['input_2', 0, 0, {}]]],
                    name: 'dense_2'
                }
            ],
            name: 'test',
            outputLayers: [['dense_2', 0, 0]]
        };
        var container = container_1.Container.fromConfig(ContainerForTest, config);
        expect(container.name).toEqual('test');
        var allZeros = tfjs_core_1.zeros([1, 32]);
        test_utils_1.expectTensorsClose(container.apply(allZeros), allZeros);
    });
});
test_utils_1.describeMathCPUAndGPU('Container', function () {
    var inputLayerName = 'inputLayerName';
    var layerName = 'layerName';
    var containerName = 'simpleContainer';
    var inputTensor;
    var layer;
    var output;
    var simpleContainer;
    beforeEach(function () {
        inputTensor =
            tfl.input({ shape: [1], name: inputLayerName, dtype: 'float32' });
        layer = new LayerForTest({ name: layerName });
        output = layer.apply(inputTensor);
        simpleContainer = new ContainerForTest({ inputs: [inputTensor], outputs: [output], name: containerName });
    });
    it('initializes with no inputs or outputs and a default name', function () {
        var container = new ContainerForTest({ inputs: [], outputs: [] });
        expect(container.name).toMatch(/^container.+$/);
    });
    it('initializes with no inputs or outputs and a given name', function () {
        var name = 'xyz';
        var container = new ContainerForTest({ inputs: [], outputs: [], name: name });
        expect(container.name).toMatch(name);
    });
    it('throws an exception if same input provided twice', function () {
        var makeContainer = function () {
            new ContainerForTest({ inputs: [inputTensor, inputTensor], outputs: [] });
        };
        expect(makeContainer).toThrowError(/inputs.*redundant/);
    });
    it('throws an exception if graph is disconnected', function () {
        var makeContainer = function () {
            new ContainerForTest({ inputs: [], outputs: [output] });
        };
        expect(makeContainer).toThrowError(/disconnected/);
    });
    it('creates inputLayers', function () {
        expect(simpleContainer.inputLayers).toEqual([inputTensor.sourceLayer]);
    });
    it('creates outputLayers', function () {
        expect(simpleContainer.outputLayers).toEqual([layer]);
    });
    it('creates inputNames', function () {
        expect(simpleContainer.inputNames).toEqual([inputLayerName]);
    });
    it('creates outputNames', function () {
        expect(simpleContainer.outputNames).toEqual([layerName]);
    });
    it('throws exception if given a non-input layer as input', function () {
        var makeContainer = function () {
            new ContainerForTest({ inputs: [output], outputs: [] });
        };
        expect(makeContainer).toThrowError(/must be InputLayer objects/);
    });
    it('creates layers for simplest case', function () {
        expect(simpleContainer.layers).toEqual([inputTensor.sourceLayer, layer]);
    });
    it('creates layers when multiple layers specified', function () {
        var layer1 = new LayerForTest({ name: 'layer1' });
        var layer2 = new LayerForTest({ name: 'layer2' });
        var output = layer2.apply(layer1.apply(inputTensor));
        var container = new ContainerForTest({ inputs: [inputTensor], outputs: [output] });
        expect(container.layers).toEqual([inputTensor.sourceLayer, layer1, layer2]);
    });
    it('correctly creates model with shared subgraphs.', function () {
        var layerA = new LayerForTest({ name: 'A' });
        var layerB = new LayerForTest({ name: 'B' });
        var layerC = new LayerForTest({ name: 'C' });
        var layerX = new LayerForTest({ name: 'X' });
        var aOutput = layerA.apply(inputTensor);
        var output1 = layerC.apply(layerB.apply(aOutput));
        var output2 = layerC.apply(layerB.apply(layerX.apply(aOutput)));
        var container = new ContainerForTest({ inputs: [inputTensor], outputs: [output1, output2] });
        var compareFunction = function (a, b) {
            if (a.name < b.name) {
                return -1;
            }
            else if (a.name > b.name) {
                return 1;
            }
            else {
                return 0;
            }
        };
        var sortedLayers = container.layers.slice().sort(compareFunction);
        var expectedSortedLayers = [
            inputTensor.sourceLayer, layerA, layerB, layerC, layerX
        ].sort(compareFunction);
        expect(sortedLayers).toEqual(expectedSortedLayers);
    });
    it('throws exception if multiple layers have the same name', function () {
        var name = 'abc';
        var layer1 = new LayerForTest({ name: name });
        var layer2 = new LayerForTest({ name: name });
        var output = layer2.apply(layer1.apply(inputTensor));
        var makeContainer = function () {
            new ContainerForTest({ inputs: [inputTensor], outputs: [output] });
        };
        expect(makeContainer).toThrowError(/layer names should be unique/);
    });
    it('weights gets all weights.', function () {
        var inputShape = [1, 6];
        var inputLayer = tfl.layers.input({ shape: inputShape });
        var layer1 = tfl.layers.dense({ units: 2, useBias: false });
        var layer2 = tfl.layers.dense({ units: 1, useBias: true });
        var output = layer2.apply(layer1.apply(inputLayer));
        var container = new ContainerForTest({ inputs: [inputLayer], outputs: [output] });
        expect(container.weights.length).toEqual(3);
        expect(container.weights[0].name).toEqual(layer1.weights[0].name);
        expect(container.weights[1].name).toEqual(layer2.weights[0].name);
        expect(container.weights[2].name).toEqual(layer2.weights[1].name);
    });
    it('trainableWeights and nonTrainableWeights.', function () {
        var inputShape = [1, 6];
        var inputLayer = tfl.layers.input({ shape: inputShape });
        var layer1 = tfl.layers.dense({ units: 2, useBias: false });
        var layer2 = tfl.layers.dense({ units: 1, useBias: true });
        var output = layer2.apply(layer1.apply(inputLayer));
        var container = new ContainerForTest({ inputs: [inputLayer], outputs: [output] });
        expect(container.trainableWeights.length).toEqual(3);
        expect(container.trainableWeights[0].name).toEqual(layer1.weights[0].name);
        expect(container.trainableWeights[1].name).toEqual(layer2.weights[0].name);
        expect(container.trainableWeights[2].name).toEqual(layer2.weights[1].name);
        expect(container.nonTrainableWeights.length).toEqual(0);
    });
    it('call() executes all layers.', function () {
        var inputShape = [1, 6];
        var finalShape = [3, 2];
        var inputLayer = tfl.layers.input({ shape: inputShape });
        var layer1 = tfl.layers.reshape({ name: 'layer1', targetShape: [2, 3] });
        var layer2 = tfl.layers.reshape({ name: 'layer2', targetShape: finalShape });
        var output = layer2.apply(layer1.apply(inputLayer));
        var container = new ContainerForTest({ inputs: [inputLayer], outputs: [output] });
        var result = container.call(tfjs_core_1.ones([1, 1, 6]), {});
        var resultShape = [1].concat(finalShape);
        test_utils_1.expectTensorsClose(result[0], tfjs_core_1.ones(resultShape));
    });
    it('apply() executes all layers with concrete tensors.', function () {
        var inputShape = [1, 6];
        var finalShape = [3, 2];
        var inputLayer = tfl.layers.input({ shape: inputShape });
        var layer1 = tfl.layers.reshape({ name: 'layer1', targetShape: [2, 3] });
        var layer2 = tfl.layers.reshape({ name: 'layer2', targetShape: finalShape });
        var output = layer2.apply(layer1.apply(inputLayer));
        var container = new ContainerForTest({ inputs: [inputLayer], outputs: [output] });
        var result = container.apply(tfjs_core_1.ones([1, 1, 6]));
        var resultShape = [1].concat(finalShape);
        test_utils_1.expectTensorsClose(result, tfjs_core_1.ones(resultShape));
    });
    it('apply() executes all layers with symbolic tensors.', function () {
        var inputShape = [1, 6];
        var finalShape = [3, 2];
        var inputLayer = tfl.layers.input({ shape: inputShape });
        var layer1 = tfl.layers.reshape({ name: 'layer1', targetShape: [2, 3] });
        var layer2 = tfl.layers.reshape({ name: 'layer2', targetShape: finalShape });
        var output = layer2.apply(layer1.apply(inputLayer));
        var container = new ContainerForTest({ inputs: [inputLayer], outputs: [output] });
        var newInput = tfl.layers.input({ shape: [1, 6] });
        var symbolicResult = container.apply(newInput);
        expect(symbolicResult instanceof tfl.SymbolicTensor).toEqual(true);
        var concreteResult = executor_1.execute(symbolicResult, new executor_1.FeedDict([{ key: newInput, value: tfjs_core_1.ones([1, 1, 6]) }]));
        var resultShape = [1].concat(finalShape);
        test_utils_1.expectTensorsClose(concreteResult, tfjs_core_1.ones(resultShape));
    });
    it('computeOutputShape() computes the correct outputShape', function () {
        var inputShape = [2, 3];
        var finalShape = [3, 2];
        var inputLayer = tfl.layers.input({ shape: inputShape });
        var layer = tfl.layers.reshape({ targetShape: finalShape });
        var output = layer.apply(inputLayer);
        var container = new ContainerForTest({ inputs: [inputLayer], outputs: [output] });
        expect(container.computeOutputShape([1].concat(inputShape))).toEqual([
            1
        ].concat(finalShape));
    });
    it('trainableWeights is initially an empty Array', function () {
        expect(simpleContainer.trainableWeights).toEqual([]);
    });
    it('trainableWeights tracks only trainable weights', function () {
        var inputShape = [2, 2];
        var inputLayer = tfl.layers.input({ shape: inputShape });
        var layer1 = tfl.layers.reshape({ targetShape: [4], name: 'reshapeLayer' });
        var layer1Output = layer1.apply(inputLayer);
        var layer2 = tfl.layers.dense({ units: 2, useBias: false, name: 'denseLayer' });
        var layer2Output = layer2.apply(layer1Output);
        var container = new ContainerForTest({ inputs: [inputLayer], outputs: [layer2Output] });
        expect(container.trainableWeights.length).toEqual(1);
    });
    it('stateful is initially false', function () {
        expect(simpleContainer.stateful).toEqual(false);
    });
    function createSimpleTwoLayerContainer() {
        var inputShape = [2, 2];
        var inputLayer = tfl.layers.input({ shape: inputShape });
        var layer1 = tfl.layers.reshape({ targetShape: [4], name: 'reshapeLayer' });
        var layer1Output = layer1.apply(inputLayer);
        var layer2 = tfl.layers.dense({ units: 2, useBias: false, name: 'denseLayer' });
        var layer2Output = layer2.apply(layer1Output);
        var container = new ContainerForTest({ inputs: [inputLayer], outputs: [layer2Output] });
        return [container, [container.inputLayers[0], layer1, layer2]];
    }
    it('getLayer works by name', function () {
        var _a = createSimpleTwoLayerContainer(), container = _a[0], layers = _a[1];
        expect(container.getLayer(layers[0].name)).toEqual(layers[0]);
        expect(container.getLayer(layers[1].name)).toEqual(layers[1]);
        expect(container.getLayer(layers[2].name)).toEqual(layers[2]);
    });
    it('getLayer works by index', function () {
        var _a = createSimpleTwoLayerContainer(), container = _a[0], layers = _a[1];
        expect(container.getLayer(null, 0)).toEqual(layers[0]);
        expect(container.getLayer(null, 1)).toEqual(layers[1]);
        expect(container.getLayer(null, 2)).toEqual(layers[2]);
    });
    it('getLayer throws error for nonexistent layer name', function () {
        var _a = createSimpleTwoLayerContainer(), container = _a[0], layers = _a[1];
        expect(function () { return container.getLayer(layers[0].name + '_suffixToMakeLayerNameNonexistent'); })
            .toThrowError(/No such layer/);
    });
    it('getLayer throws error for index out of bound', function () {
        var container = createSimpleTwoLayerContainer()[0];
        expect(function () { return container.getLayer(null, 3); }).toThrowError(/only has 3 layer/);
    });
    it('getLayer throws error when neither name or index is specified', function () {
        var container = createSimpleTwoLayerContainer()[0];
        expect(function () { return container.getLayer(); })
            .toThrowError(/Provide either a layer name or layer index/);
    });
});
test_utils_1.describeMathCPUAndGPU('Container.calculateLosses', function () {
    function createSimpleOneLayerContainer(useRegularizers) {
        var inputShape = [2];
        var inputLayer = tfl.layers.input({ shape: inputShape });
        var kernelRegularizer = useRegularizers ? tfl.regularizers.l1({ l1: 2 }) : null;
        var biasRegularizer = useRegularizers ? tfl.regularizers.l2({ l2: 3 }) : null;
        var denseLayer = tfl.layers.dense({
            units: 2,
            kernelInitializer: 'ones',
            biasInitializer: 'ones',
            kernelRegularizer: kernelRegularizer,
            biasRegularizer: biasRegularizer,
            name: 'denseLayer'
        });
        var layer2Output = denseLayer.apply(inputLayer);
        var container = new ContainerForTest({ inputs: [inputLayer], outputs: [layer2Output] });
        return [container, [denseLayer]];
    }
    it('L1 and L2', function () {
        var container = createSimpleOneLayerContainer(true)[0];
        var losses = container.calculateLosses();
        expect(losses.length).toEqual(2);
        test_utils_1.expectTensorsClose(losses[0], tfjs_core_1.scalar(2 * (1 + 1 + 1 + 1)));
        test_utils_1.expectTensorsClose(losses[1], tfjs_core_1.scalar(3 * (1 + 1)));
    });
    it('No regularizers', function () {
        var container = createSimpleOneLayerContainer(false)[0];
        var losses = container.calculateLosses();
        expect(losses.length).toEqual(0);
    });
});
describe('getSourceInputs()', function () {
    it('returns the single source input', function () {
        var inputTensor = tfl.layers.input({ shape: [1] });
        var layer1 = new LayerForTest({ name: 'layer1' });
        var layer2 = new LayerForTest({ name: 'layer2' });
        var output = layer2.apply(layer1.apply(inputTensor));
        expect(topology_1.getSourceInputs(output)).toEqual([inputTensor]);
    });
    it('returns all inputs', function () {
        var input1 = tfl.layers.input({ shape: [1], name: 'input1' });
        var input2 = tfl.layers.input({ shape: [1], name: 'input2' });
        var layer = new LayerForTest({});
        var output1 = layer.apply(input1);
        var output2 = layer.apply(input2);
        expect(topology_1.getSourceInputs(output1)).toEqual([input1]);
        expect(topology_1.getSourceInputs(output2)).toEqual([input2]);
    });
});
test_utils_1.describeMathCPUAndGPU('Model-dispose', function () {
    it('Dispose Sequential model frees memory', function () {
        var model = tfl.sequential();
        model.add(tfl.layers.dense({ units: 2, inputShape: [3], activation: 'relu' }));
        model.add(tfl.layers.dense({ units: 1 }));
        model.build([3, 3]);
        var numTensors0 = tfjs_core_1.memory().numTensors;
        var result = model.dispose();
        expect(result.refCountAfterDispose).toEqual(0);
        expect(result.numDisposedVariables).toEqual(4);
        expect(tfjs_core_1.memory().numTensors).toEqual(numTensors0 - 4);
    });
    it('Dispose Sequential model twice leads to Error', function () {
        var model = tfl.sequential();
        model.add(tfl.layers.dense({ units: 2, inputShape: [3], activation: 'relu' }));
        model.add(tfl.layers.dense({ units: 1 }));
        model.build([3, 3]);
        model.dispose();
        expect(function () { return model.dispose(); }).toThrowError(/Container .* already disposed/);
    });
    it('Using disposed Sequential model leads to Error', function () { return __awaiter(_this, void 0, void 0, function () {
        var model, xs, ys, errorCaughtDuringFit, err_1;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    model = tfl.sequential();
                    model.add(tfl.layers.dense({ units: 2, inputShape: [3], activation: 'relu' }));
                    model.add(tfl.layers.dense({ units: 1, activation: 'sigmoid' }));
                    model.build([3, 3]);
                    model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });
                    model.dispose();
                    xs = tfjs_core_1.zeros([3, 3]);
                    ys = tfjs_core_1.zeros([3, 1]);
                    expect(function () { return model.predict(xs); }).toThrowError(/already disposed/);
                    expect(function () { return model.evaluate(xs, ys); }).toThrowError(/already disposed/);
                    errorCaughtDuringFit = false;
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 3, , 4]);
                    return [4, model.fit(xs, ys)];
                case 2:
                    _a.sent();
                    return [3, 4];
                case 3:
                    err_1 = _a.sent();
                    errorCaughtDuringFit = true;
                    return [3, 4];
                case 4:
                    expect(errorCaughtDuringFit).toEqual(true);
                    return [2];
            }
        });
    }); });
    it('Dispose functional model frees memory', function () {
        var input = tfl.input({ shape: [4] });
        var dense1 = tfl.layers.dense({ units: 3 }).apply(input);
        var dense2 = tfl.layers.dense({ units: 2, useBias: false }).apply(input);
        var model = tfl.model({ inputs: [input], outputs: [dense1, dense2] });
        model.predict(tfjs_core_1.zeros([2, 4]));
        var numTensors0 = tfjs_core_1.memory().numTensors;
        var result = model.dispose();
        expect(result.refCountAfterDispose).toEqual(0);
        expect(result.numDisposedVariables).toEqual(3);
        expect(tfjs_core_1.memory().numTensors).toEqual(numTensors0 - 3);
    });
    it('Dispose functional model twice leads to Error', function () {
        var input = tfl.input({ shape: [4] });
        var dense1 = tfl.layers.dense({ units: 3 }).apply(input);
        var dense2 = tfl.layers.dense({ units: 2, useBias: false }).apply(input);
        var model = tfl.model({ inputs: [input], outputs: [dense1, dense2] });
        model.predict(tfjs_core_1.zeros([2, 4]));
        model.dispose();
        expect(function () { return model.dispose(); }).toThrowError(/Container .* already disposed/);
    });
    it('Layer shared between two functional models is not disposed', function () {
        var input1 = tfl.input({ shape: [4] });
        var input2 = tfl.input({ shape: [4] });
        var sharedDenseLayer = tfl.layers.dense({ units: 3, activation: 'relu' });
        var nonSharedDenseLayer1 = tfl.layers.dense({ units: 1, useBias: false });
        var nonSharedDenseLayer2 = tfl.layers.dense({ units: 1, useBias: false });
        var output1 = nonSharedDenseLayer1.apply(sharedDenseLayer.apply(input1));
        var output2 = nonSharedDenseLayer2.apply(sharedDenseLayer.apply(input2));
        var model1 = tfl.model({ inputs: [input1], outputs: [output1] });
        var model2 = tfl.model({ inputs: [input2], outputs: [output2] });
        model1.predict(tfjs_core_1.zeros([2, 4]));
        model2.predict(tfjs_core_1.zeros([2, 4]));
        var xs = tfjs_core_1.zeros([2, 4]);
        var numTensors0 = tfjs_core_1.memory().numTensors;
        var result1 = model1.dispose();
        expect(result1.refCountAfterDispose).toEqual(0);
        expect(result1.numDisposedVariables).toEqual(1);
        expect(tfjs_core_1.memory().numTensors).toEqual(numTensors0 - 1);
        expect(function () { return model1.predict(xs); }).toThrowError(/already disposed/);
        var ys = model2.predict(xs);
        expect(ys.shape).toEqual([2, 1]);
        ys.dispose();
        var result2 = model2.dispose();
        expect(result2.refCountAfterDispose).toEqual(0);
        expect(result2.numDisposedVariables).toEqual(3);
        expect(tfjs_core_1.memory().numTensors).toEqual(numTensors0 - 4);
        expect(function () { return model1.predict(xs); }).toThrowError(/already disposed/);
        expect(function () { return model2.predict(xs); }).toThrowError(/already disposed/);
    });
    it('Disposing nested sequential model preserves the inner model', function () {
        var innerModel = tfl.sequential();
        innerModel.add(tfl.layers.reshape({ targetShape: [10], inputShape: [2, 5] }));
        innerModel.add(tfl.layers.dense({ units: 6, activation: 'relu' }));
        innerModel.add(tfl.layers.dense({ units: 4, activation: 'relu' }));
        var outerModel = tfl.sequential();
        outerModel.add(tfl.layers.reshape({ targetShape: [2, 5], inputShape: [5, 2] }));
        outerModel.add(innerModel);
        outerModel.add(tfl.layers.dense({ units: 3, activation: 'softmax' }));
        var xsOuter = tfjs_core_1.zeros([1, 5, 2]);
        var xsInner = tfjs_core_1.zeros([1, 2, 5]);
        outerModel.predict(xsOuter);
        var numTensors0 = tfjs_core_1.memory().numTensors;
        var result1 = outerModel.dispose();
        expect(result1.refCountAfterDispose).toEqual(0);
        expect(result1.numDisposedVariables).toEqual(2);
        expect(tfjs_core_1.memory().numTensors).toEqual(numTensors0 - 2);
        expect(function () { return outerModel.dispose(); })
            .toThrowError(/Container .* already disposed/);
        expect(function () { return outerModel.predict(xsOuter); }).toThrowError(/already disposed/);
        var ysInner = innerModel.predict(xsInner);
        expect(ysInner.shape).toEqual([1, 4]);
        ysInner.dispose();
        var result2 = innerModel.dispose();
        expect(result2.refCountAfterDispose).toEqual(0);
        expect(result2.numDisposedVariables).toEqual(4);
        expect(tfjs_core_1.memory().numTensors).toEqual(numTensors0 - 6);
        expect(function () { return innerModel.predict(xsInner); }).toThrowError(/already disposed/);
    });
});
//# sourceMappingURL=container_test.js.map