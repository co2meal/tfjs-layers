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
var tfl = require("./index");
var core_1 = require("./layers/core");
var serialization_1 = require("./layers/serialization");
var models_1 = require("./models");
var serialization_utils_1 = require("./utils/serialization_utils");
var test_utils_1 = require("./utils/test_utils");
var version_1 = require("./version");
test_utils_1.describeMathCPU('Nested model topology', function () {
    it('Nested Sequential model: Sequential as first layer', function (done) {
        var modelLevel1 = tfl.sequential({ layers: [tfl.layers.dense({ units: 2, inputShape: [3] })] });
        var x = tfjs_core_1.ones([1, 3]);
        var y = modelLevel1.predict(x);
        var modelLevel2 = tfl.sequential();
        modelLevel2.add(modelLevel1);
        test_utils_1.expectTensorsClose(modelLevel2.predict(x), y);
        var modelLevel3 = tfl.sequential();
        modelLevel3.add(modelLevel2);
        test_utils_1.expectTensorsClose(modelLevel3.predict(x), y);
        var xs = tfjs_core_1.ones([8, 3]);
        var ys = tfjs_core_1.zeros([8, 2]);
        modelLevel3.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });
        modelLevel3.fit(xs, ys)
            .then(function (history) {
            var newY = modelLevel1.predict(x);
            test_utils_1.expectTensorsClose(modelLevel2.predict(x), newY);
            test_utils_1.expectTensorsClose(modelLevel3.predict(x), newY);
            done();
        })
            .catch(function (err) { return done.fail(err.stack); });
    });
    it('Nested Sequential model: Functional model as first layer', function (done) {
        var input = tfl.input({ shape: [3] });
        var output = tfl.layers.dense({ units: 2 }).apply(input);
        var modelLevel1 = tfl.model({ inputs: input, outputs: output });
        var x = tfjs_core_1.ones([1, 3]);
        var y = modelLevel1.predict(x);
        var modelLevel2 = tfl.sequential();
        modelLevel2.add(modelLevel1);
        test_utils_1.expectTensorsClose(modelLevel2.predict(x), y);
        var modelLevel3 = tfl.sequential();
        modelLevel3.add(modelLevel2);
        test_utils_1.expectTensorsClose(modelLevel3.predict(x), y);
        var xs = tfjs_core_1.ones([8, 3]);
        var ys = tfjs_core_1.zeros([8, 2]);
        modelLevel3.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });
        modelLevel3.fit(xs, ys)
            .then(function (history) {
            var newY = modelLevel1.predict(x);
            test_utils_1.expectTensorsClose(modelLevel2.predict(x), newY);
            test_utils_1.expectTensorsClose(modelLevel3.predict(x), newY);
            done();
        })
            .catch(function (err) { return done.fail(err.stack); });
    });
    it('Nested Sequential model: Sequential as second layer', function (done) {
        var innerModel = tfl.sequential({ layers: [tfl.layers.dense({ units: 2, inputShape: [4] })] });
        var x = tfjs_core_1.ones([1, 4]);
        var y = innerModel.predict(x);
        var x2By2 = tfjs_core_1.ones([1, 2, 2]);
        var outerModel = tfl.sequential({ layers: [tfl.layers.reshape({ targetShape: [4], inputShape: [2, 2] })] });
        outerModel.add(innerModel);
        test_utils_1.expectTensorsClose(outerModel.predict(x2By2), y);
        var xs = tfjs_core_1.ones([8, 2, 2]);
        var ys = tfjs_core_1.zeros([8, 2]);
        outerModel.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });
        outerModel.fit(xs, ys)
            .then(function (history) {
            var newY = innerModel.predict(x);
            test_utils_1.expectTensorsClose(outerModel.predict(x2By2), newY);
            done();
        })
            .catch(function (err) { return done.fail(err.stack); });
    });
    it('Nested Sequential model: Sequential as middle layer', function (done) {
        var innerModel = tfl.sequential({ layers: [tfl.layers.dense({ units: 4, inputShape: [4] })] });
        var x = tfjs_core_1.ones([1, 4]);
        var y = innerModel.predict(x);
        var x2By2 = tfjs_core_1.ones([1, 2, 2]);
        var outerModel = tfl.sequential({
            layers: [
                tfl.layers.reshape({ targetShape: [4], inputShape: [2, 2] }), innerModel,
                tfl.layers.reshape({ targetShape: [2, 2] })
            ]
        });
        test_utils_1.expectTensorsClose(outerModel.predict(x2By2), y.reshape([1, 2, 2]));
        var xs = tfjs_core_1.ones([8, 2, 2]);
        var ys = tfjs_core_1.zeros([8, 2, 2]);
        outerModel.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });
        outerModel.fit(xs, ys)
            .then(function (history) {
            var newY = innerModel.predict(x);
            test_utils_1.expectTensorsClose(outerModel.predict(x2By2), newY.reshape([1, 2, 2]));
            done();
        })
            .catch(function (err) { return done.fail(err.stack); });
    });
    it('getLayer() works for nested sequential model', function () {
        var innerModel = tfl.sequential({
            layers: [
                tfl.layers.dense({
                    units: 4,
                    inputShape: [4],
                    activation: 'relu',
                    kernelInitializer: 'ones',
                    biasInitializer: 'ones'
                }),
                tfl.layers.dense({
                    units: 2,
                    activation: 'tanh',
                    kernelInitializer: 'ones',
                    biasInitializer: 'ones'
                })
            ]
        });
        var outerModel = tfl.sequential({
            layers: [
                innerModel,
                tfl.layers.dense({ units: 1, kernelInitializer: 'ones', biasInitializer: 'ones' })
            ]
        });
        expect(outerModel.getLayer(null, 0) instanceof tfl.Sequential)
            .toEqual(true);
        test_utils_1.expectTensorsClose(outerModel.getLayer(null, 1).getWeights()[0], tfjs_core_1.ones([2, 1]));
        test_utils_1.expectTensorsClose(outerModel.getLayer(null, 1).getWeights()[1], tfjs_core_1.ones([1]));
        expect(function () { return outerModel.getLayer(null, 2); }).toThrow();
    });
    it('getWeights() works for nested sequential model', function () {
        var innerModel = tfl.sequential({
            layers: [
                tfl.layers.dense({
                    units: 4,
                    inputShape: [4],
                    activation: 'relu',
                    kernelInitializer: 'ones',
                    biasInitializer: 'ones'
                }),
                tfl.layers.dense({
                    units: 2,
                    activation: 'tanh',
                    kernelInitializer: 'ones',
                    biasInitializer: 'ones'
                })
            ]
        });
        var outerModel = tfl.sequential({ layers: [innerModel, tfl.layers.dense({ units: 1 })] });
        var weights = outerModel.getWeights();
        expect(weights.length).toEqual(6);
        expect(weights[0].shape).toEqual([4, 4]);
        expect(weights[1].shape).toEqual([4]);
        expect(weights[2].shape).toEqual([4, 2]);
        expect(weights[3].shape).toEqual([2]);
        expect(weights[4].shape).toEqual([2, 1]);
        expect(weights[5].shape).toEqual([1]);
    });
    it('setWeights() works for nested sequential model', function () {
        var innerModel = tfl.sequential({
            layers: [
                tfl.layers.dense({
                    units: 2,
                    inputShape: [3],
                    activation: 'relu',
                    kernelInitializer: 'zeros',
                    useBias: false
                }),
            ]
        });
        var outerModel = tfl.sequential({
            layers: [
                innerModel,
                tfl.layers.dense({ units: 1, kernelInitializer: 'zeros', useBias: false })
            ]
        });
        outerModel.setWeights([tfjs_core_1.ones([3, 2]), tfjs_core_1.ones([2, 1])]);
        test_utils_1.expectTensorsClose(outerModel.getWeights()[0], tfjs_core_1.ones([3, 2]));
        test_utils_1.expectTensorsClose(outerModel.getWeights()[1], tfjs_core_1.ones([2, 1]));
        test_utils_1.expectTensorsClose(outerModel.predict(tfjs_core_1.ones([1, 3])), tfjs_core_1.tensor2d([[6]]));
    });
    it('Sequential as layer: save-load round trip', function () {
        var innerModel = tfl.sequential({
            layers: [
                tfl.layers.dense({
                    units: 4,
                    inputShape: [4],
                    activation: 'relu',
                    kernelInitializer: 'ones',
                    biasInitializer: 'ones'
                }),
                tfl.layers.dense({
                    units: 4,
                    activation: 'tanh',
                    kernelInitializer: 'ones',
                    biasInitializer: 'ones'
                })
            ]
        });
        var outerModel = tfl.sequential({
            layers: [
                tfl.layers.reshape({ targetShape: [4], inputShape: [2, 2] }), innerModel,
                tfl.layers.dense({ units: 1, kernelInitializer: 'ones', biasInitializer: 'ones' })
            ]
        });
        var x = tfjs_core_1.randomNormal([1, 2, 2]);
        var y = outerModel.predict(x);
        var unusedArg = null;
        var returnString = false;
        var outerModelJSON = outerModel.toJSON(unusedArg, returnString);
        var reconstructedModel = serialization_1.deserialize(serialization_utils_1.convertPythonicToTs(outerModelJSON));
        expect(reconstructedModel.toJSON(unusedArg, returnString))
            .toEqual(outerModelJSON);
        test_utils_1.expectTensorsClose(reconstructedModel.predict(x), y);
    });
    it('Functional model as layer: save-load round trip', function () {
        var input = tfl.input({ shape: [4] });
        var layer1 = tfl.layers.dense({
            units: 4,
            activation: 'relu',
            kernelInitializer: 'ones',
            biasInitializer: 'ones'
        });
        var layer2 = tfl.layers.dense({
            units: 4,
            activation: 'tanh',
            kernelInitializer: 'ones',
            biasInitializer: 'ones'
        });
        var output = layer2.apply(layer1.apply(input));
        var innerModel = tfl.model({ inputs: input, outputs: output });
        var outerModel = tfl.sequential({
            layers: [
                tfl.layers.reshape({ targetShape: [4], inputShape: [2, 2] }), innerModel,
                tfl.layers.reshape({ targetShape: [2, 2] })
            ]
        });
        var x = tfjs_core_1.randomNormal([1, 2, 2]);
        var y = outerModel.predict(x);
        var unusedArg = null;
        var returnString = false;
        var outerModelJSON = outerModel.toJSON(unusedArg, returnString);
        var reconstructedModel = serialization_1.deserialize(serialization_utils_1.convertPythonicToTs(outerModelJSON));
        expect(reconstructedModel.toJSON(unusedArg, returnString))
            .toEqual(outerModelJSON);
        test_utils_1.expectTensorsClose(reconstructedModel.predict(x), y);
    });
    it('Attempt to nest two-input functional model fails', function () {
        var input1 = tfl.input({ shape: [4] });
        var input2 = tfl.input({ shape: [5] });
        var output = tfl.layers.concatenate().apply([input1, input2]);
        var innerModel = tfl.model({ inputs: [input1, input2], outputs: output });
        var outerModel = tfl.sequential();
        expect(function () { return outerModel.add(innerModel); })
            .toThrowError(/should have a single input tensor/);
        var outerModel2 = tfl.sequential({ layers: [tfl.layers.dense({ units: 4, inputShape: [8] })] });
        expect(function () { return outerModel2.add(innerModel); })
            .toThrowError(/should have a single input tensor/);
        expect(function () { return tfl.sequential({
            layers: [innerModel]
        }); }).toThrowError(/should have a single input tensor/);
    });
    it('Attempt to nest two-output functional model fails', function () {
        var input = tfl.input({ shape: [12] });
        var output1 = tfl.layers.reshape({ targetShape: [2, 6] }).apply(input);
        var output2 = tfl.layers.reshape({ targetShape: [3, 4] }).apply(input);
        var innerModel = tfl.model({ inputs: input, outputs: [output1, output2] });
        var outerModel = tfl.sequential();
        expect(function () { return outerModel.add(innerModel); })
            .toThrowError(/should have a single output tensor/);
        var outerModel2 = tfl.sequential({ layers: [tfl.layers.dense({ units: 4, inputShape: [8] })] });
        expect(function () { return outerModel2.add(innerModel); })
            .toThrowError(/should have a single output tensor/);
        expect(function () { return tfl.sequential({
            layers: [innerModel]
        }); }).toThrowError(/should have a single output tensor/);
    });
});
test_utils_1.describeMathCPU('modelFromJSON', function () {
    it('reconstitutes pythonic json string', function (done) {
        models_1.modelFromJSON(fakeSequentialModel)
            .then(function (model) {
            expect(model.name).toEqual('test');
            var allZeros = tfjs_core_1.zeros([1, 32]);
            test_utils_1.expectTensorsClose(model.apply(allZeros), allZeros);
            done();
        })
            .catch(done.fail);
    });
    it('reconstitutes mnist non-sequential mode.', function (done) { return __awaiter(_this, void 0, void 0, function () {
        return __generator(this, function (_a) {
            models_1.modelFromJSON(fakeNonSequentialModel)
                .then(function (model) {
                expect(model.name).toEqual('mnist');
                expect(model.layers.length).toEqual(9);
                var prediction = model.predict(tfjs_core_1.zeros([1, 28, 28, 1]));
                expect(prediction.shape).toEqual([1, 10]);
                expect(tfjs_core_1.sum(prediction).dataSync()).toBeCloseTo(1);
                done();
            })
                .catch(done.fail);
            return [2];
        });
    }); });
    it('reconstitutes mnist sequential mode.', function (done) { return __awaiter(_this, void 0, void 0, function () {
        var _this = this;
        return __generator(this, function (_a) {
            models_1.modelFromJSON(fakeMnistModel).then(function (model) { return __awaiter(_this, void 0, void 0, function () {
                var prediction;
                return __generator(this, function (_a) {
                    expect(model.layers.length).toEqual(8);
                    prediction = model.predict(tfjs_core_1.zeros([1, 28, 28, 1]));
                    expect(prediction.shape).toEqual([1, 10]);
                    expect(tfjs_core_1.sum(prediction).dataSync()).toBeCloseTo(1);
                    done();
                    return [2];
                });
            }); });
            return [2];
        });
    }); });
    it('Serialization round-tripping', function (done) {
        models_1.modelFromJSON(fakeRoundtripModel)
            .then(function (model) {
            var serializedModel = model.toJSON();
            expect(typeof serializedModel).toEqual('string');
            var reparsedJson = JSON.parse(serializedModel);
            expect(reparsedJson['class_name'])
                .toEqual(fakeRoundtripModel.modelTopology['class_name']);
            expect(reparsedJson['config'])
                .toEqual(fakeRoundtripModel.modelTopology['config']);
        })
            .then(done)
            .catch(done.fail);
    });
    it('toJSON with returnString = false', function (done) {
        models_1.modelFromJSON(fakeRoundtripModel)
            .then(function (model) {
            var serializedModel = model.toJSON(null, false);
            expect(serializedModel['class_name'])
                .toEqual(fakeRoundtripModel.modelTopology['class_name']);
            expect(serializedModel['config'])
                .toEqual(fakeRoundtripModel.modelTopology['config']);
        })
            .then(done)
            .catch(done.fail);
    });
    it('toJSON return value includes correct versions', function () { return __awaiter(_this, void 0, void 0, function () {
        var model, serializedModel;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0: return [4, models_1.modelFromJSON(fakeRoundtripModel)];
                case 1:
                    model = _a.sent();
                    serializedModel = model.toJSON(null, false);
                    expect(serializedModel['keras_version'])
                        .toEqual("tfjs-layers " + version_1.version);
                    return [2];
            }
        });
    }); });
});
test_utils_1.describeMathCPU('loadModel from URL', function () {
    var setupFakeWeightFiles = function (fileBufferMap) {
        spyOn(window, 'fetch').and.callFake(function (path) {
            return new Response(fileBufferMap[path]);
        });
    };
    var isModelConfigNestedValues = [false, true];
    var pathPrefixes = ['.', './', './model-home', './model-home/'];
    var _loop_1 = function (isModelConfigNested) {
        var _loop_2 = function (pathPrefix) {
            it("pathPrefix=" + pathPrefix, function (done) {
                var path0 = pathPrefix.endsWith('/') ? pathPrefix + "weight_0" :
                    pathPrefix + "/weight_0";
                var path1 = pathPrefix.endsWith('/') ? pathPrefix + "weight_1" :
                    pathPrefix + "/weight_1";
                var fileBufferMap = {};
                fileBufferMap[path0] =
                    tfjs_core_1.ones([32, 32], 'float32').dataSync();
                fileBufferMap[path1] = tfjs_core_1.ones([32], 'float32').dataSync();
                setupFakeWeightFiles(fileBufferMap);
                var denseLayerName = 'dense_' + Math.floor(Math.random() * 1e9);
                var weightsManifest = [
                    {
                        'paths': ['weight_0'],
                        'weights': [{
                                'name': denseLayerName + "/kernel",
                                'dtype': 'float32',
                                'shape': [32, 32]
                            }],
                    },
                    {
                        'paths': ['weight_1'],
                        'weights': [{
                                'name': denseLayerName + "/bias",
                                'dtype': 'float32',
                                'shape': [32]
                            }],
                    }
                ];
                var modelTopology = JSON.parse(JSON.stringify(fakeSequentialModel)).modelTopology;
                modelTopology['config']['layers'][1]['config']['name'] = denseLayerName;
                if (isModelConfigNested) {
                    modelTopology = { 'model_config': modelTopology };
                }
                models_1.modelFromJSON({ modelTopology: modelTopology, weightsManifest: weightsManifest, pathPrefix: pathPrefix })
                    .then(function (model) {
                    test_utils_1.expectTensorsClose(model.weights[0].read(), tfjs_core_1.ones([32, 32], 'float32'));
                    test_utils_1.expectTensorsClose(model.weights[1].read(), tfjs_core_1.ones([32], 'float32'));
                })
                    .then(done)
                    .catch(done.fail);
            });
        };
        for (var _i = 0, pathPrefixes_1 = pathPrefixes; _i < pathPrefixes_1.length; _i++) {
            var pathPrefix = pathPrefixes_1[_i];
            _loop_2(pathPrefix);
        }
    };
    for (var _i = 0, isModelConfigNestedValues_1 = isModelConfigNestedValues; _i < isModelConfigNestedValues_1.length; _i++) {
        var isModelConfigNested = isModelConfigNestedValues_1[_i];
        _loop_1(isModelConfigNested);
    }
    it('load topology and weights from implicit relative http path', function (done) { return __awaiter(_this, void 0, void 0, function () {
        var modelTopology, weightsManifest;
        return __generator(this, function (_a) {
            modelTopology = JSON.parse(JSON.stringify(fakeSequentialModel)).modelTopology;
            weightsManifest = [
                {
                    'paths': ['weight_0'],
                    'weights': [
                        { 'name': "dense_6/kernel", 'dtype': 'float32', 'shape': [32, 32] }
                    ],
                },
                {
                    'paths': ['weight_1'],
                    'weights': [{ 'name': "dense_6/bias", 'dtype': 'float32', 'shape': [32] }],
                }
            ];
            spyOn(window, 'fetch').and.callFake(function (path) {
                if (path === 'model/model.json') {
                    return new Response(JSON.stringify({
                        modelTopology: modelTopology,
                        weightsManifest: weightsManifest,
                    }));
                }
                else if (path === 'model/weight_0') {
                    return new Response(tfjs_core_1.ones([32, 32], 'float32').dataSync());
                }
                else if (path === 'model/weight_1') {
                    return new Response(tfjs_core_1.zeros([32], 'float32').dataSync());
                }
                else {
                    throw new Error("Invalid path: " + path);
                }
            });
            models_1.loadModelInternal('model/model.json')
                .then(function (model) {
                expect(model.layers.length).toEqual(2);
                expect(model.inputs.length).toEqual(1);
                expect(model.inputs[0].shape).toEqual([null, 32]);
                expect(model.outputs.length).toEqual(1);
                expect(model.outputs[0].shape).toEqual([null, 32]);
                var weightValues = model.getWeights();
                expect(weightValues.length).toEqual(2);
                test_utils_1.expectTensorsClose(weightValues[0], tfjs_core_1.ones([32, 32]));
                test_utils_1.expectTensorsClose(weightValues[1], tfjs_core_1.zeros([32]));
                done();
            })
                .catch(function (err) {
                done.fail(err.stack);
            });
            return [2];
        });
    }); });
    it('load topology and weights from implicit relative http path: HDF5 format', function (done) { return __awaiter(_this, void 0, void 0, function () {
        var modelTopology, weightsManifest;
        return __generator(this, function (_a) {
            modelTopology = JSON.parse(JSON.stringify(fakeSequentialModelFromHDF5))
                .modelTopology;
            weightsManifest = [
                {
                    'paths': ['weight_0'],
                    'weights': [
                        { 'name': "dense_1/kernel", 'dtype': 'float32', 'shape': [10, 2] }
                    ],
                },
                {
                    'paths': ['weight_1'],
                    'weights': [{ 'name': "dense_1/bias", 'dtype': 'float32', 'shape': [2] }],
                },
                {
                    'paths': ['weight_2'],
                    'weights': [
                        { 'name': "dense_2/kernel", 'dtype': 'float32', 'shape': [2, 1] }
                    ],
                },
                {
                    'paths': ['weight_3'],
                    'weights': [{ 'name': "dense_2/bias", 'dtype': 'float32', 'shape': [1] }],
                }
            ];
            spyOn(window, 'fetch').and.callFake(function (path) {
                if (path === 'model/model.json') {
                    return new Response(JSON.stringify({
                        modelTopology: modelTopology,
                        weightsManifest: weightsManifest,
                    }));
                }
                else if (path === 'model/weight_0') {
                    return new Response(tfjs_core_1.ones([10, 2], 'float32').dataSync());
                }
                else if (path === 'model/weight_1') {
                    return new Response(tfjs_core_1.zeros([2], 'float32').dataSync());
                }
                else if (path === 'model/weight_2') {
                    return new Response(tfjs_core_1.zeros([2, 1], 'float32').dataSync());
                }
                else if (path === 'model/weight_3') {
                    return new Response(tfjs_core_1.ones([1], 'float32').dataSync());
                }
                else {
                    throw new Error("Invalid path: " + path);
                }
            });
            models_1.loadModelInternal('model/model.json')
                .then(function (model) {
                expect(model.layers.length).toEqual(2);
                expect(model.inputs.length).toEqual(1);
                expect(model.inputs[0].shape).toEqual([null, 10]);
                expect(model.outputs.length).toEqual(1);
                expect(model.outputs[0].shape).toEqual([null, 1]);
                var weightValues = model.getWeights();
                expect(weightValues.length).toEqual(4);
                test_utils_1.expectTensorsClose(weightValues[0], tfjs_core_1.ones([10, 2]));
                test_utils_1.expectTensorsClose(weightValues[1], tfjs_core_1.zeros([2]));
                test_utils_1.expectTensorsClose(weightValues[2], tfjs_core_1.zeros([2, 1]));
                test_utils_1.expectTensorsClose(weightValues[3], tfjs_core_1.ones([1]));
                done();
            })
                .catch(function (err) {
                done.fail(err.stack);
            });
            return [2];
        });
    }); });
    it('load topology and weights with browserHTTPRequest with requestInit', function (done) { return __awaiter(_this, void 0, void 0, function () {
        var modelTopology, weightsManifest, requestHeaders, requestCredentials;
        return __generator(this, function (_a) {
            modelTopology = JSON.parse(JSON.stringify(fakeSequentialModel)).modelTopology;
            weightsManifest = [
                {
                    'paths': ['weight_0'],
                    'weights': [
                        { 'name': "dense_6/kernel", 'dtype': 'float32', 'shape': [32, 32] }
                    ],
                },
                {
                    'paths': ['weight_1'],
                    'weights': [{ 'name': "dense_6/bias", 'dtype': 'float32', 'shape': [32] }],
                }
            ];
            requestHeaders = [];
            requestCredentials = [];
            spyOn(window, 'fetch')
                .and.callFake(function (path, requestInit) {
                if (requestInit != null) {
                    requestHeaders.push(requestInit.headers);
                    requestCredentials.push(requestInit.credentials);
                }
                if (path === 'model/model.json') {
                    return new Response(JSON.stringify({
                        modelTopology: modelTopology,
                        weightsManifest: weightsManifest,
                    }));
                }
                else if (path === 'model/weight_0') {
                    return new Response(tfjs_core_1.ones([32, 32], 'float32').dataSync());
                }
                else if (path === 'model/weight_1') {
                    return new Response(tfjs_core_1.zeros([32], 'float32').dataSync());
                }
                else {
                    throw new Error("Invalid path: " + path);
                }
            });
            models_1.loadModelInternal(tfjs_core_1.io.browserHTTPRequest('model/model.json', {
                headers: { 'header_key_1': 'header_value_1' },
                credentials: 'include',
            }))
                .then(function (model) {
                expect(model.layers.length).toEqual(2);
                expect(model.inputs.length).toEqual(1);
                expect(model.inputs[0].shape).toEqual([null, 32]);
                expect(model.outputs.length).toEqual(1);
                expect(model.outputs[0].shape).toEqual([null, 32]);
                var weightValues = model.getWeights();
                expect(weightValues.length).toEqual(2);
                test_utils_1.expectTensorsClose(weightValues[0], tfjs_core_1.ones([32, 32]));
                test_utils_1.expectTensorsClose(weightValues[1], tfjs_core_1.zeros([32]));
                expect(requestHeaders).toEqual([
                    { 'header_key_1': 'header_value_1' },
                    { 'header_key_1': 'header_value_1' },
                    { 'header_key_1': 'header_value_1' }
                ]);
                expect(requestCredentials).toEqual([
                    'include', 'include', 'include'
                ]);
                done();
            })
                .catch(function (err) {
                done.fail(err.stack);
            });
            return [2];
        });
    }); });
    var httpProtocols = ['http://', 'https://'];
    var _loop_3 = function (protocol) {
        it("load topology and weights: explicit relative " + protocol + " path", function (done) { return __awaiter(_this, void 0, void 0, function () {
            var modelTopology, weightsManifest;
            return __generator(this, function (_a) {
                modelTopology = JSON.parse(JSON.stringify(fakeSequentialModel)).modelTopology;
                weightsManifest = [
                    {
                        'paths': ['weight_0'],
                        'weights': [
                            { 'name': "dense_6/kernel", 'dtype': 'float32', 'shape': [32, 32] }
                        ],
                    },
                    {
                        'paths': ['weight_1'],
                        'weights': [{ 'name': "dense_6/bias", 'dtype': 'float32', 'shape': [32] }],
                    }
                ];
                spyOn(window, 'fetch').and.callFake(function (path) {
                    if (path === protocol + "localhost:8888/models/model.json") {
                        return new Response(JSON.stringify({
                            modelTopology: modelTopology,
                            weightsManifest: weightsManifest,
                        }));
                    }
                    else if (path === protocol + "localhost:8888/models/weight_0") {
                        return new Response(tfjs_core_1.ones([32, 32], 'float32').dataSync());
                    }
                    else if (path === protocol + "localhost:8888/models/weight_1") {
                        return new Response(tfjs_core_1.zeros([32], 'float32').dataSync());
                    }
                    else {
                        throw new Error("Invalid path: " + path);
                    }
                });
                models_1.loadModelInternal(protocol + "localhost:8888/models/model.json")
                    .then(function (model) {
                    expect(model.layers.length).toEqual(2);
                    expect(model.inputs.length).toEqual(1);
                    expect(model.inputs[0].shape).toEqual([null, 32]);
                    expect(model.outputs.length).toEqual(1);
                    expect(model.outputs[0].shape).toEqual([null, 32]);
                    var weightValues = model.getWeights();
                    expect(weightValues.length).toEqual(2);
                    test_utils_1.expectTensorsClose(weightValues[0], tfjs_core_1.ones([32, 32]));
                    test_utils_1.expectTensorsClose(weightValues[1], tfjs_core_1.zeros([32]));
                    done();
                })
                    .catch(function (err) {
                    done.fail(err.stack);
                });
                return [2];
            });
        }); });
    };
    for (var _a = 0, httpProtocols_1 = httpProtocols; _a < httpProtocols_1.length; _a++) {
        var protocol = httpProtocols_1[_a];
        _loop_3(protocol);
    }
    it('Missing weight in manifest leads to error', function (done) {
        setupFakeWeightFiles({
            './weight_0': tfjs_core_1.ones([32, 32], 'float32').dataSync(),
            './weight_1': tfjs_core_1.ones([32], 'float32').dataSync(),
        });
        var denseLayerName = 'dense_' + Math.floor(Math.random() * 1e9);
        var weightsManifest = [
            {
                'paths': ['weight_0'],
                'weights': [{
                        'name': denseLayerName + "/kernel",
                        'dtype': 'float32',
                        'shape': [32, 32]
                    }],
            },
        ];
        var configJson = JSON.parse(JSON.stringify(fakeSequentialModel)).modelTopology;
        configJson['config']['layers'][1]['config']['name'] = denseLayerName;
        models_1.modelFromJSON({ modelTopology: configJson, weightsManifest: weightsManifest, pathPrefix: '.' })
            .then(function () { return done.fail(); })
            .catch(function () { return done(); });
    });
    it('Loads weights despite uniqueified tensor names', function (done) { return __awaiter(_this, void 0, void 0, function () {
        var denseLayerName, weightsManifest, configJson, model1, model2, e_1;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    _a.trys.push([0, 3, , 4]);
                    setupFakeWeightFiles({
                        './weight_0': tfjs_core_1.ones([32, 32], 'float32').dataSync(),
                        './weight_1': tfjs_core_1.ones([32], 'float32').dataSync(),
                    });
                    denseLayerName = 'dense_uniqueify';
                    weightsManifest = [
                        {
                            'paths': ['weight_0'],
                            'weights': [{
                                    'name': denseLayerName + "/kernel",
                                    'dtype': 'float32',
                                    'shape': [32, 32]
                                }],
                        },
                        {
                            'paths': ['weight_1'],
                            'weights': [{
                                    'name': denseLayerName + "/bias",
                                    'dtype': 'float32',
                                    'shape': [32]
                                }],
                        }
                    ];
                    configJson = JSON.parse(JSON.stringify(fakeSequentialModel)).modelTopology;
                    configJson['config']['layers'][1]['config']['name'] = denseLayerName;
                    return [4, models_1.modelFromJSON({ modelTopology: configJson, weightsManifest: weightsManifest, pathPrefix: '.' })];
                case 1:
                    model1 = _a.sent();
                    expect(model1.weights[0].name).toEqual('dense_uniqueify/kernel');
                    expect(model1.weights[0].originalName).toEqual('dense_uniqueify/kernel');
                    expect(model1.weights[1].name).toEqual('dense_uniqueify/bias');
                    expect(model1.weights[1].originalName).toEqual('dense_uniqueify/bias');
                    test_utils_1.expectTensorsClose(model1.weights[0].read(), tfjs_core_1.ones([32, 32], 'float32'));
                    test_utils_1.expectTensorsClose(model1.weights[1].read(), tfjs_core_1.ones([32], 'float32'));
                    return [4, models_1.modelFromJSON({ modelTopology: configJson, weightsManifest: weightsManifest, pathPrefix: '.' })];
                case 2:
                    model2 = _a.sent();
                    expect(model2.weights[0].name).toEqual('dense_uniqueify/kernel_1');
                    expect(model2.weights[0].originalName).toEqual('dense_uniqueify/kernel');
                    expect(model2.weights[1].name).toEqual('dense_uniqueify/bias_1');
                    expect(model2.weights[1].originalName).toEqual('dense_uniqueify/bias');
                    test_utils_1.expectTensorsClose(model2.weights[0].read(), tfjs_core_1.ones([32, 32], 'float32'));
                    test_utils_1.expectTensorsClose(model2.weights[1].read(), tfjs_core_1.ones([32], 'float32'));
                    done();
                    return [3, 4];
                case 3:
                    e_1 = _a.sent();
                    done.fail(e_1.stack);
                    return [3, 4];
                case 4: return [2];
            }
        });
    }); });
    it('Repeated saving and loading of Model works', function () {
        var model1 = tfl.sequential();
        model1.add(tfl.layers.dense({ units: 3, inputShape: [4], activation: 'relu' }));
        model1.add(tfl.layers.dense({ units: 1, activation: 'sigmoid' }));
        var json1 = model1.toJSON(null, false);
        var model2 = serialization_1.deserialize(serialization_utils_1.convertPythonicToTs(json1));
        var json2 = model2.toJSON(null, false);
        expect(json2).toEqual(json1);
    });
});
test_utils_1.describeMathCPU('loadModel from IOHandler', function () {
    var modelTopology = {
        'class_name': 'Sequential',
        'keras_version': '2.1.4',
        'config': [{
                'class_name': 'Dense',
                'config': {
                    'kernel_initializer': {
                        'class_name': 'VarianceScaling',
                        'config': {
                            'distribution': 'uniform',
                            'scale': 1.0,
                            'seed': null,
                            'mode': 'fan_avg'
                        }
                    },
                    'name': 'dense_1',
                    'kernel_constraint': null,
                    'bias_regularizer': null,
                    'bias_constraint': null,
                    'dtype': 'float32',
                    'activation': 'sigmoid',
                    'trainable': true,
                    'kernel_regularizer': null,
                    'bias_initializer': { 'class_name': 'Zeros', 'config': {} },
                    'units': 1,
                    'batch_input_shape': [null, 4],
                    'use_bias': true,
                    'activity_regularizer': null
                }
            }],
        'backend': 'tensorflow'
    };
    var weightSpecs = [
        {
            name: 'dense_1/kernel',
            shape: [4, 1],
            dtype: 'float32',
        },
        {
            name: 'dense_1/bias',
            shape: [1],
            dtype: 'float32',
        }
    ];
    var weightData = new Float32Array([1.1, 2.2, 3.3, 4.4, 5.5]).buffer;
    var IOHandlerForTest = (function () {
        function IOHandlerForTest(includeWeights) {
            if (includeWeights === void 0) { includeWeights = true; }
            this.includeWeights = includeWeights;
        }
        IOHandlerForTest.prototype.load = function () {
            return __awaiter(this, void 0, void 0, function () {
                return __generator(this, function (_a) {
                    return [2, this.includeWeights ? { modelTopology: modelTopology, weightSpecs: weightSpecs, weightData: weightData } :
                            { modelTopology: modelTopology }];
                });
            });
        };
        return IOHandlerForTest;
    }());
    var IOHandlerWithoutLoad = (function () {
        function IOHandlerWithoutLoad() {
        }
        return IOHandlerWithoutLoad;
    }());
    it('load topology and weights', function (done) { return __awaiter(_this, void 0, void 0, function () {
        return __generator(this, function (_a) {
            models_1.loadModelInternal(new IOHandlerForTest(true))
                .then(function (model) {
                expect(model.layers.length).toEqual(1);
                expect(model.inputs.length).toEqual(1);
                expect(model.inputs[0].shape).toEqual([null, 4]);
                expect(model.outputs.length).toEqual(1);
                expect(model.outputs[0].shape).toEqual([null, 1]);
                var weightValues = model.getWeights();
                expect(weightValues.length).toEqual(2);
                test_utils_1.expectTensorsClose(weightValues[0], tfjs_core_1.tensor2d([1.1, 2.2, 3.3, 4.4], [4, 1]));
                test_utils_1.expectTensorsClose(weightValues[1], tfjs_core_1.tensor1d([5.5]));
                done();
            })
                .catch(function (err) {
                done.fail(err.stack);
            });
            return [2];
        });
    }); });
    it('load topology only', function (done) { return __awaiter(_this, void 0, void 0, function () {
        return __generator(this, function (_a) {
            models_1.loadModelInternal(new IOHandlerForTest(false))
                .then(function (model) {
                expect(model.layers.length).toEqual(1);
                expect(model.inputs.length).toEqual(1);
                expect(model.inputs[0].shape).toEqual([null, 4]);
                expect(model.outputs.length).toEqual(1);
                expect(model.outputs[0].shape).toEqual([null, 1]);
                done();
            })
                .catch(function (err) {
                done.fail(err.stack);
            });
            return [2];
        });
    }); });
    it('IOHandler without load method causes error', function (done) { return __awaiter(_this, void 0, void 0, function () {
        return __generator(this, function (_a) {
            models_1.loadModelInternal(new IOHandlerWithoutLoad())
                .then(function (model) {
                done.fail('Loading with an IOHandler without load method succeeded ' +
                    'unexpectedly.');
            })
                .catch(function (err) {
                expect(err.message).toMatch(/does not have .*load.* method/);
                done();
            });
            return [2];
        });
    }); });
});
test_utils_1.describeMathCPUAndGPU('Sequential', function () {
    var inputShape = [1, 6];
    var batchInputShape = [1].concat(inputShape);
    var firstReshape = [2, 3];
    var secondReshape = [3, 2];
    var layers = [
        new core_1.Reshape({ targetShape: firstReshape, batchInputShape: batchInputShape, name: 'layer1' }),
        new core_1.Reshape({ targetShape: secondReshape, name: 'layer2' })
    ];
    function getInputs() {
        return tfjs_core_1.ones(batchInputShape);
    }
    function getExpectedOutputs() {
        return tfjs_core_1.ones([1].concat(secondReshape));
    }
    it('throws an exception if the first layer is not an input layer', function () {
        var layer = new core_1.Reshape({ targetShape: firstReshape });
        expect(function () { return tfl.sequential({ layers: [layer] }); })
            .toThrowError(/The first layer in a Sequential model must get an `inputShape`/);
    });
    it('can accept a list of layers in constructor', function () {
        var model = tfl.sequential({ layers: layers });
        expect(model.layers).toEqual(layers);
    });
    it('can add layers', function () {
        var model = tfl.sequential();
        for (var _i = 0, layers_1 = layers; _i < layers_1.length; _i++) {
            var layer = layers_1[_i];
            model.add(layer);
        }
        expect(model.layers).toEqual(layers);
    });
    it('can pop layers', function () {
        var model = tfl.sequential({ layers: layers });
        model.pop();
        expect(model.layers).toEqual(layers.slice(0, 1));
    });
    it('Incompatible inputShape leads to warning', function () {
        var recordedWarnMessage;
        spyOn(console, 'warn')
            .and.callFake(function (message) { return recordedWarnMessage = message; });
        tfl.sequential({
            layers: [
                tfl.layers.reshape({ targetShape: [8], inputShape: [2, 4] }),
                tfl.layers.dense({ units: 1, inputShape: [7] })
            ]
        });
        expect(recordedWarnMessage)
            .toMatch(/shape of the input tensor .*null,8.* not match .*null,7.*/);
    });
    it('Incompatible inputShape rank leads to warning', function () {
        var recordedWarnMessage;
        spyOn(console, 'warn')
            .and.callFake(function (message) { return recordedWarnMessage = message; });
        tfl.sequential({
            layers: [
                tfl.layers.reshape({ targetShape: [8], inputShape: [2, 4] }),
                tfl.layers.dense({ units: 1, inputShape: [3, 7] })
            ]
        });
        expect(recordedWarnMessage)
            .toMatch(/rank .*null,8.* does not match .*null,3,7.* /);
    });
    it('Compatible inputShape leads to NO warning', function () {
        var recordedWarnMessage;
        spyOn(console, 'warn')
            .and.callFake(function (message) { return recordedWarnMessage = message; });
        tfl.sequential({
            layers: [
                tfl.layers.reshape({ targetShape: [8], inputShape: [2, 4] }),
                tfl.layers.dense({ units: 1, inputShape: [8] })
            ]
        });
        expect(recordedWarnMessage).toEqual(undefined);
    });
    it('throws error if try to pop too many layers', function () {
        var model = tfl.sequential();
        expect(function () { return model.pop(); }).toThrowError(/There are no layers in the model/);
    });
    it('apply() threads data through the model.', function () {
        var model = tfl.sequential({ layers: layers });
        test_utils_1.expectTensorsClose(model.apply(getInputs()), getExpectedOutputs());
    });
    it('predict() threads data through the model.', function () {
        var model = tfl.sequential({ layers: layers });
        test_utils_1.expectTensorsClose(model.predict(getInputs()), getExpectedOutputs());
    });
    var dtypes = ['int32', 'float32'];
    var _loop_4 = function (dtype) {
        it("predict() works with input dtype " + dtype + ".", function () {
            var embModel = tfl.sequential();
            embModel.add(tfl.layers.embedding({ inputShape: [1], inputDim: 10, outputDim: 2 }));
            var x = tfjs_core_1.tensor2d([[0], [0], [1]], [3, 1], dtype);
            var y = embModel.predict(x);
            expect(y.dtype).toBe('float32');
        });
        it("fit() works with input dtype " + dtype + ".", function () {
            var embModel = tfl.sequential();
            embModel.add(tfl.layers.embedding({ inputShape: [1], inputDim: 10, outputDim: 2 }));
            embModel.compile({ optimizer: 'sgd', loss: 'meanSquaredError' });
            var x = tfjs_core_1.tensor2d([[0]], [1, 1], dtype);
            var y = tfjs_core_1.tensor2d([[0.5, 0.5]], [1, 2], 'float32');
            embModel.fit(x, y);
        });
    };
    for (var _i = 0, dtypes_1 = dtypes; _i < dtypes_1.length; _i++) {
        var dtype = dtypes_1[_i];
        _loop_4(dtype);
    }
    it('predictOnBatch() threads data through the model.', function () {
        var batchSize = 10;
        var inputShape = [1, 6];
        var batchInputShape = [batchSize].concat(inputShape);
        var firstReshape = [2, 3];
        var secondReshape = [3, 2];
        var layers = [
            new core_1.Reshape({ targetShape: firstReshape, batchInputShape: batchInputShape, name: 'layer1' }),
            new core_1.Reshape({ targetShape: secondReshape, name: 'layer2' })
        ];
        var inputBatch = tfjs_core_1.ones([batchSize].concat(inputShape));
        var expectedOutput = tfjs_core_1.ones([batchSize].concat(secondReshape));
        var model = tfl.sequential({ layers: layers });
        test_utils_1.expectTensorsClose(model.predictOnBatch(inputBatch), expectedOutput);
    });
    it('compile() and fit()', function () { return __awaiter(_this, void 0, void 0, function () {
        var batchSize, inputSize, xs, ys, denseLayer1, denseLayer2, model, history;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    batchSize = 5;
                    inputSize = 4;
                    xs = tfjs_core_1.ones([batchSize, inputSize]);
                    ys = tfjs_core_1.ones([batchSize, 1]);
                    denseLayer1 = tfl.layers.dense({
                        units: 3,
                        useBias: false,
                        kernelInitializer: 'ones',
                        inputShape: [inputSize]
                    });
                    denseLayer2 = tfl.layers.dense({ units: 1, useBias: false, kernelInitializer: 'ones' });
                    model = tfl.sequential({ layers: [denseLayer1, denseLayer2] });
                    model.compile({ optimizer: 'sgd', loss: 'meanSquaredError' });
                    return [4, model.fit(xs, ys, { batchSize: batchSize, epochs: 2 })];
                case 1:
                    history = _a.sent();
                    expect(history.history['loss'][0]).toBe(121);
                    expect(history.history['loss'][1]).toBeCloseTo(0.015178224071860313);
                    return [2];
            }
        });
    }); });
    it('calling fit twice in a row leads to error', function () { return __awaiter(_this, void 0, void 0, function () {
        var batchSize, inputSize, xs, ys, denseLayer1, denseLayer2, model, errorCaught, err_1;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    batchSize = 5;
                    inputSize = 4;
                    xs = tfjs_core_1.ones([batchSize, inputSize]);
                    ys = tfjs_core_1.ones([batchSize, 1]);
                    denseLayer1 = tfl.layers.dense({ units: 3, inputShape: [inputSize] });
                    denseLayer2 = tfl.layers.dense({ units: 1 });
                    model = tfl.sequential({ layers: [denseLayer1, denseLayer2] });
                    model.compile({ optimizer: 'sgd', loss: 'meanSquaredError' });
                    model.fit(xs, ys, { batchSize: batchSize, epochs: 8 });
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 3, , 4]);
                    return [4, model.fit(xs, ys)];
                case 2:
                    _a.sent();
                    return [3, 4];
                case 3:
                    err_1 = _a.sent();
                    errorCaught = err_1;
                    return [3, 4];
                case 4:
                    expect(errorCaught.message)
                        .toEqual('Cannot start training because another fit() call is ongoing.');
                    return [2];
            }
        });
    }); });
    it('Stop Sequential.fit() using non-class callback function', function () { return __awaiter(_this, void 0, void 0, function () {
        var _this = this;
        var batchSize, inputSize, xs, ys, denseLayer1, denseLayer2, model, numEpochsDone, epochs, stopAfterEpoch, history;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    batchSize = 5;
                    inputSize = 4;
                    xs = tfjs_core_1.ones([batchSize, inputSize]);
                    ys = tfjs_core_1.ones([batchSize, 1]);
                    denseLayer1 = tfl.layers.dense({ units: 3, inputShape: [inputSize] });
                    denseLayer2 = tfl.layers.dense({ units: 1 });
                    model = tfl.sequential({ layers: [denseLayer1, denseLayer2] });
                    model.compile({ optimizer: 'sgd', loss: 'meanSquaredError' });
                    numEpochsDone = 0;
                    epochs = 8;
                    stopAfterEpoch = 3;
                    return [4, model.fit(xs, ys, {
                            epochs: epochs,
                            callbacks: {
                                onEpochEnd: function (epoch, logs) { return __awaiter(_this, void 0, void 0, function () {
                                    return __generator(this, function (_a) {
                                        numEpochsDone++;
                                        if (epoch === stopAfterEpoch) {
                                            model.stopTraining = true;
                                        }
                                        return [2];
                                    });
                                }); }
                            }
                        })];
                case 1:
                    history = _a.sent();
                    expect(numEpochsDone).toEqual(stopAfterEpoch + 1);
                    expect(history.history.loss.length).toEqual(stopAfterEpoch + 1);
                    return [4, model.fit(xs, ys, { epochs: 2 })];
                case 2:
                    history = _a.sent();
                    expect(history.history.loss.length).toEqual(2);
                    return [2];
            }
        });
    }); });
    it('Calling evaluate before compile leads to error', function () {
        var batchSize = 5;
        var inputSize = 4;
        var denseLayer1 = tfl.layers.dense({ units: 3, inputShape: [inputSize] });
        var model = tfl.sequential({ layers: [denseLayer1] });
        var xs = tfjs_core_1.ones([batchSize, inputSize]);
        var ys = tfjs_core_1.ones([batchSize, 1]);
        expect(function () { return model.evaluate(xs, ys); })
            .toThrowError(/needs to be compiled before/);
    });
    it('compile() and evaluate()', function () {
        var batchSize = 5;
        var inputSize = 4;
        var xs = tfjs_core_1.ones([batchSize, inputSize]);
        var ys = tfjs_core_1.ones([batchSize, 1]);
        var denseLayer1 = tfl.layers.dense({
            units: 3,
            useBias: false,
            kernelInitializer: 'ones',
            inputShape: [inputSize]
        });
        var denseLayer2 = tfl.layers.dense({ units: 1, useBias: false, kernelInitializer: 'ones' });
        var model = tfl.sequential({ layers: [denseLayer1, denseLayer2] });
        model.compile({ optimizer: 'sgd', loss: 'meanSquaredError' });
        var losses = model.evaluate(xs, ys, { batchSize: batchSize });
        test_utils_1.expectTensorsClose(losses, tfjs_core_1.scalar(121));
    });
    it('getConfig returns an Array', function () {
        var model = tfl.sequential({ layers: layers });
        var config = model.getConfig();
        expect(Array.isArray(config)).toEqual(true);
        expect(config.length).toEqual(layers.length);
    });
});
var fakeSequentialModel = {
    modelTopology: {
        'class_name': 'Model',
        'keras_version': '2.0.7',
        'config': {
            'layers': [
                {
                    'class_name': 'InputLayer',
                    'config': {
                        'dtype': 'float32',
                        'batch_input_shape': [null, 32],
                        'name': 'input_6',
                        'sparse': false
                    },
                    'inbound_nodes': [],
                    'name': 'input_6'
                },
                {
                    'class_name': 'Dense',
                    'config': {
                        'units': 32,
                        'bias_constraint': null,
                        'use_bias': true,
                        'kernel_initializer': {
                            'class_name': 'VarianceScaling',
                            'config': {
                                'distribution': 'uniform',
                                'scale': 1,
                                'seed': null,
                                'mode': 'fan_avg'
                            }
                        },
                        'activation': 'linear',
                        'bias_regularizer': null,
                        'activity_regularizer': null,
                        'trainable': true,
                        'kernel_constraint': null,
                        'kernel_regularizer': null,
                        'name': 'dense_6',
                        'bias_initializer': { 'class_name': 'Zeros', 'config': {} }
                    },
                    'inbound_nodes': [[['input_6', 0, 0, {}]]],
                    'name': 'dense_6'
                }
            ],
            'input_layers': [['input_6', 0, 0]],
            'output_layers': [['dense_6', 0, 0]],
            'name': 'test'
        },
        'backend': 'tensorflow'
    }
};
var fakeSequentialModelFromHDF5 = {
    modelTopology: {
        'backend': 'tensorflow',
        'keras_version': '2.1.4',
        'model_config': {
            'class_name': 'Sequential',
            'config': [
                {
                    'class_name': 'Dense',
                    'config': {
                        'kernel_initializer': {
                            'class_name': 'VarianceScaling',
                            'config': {
                                'distribution': 'uniform',
                                'scale': 1.0,
                                'seed': null,
                                'mode': 'fan_avg'
                            }
                        },
                        'name': 'dense_1',
                        'kernel_constraint': null,
                        'bias_regularizer': null,
                        'bias_constraint': null,
                        'dtype': 'float32',
                        'activation': 'relu',
                        'trainable': true,
                        'kernel_regularizer': null,
                        'bias_initializer': { 'class_name': 'Zeros', 'config': {} },
                        'units': 2,
                        'batch_input_shape': [null, 10],
                        'use_bias': true,
                        'activity_regularizer': null
                    }
                },
                {
                    'class_name': 'Dense',
                    'config': {
                        'kernel_initializer': {
                            'class_name': 'VarianceScaling',
                            'config': {
                                'distribution': 'uniform',
                                'scale': 1.0,
                                'seed': null,
                                'mode': 'fan_avg'
                            }
                        },
                        'name': 'dense_2',
                        'kernel_constraint': null,
                        'bias_regularizer': null,
                        'bias_constraint': null,
                        'activation': 'sigmoid',
                        'trainable': true,
                        'kernel_regularizer': null,
                        'bias_initializer': { 'class_name': 'Zeros', 'config': {} },
                        'units': 1,
                        'use_bias': true,
                        'activity_regularizer': null
                    }
                }
            ]
        },
    }
};
var fakeNonSequentialModel = {
    modelTopology: {
        'backend': 'tensorflow',
        'class_name': 'Model',
        'keras_version': '2.1.1',
        'config': {
            'name': 'mnist',
            'output_layers': [['dense_16', 0, 0]],
            'layers': [
                {
                    'class_name': 'InputLayer',
                    'name': 'input_6',
                    'inbound_nodes': [],
                    'config': {
                        'batch_input_shape': [null, 28, 28, 1],
                        'sparse': false,
                        'name': 'input_6',
                        'dtype': 'float32'
                    }
                },
                {
                    'class_name': 'Conv2D',
                    'name': 'conv2d_15',
                    'inbound_nodes': [[['input_6', 0, 0, {}]]],
                    'config': {
                        'bias_initializer': { 'class_name': 'Zeros', 'config': {} },
                        'padding': 'valid',
                        'use_bias': true,
                        'strides': [1, 1],
                        'bias_regularizer': null,
                        'activity_regularizer': null,
                        'kernel_initializer': {
                            'class_name': 'VarianceScaling',
                            'config': {
                                'distribution': 'uniform',
                                'scale': 1.0,
                                'mode': 'fan_avg',
                                'seed': null
                            }
                        },
                        'data_format': 'channels_last',
                        'dilation_rate': [1, 1],
                        'kernel_constraint': null,
                        'kernel_regularizer': null,
                        'kernel_size': [3, 3],
                        'activation': 'relu',
                        'name': 'conv2d_15',
                        'filters': 32,
                        'trainable': true,
                        'bias_constraint': null
                    }
                },
                {
                    'class_name': 'Conv2D',
                    'name': 'conv2d_16',
                    'inbound_nodes': [[['conv2d_15', 0, 0, {}]]],
                    'config': {
                        'bias_initializer': { 'class_name': 'Zeros', 'config': {} },
                        'padding': 'valid',
                        'use_bias': true,
                        'strides': [1, 1],
                        'bias_regularizer': null,
                        'activity_regularizer': null,
                        'kernel_initializer': {
                            'class_name': 'VarianceScaling',
                            'config': {
                                'distribution': 'uniform',
                                'scale': 1.0,
                                'mode': 'fan_avg',
                                'seed': null
                            }
                        },
                        'data_format': 'channels_last',
                        'dilation_rate': [1, 1],
                        'kernel_constraint': null,
                        'kernel_regularizer': null,
                        'kernel_size': [3, 3],
                        'activation': 'relu',
                        'name': 'conv2d_16',
                        'filters': 64,
                        'trainable': true,
                        'bias_constraint': null
                    }
                },
                {
                    'class_name': 'MaxPooling2D',
                    'name': 'max_pooling2d_8',
                    'inbound_nodes': [[['conv2d_16', 0, 0, {}]]],
                    'config': {
                        'padding': 'valid',
                        'strides': [2, 2],
                        'pool_size': [2, 2],
                        'data_format': 'channels_last',
                        'name': 'max_pooling2d_8',
                        'trainable': true
                    }
                },
                {
                    'class_name': 'Dropout',
                    'name': 'dropout_15',
                    'inbound_nodes': [[['max_pooling2d_8', 0, 0, {}]]],
                    'config': {
                        'rate': 0.25,
                        'noise_shape': null,
                        'name': 'dropout_15',
                        'trainable': true,
                        'seed': null
                    }
                },
                {
                    'class_name': 'Flatten',
                    'name': 'flatten_8',
                    'inbound_nodes': [[['dropout_15', 0, 0, {}]]],
                    'config': { 'name': 'flatten_8', 'trainable': true }
                },
                {
                    'class_name': 'Dense',
                    'name': 'dense_15',
                    'inbound_nodes': [[['flatten_8', 0, 0, {}]]],
                    'config': {
                        'use_bias': true,
                        'bias_regularizer': null,
                        'kernel_initializer': {
                            'class_name': 'VarianceScaling',
                            'config': {
                                'distribution': 'uniform',
                                'scale': 1.0,
                                'mode': 'fan_avg',
                                'seed': null
                            }
                        },
                        'bias_initializer': { 'class_name': 'Zeros', 'config': {} },
                        'kernel_constraint': null,
                        'bias_constraint': null,
                        'kernel_regularizer': null,
                        'activation': 'relu',
                        'name': 'dense_15',
                        'activity_regularizer': null,
                        'trainable': true,
                        'units': 128
                    }
                },
                {
                    'class_name': 'Dropout',
                    'name': 'dropout_16',
                    'inbound_nodes': [[['dense_15', 0, 0, {}]]],
                    'config': {
                        'rate': 0.5,
                        'noise_shape': null,
                        'name': 'dropout_16',
                        'trainable': true,
                        'seed': null
                    }
                },
                {
                    'class_name': 'Dense',
                    'name': 'dense_16',
                    'inbound_nodes': [[['dropout_16', 0, 0, {}]]],
                    'config': {
                        'use_bias': true,
                        'bias_regularizer': null,
                        'kernel_initializer': {
                            'class_name': 'VarianceScaling',
                            'config': {
                                'distribution': 'uniform',
                                'scale': 1.0,
                                'mode': 'fan_avg',
                                'seed': null
                            }
                        },
                        'bias_initializer': { 'class_name': 'Zeros', 'config': {} },
                        'kernel_constraint': null,
                        'bias_constraint': null,
                        'kernel_regularizer': null,
                        'activation': 'softmax',
                        'name': 'dense_16',
                        'activity_regularizer': null,
                        'trainable': true,
                        'units': 10
                    }
                }
            ],
            'input_layers': [['input_6', 0, 0]]
        }
    }
};
var fakeMnistModel = {
    modelTopology: {
        'backend': 'tensorflow',
        'config': [
            {
                'config': {
                    'kernel_size': [3, 3],
                    'use_bias': true,
                    'batch_input_shape': [null, 28, 28, 1],
                    'filters': 32,
                    'kernel_regularizer': null,
                    'dilation_rate': [1, 1],
                    'strides': [1, 1],
                    'padding': 'valid',
                    'bias_constraint': null,
                    'kernel_constraint': null,
                    'data_format': 'channels_last',
                    'trainable': true,
                    'activation': 'relu',
                    'dtype': 'float32',
                    'bias_initializer': { 'config': {}, 'class_name': 'Zeros' },
                    'bias_regularizer': null,
                    'name': 'conv2d_1',
                    'kernel_initializer': {
                        'config': {
                            'scale': 1.0,
                            'mode': 'fan_avg',
                            'seed': null,
                            'distribution': 'uniform'
                        },
                        'class_name': 'VarianceScaling'
                    },
                    'activity_regularizer': null
                },
                'class_name': 'Conv2D'
            },
            {
                'config': {
                    'kernel_size': [3, 3],
                    'use_bias': true,
                    'filters': 64,
                    'kernel_regularizer': null,
                    'dilation_rate': [1, 1],
                    'strides': [1, 1],
                    'padding': 'valid',
                    'bias_constraint': null,
                    'data_format': 'channels_last',
                    'trainable': true,
                    'activation': 'relu',
                    'kernel_constraint': null,
                    'bias_initializer': { 'config': {}, 'class_name': 'Zeros' },
                    'bias_regularizer': null,
                    'name': 'conv2d_2',
                    'kernel_initializer': {
                        'config': {
                            'scale': 1.0,
                            'mode': 'fan_avg',
                            'seed': null,
                            'distribution': 'uniform'
                        },
                        'class_name': 'VarianceScaling'
                    },
                    'activity_regularizer': null
                },
                'class_name': 'Conv2D'
            },
            {
                'config': {
                    'strides': [2, 2],
                    'padding': 'valid',
                    'pool_size': [2, 2],
                    'data_format': 'channels_last',
                    'trainable': true,
                    'name': 'max_pooling2d_1'
                },
                'class_name': 'MaxPooling2D'
            },
            {
                'config': {
                    'seed': null,
                    'name': 'dropout_1',
                    'trainable': true,
                    'noise_shape': null,
                    'rate': 0.25
                },
                'class_name': 'Dropout'
            },
            {
                'config': { 'name': 'flatten_1', 'trainable': true },
                'class_name': 'Flatten'
            },
            {
                'config': {
                    'use_bias': true,
                    'units': 128,
                    'bias_initializer': { 'config': {}, 'class_name': 'Zeros' },
                    'kernel_regularizer': null,
                    'bias_regularizer': null,
                    'trainable': true,
                    'activation': 'relu',
                    'bias_constraint': null,
                    'kernel_constraint': null,
                    'name': 'dense_1',
                    'kernel_initializer': {
                        'config': {
                            'scale': 1.0,
                            'mode': 'fan_avg',
                            'seed': null,
                            'distribution': 'uniform'
                        },
                        'class_name': 'VarianceScaling'
                    },
                    'activity_regularizer': null
                },
                'class_name': 'Dense'
            },
            {
                'config': {
                    'seed': null,
                    'name': 'dropout_2',
                    'trainable': true,
                    'noise_shape': null,
                    'rate': 0.5
                },
                'class_name': 'Dropout'
            },
            {
                'config': {
                    'use_bias': true,
                    'units': 10,
                    'bias_initializer': { 'config': {}, 'class_name': 'Zeros' },
                    'kernel_regularizer': null,
                    'bias_regularizer': null,
                    'trainable': true,
                    'activation': 'softmax',
                    'bias_constraint': null,
                    'kernel_constraint': null,
                    'name': 'dense_2',
                    'kernel_initializer': {
                        'config': {
                            'scale': 1.0,
                            'mode': 'fan_avg',
                            'seed': null,
                            'distribution': 'uniform'
                        },
                        'class_name': 'VarianceScaling'
                    },
                    'activity_regularizer': null
                },
                'class_name': 'Dense'
            }
        ],
        'keras_version': '2.1.1',
        'class_name': 'Sequential'
    }
};
var fakeRoundtripModel = {
    modelTopology: {
        'backend': 'tensorflow',
        'class_name': 'Model',
        'keras_version': '2.1.1',
        'config': {
            'name': 'mnist',
            'output_layers': [['dense_16', 0, 0]],
            'layers': [
                {
                    'class_name': 'InputLayer',
                    'name': 'input_6',
                    'inbound_nodes': [],
                    'config': {
                        'batch_input_shape': [null, 28, 28, 1],
                        'sparse': false,
                        'name': 'input_6',
                        'dtype': 'float32'
                    }
                },
                {
                    'class_name': 'Conv2D',
                    'name': 'conv2d_15',
                    'inbound_nodes': [[['input_6', 0, 0, {}]]],
                    'config': {
                        'bias_initializer': { 'class_name': 'Zeros', 'config': {} },
                        'padding': 'valid',
                        'use_bias': true,
                        'strides': [1, 1],
                        'bias_regularizer': null,
                        'activity_regularizer': null,
                        'kernel_initializer': {
                            'class_name': 'VarianceScaling',
                            'config': {
                                'distribution': 'uniform',
                                'scale': 1.0,
                                'mode': 'fan_avg',
                                'seed': null
                            }
                        },
                        'data_format': 'channels_last',
                        'dilation_rate': [1, 1],
                        'kernel_constraint': null,
                        'kernel_regularizer': null,
                        'kernel_size': [3, 3],
                        'activation': 'relu',
                        'name': 'conv2d_15',
                        'filters': 32,
                        'trainable': true,
                        'bias_constraint': null
                    }
                },
                {
                    'class_name': 'Conv2D',
                    'name': 'conv2d_16',
                    'inbound_nodes': [[['conv2d_15', 0, 0, {}]]],
                    'config': {
                        'bias_initializer': { 'class_name': 'Zeros', 'config': {} },
                        'padding': 'valid',
                        'use_bias': true,
                        'strides': [1, 1],
                        'bias_regularizer': null,
                        'activity_regularizer': null,
                        'kernel_initializer': {
                            'class_name': 'VarianceScaling',
                            'config': {
                                'distribution': 'uniform',
                                'scale': 1.0,
                                'mode': 'fan_avg',
                                'seed': null
                            }
                        },
                        'data_format': 'channels_last',
                        'dilation_rate': [1, 1],
                        'kernel_constraint': null,
                        'kernel_regularizer': null,
                        'kernel_size': [3, 3],
                        'activation': 'relu',
                        'name': 'conv2d_16',
                        'filters': 64,
                        'trainable': true,
                        'bias_constraint': null
                    }
                },
                {
                    'class_name': 'MaxPooling2D',
                    'name': 'max_pooling2d_8',
                    'inbound_nodes': [[['conv2d_16', 0, 0, {}]]],
                    'config': {
                        'padding': 'valid',
                        'strides': [2, 2],
                        'pool_size': [2, 2],
                        'data_format': 'channels_last',
                        'name': 'max_pooling2d_8',
                        'trainable': true
                    }
                },
                {
                    'class_name': 'Dropout',
                    'name': 'dropout_15',
                    'inbound_nodes': [[['max_pooling2d_8', 0, 0, {}]]],
                    'config': {
                        'rate': 0.25,
                        'noise_shape': null,
                        'name': 'dropout_15',
                        'trainable': true,
                        'seed': null
                    }
                },
                {
                    'class_name': 'Flatten',
                    'name': 'flatten_8',
                    'inbound_nodes': [[['dropout_15', 0, 0, {}]]],
                    'config': { 'name': 'flatten_8', 'trainable': true }
                },
                {
                    'class_name': 'Dense',
                    'name': 'dense_15',
                    'inbound_nodes': [[['flatten_8', 0, 0, {}]]],
                    'config': {
                        'use_bias': true,
                        'bias_regularizer': null,
                        'kernel_initializer': {
                            'class_name': 'VarianceScaling',
                            'config': {
                                'distribution': 'uniform',
                                'scale': 1.0,
                                'mode': 'fan_avg',
                                'seed': null
                            }
                        },
                        'bias_initializer': { 'class_name': 'Zeros', 'config': {} },
                        'kernel_constraint': null,
                        'bias_constraint': null,
                        'kernel_regularizer': null,
                        'activation': 'relu',
                        'name': 'dense_15',
                        'activity_regularizer': null,
                        'trainable': true,
                        'units': 128
                    }
                },
                {
                    'class_name': 'Dropout',
                    'name': 'dropout_16',
                    'inbound_nodes': [[['dense_15', 0, 0, {}]]],
                    'config': {
                        'rate': 0.5,
                        'noise_shape': null,
                        'name': 'dropout_16',
                        'trainable': true,
                        'seed': null
                    }
                },
                {
                    'class_name': 'Dense',
                    'name': 'dense_16',
                    'inbound_nodes': [[['dropout_16', 0, 0, {}]]],
                    'config': {
                        'use_bias': true,
                        'bias_regularizer': null,
                        'kernel_initializer': {
                            'class_name': 'VarianceScaling',
                            'config': {
                                'distribution': 'uniform',
                                'scale': 1.0,
                                'mode': 'fan_avg',
                                'seed': null
                            }
                        },
                        'bias_initializer': { 'class_name': 'Zeros', 'config': {} },
                        'kernel_constraint': null,
                        'bias_constraint': null,
                        'kernel_regularizer': null,
                        'activation': 'softmax',
                        'name': 'dense_16',
                        'activity_regularizer': null,
                        'trainable': true,
                        'units': 10
                    }
                }
            ],
            'input_layers': [['input_6', 0, 0]]
        }
    }
};
//# sourceMappingURL=models_test.js.map