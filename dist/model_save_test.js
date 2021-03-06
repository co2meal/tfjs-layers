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
var test_utils_1 = require("./utils/test_utils");
test_utils_1.describeMathCPUAndGPU('Model.save', function () {
    var IOHandlerForTest = (function () {
        function IOHandlerForTest() {
        }
        IOHandlerForTest.prototype.save = function (modelArtifacts) {
            return __awaiter(this, void 0, void 0, function () {
                return __generator(this, function (_a) {
                    this.savedArtifacts = modelArtifacts;
                    return [2, { modelArtifactsInfo: null }];
                });
            });
        };
        return IOHandlerForTest;
    }());
    var EmptyIOHandler = (function () {
        function EmptyIOHandler() {
        }
        return EmptyIOHandler;
    }());
    it('Saving all weights succeeds', function (done) { return __awaiter(_this, void 0, void 0, function () {
        var model, handler;
        return __generator(this, function (_a) {
            model = tfl.sequential();
            model.add(tfl.layers.dense({ units: 3, inputShape: [5] }));
            handler = new IOHandlerForTest();
            model.save(handler)
                .then(function (saveResult) {
                expect(handler.savedArtifacts.modelTopology)
                    .toEqual(model.toJSON(null, false));
                expect(handler.savedArtifacts.weightSpecs.length).toEqual(2);
                expect(handler.savedArtifacts.weightSpecs[0].name.indexOf('/kernel'))
                    .toBeGreaterThan(0);
                expect(handler.savedArtifacts.weightSpecs[0].shape).toEqual([5, 3]);
                expect(handler.savedArtifacts.weightSpecs[0].dtype)
                    .toEqual('float32');
                expect(handler.savedArtifacts.weightSpecs[1].name.indexOf('/bias'))
                    .toBeGreaterThan(0);
                expect(handler.savedArtifacts.weightSpecs[1].shape).toEqual([3]);
                expect(handler.savedArtifacts.weightSpecs[1].dtype)
                    .toEqual('float32');
                done();
            })
                .catch(function (err) {
                console.error(err.stack);
            });
            return [2];
        });
    }); });
    it('Saving only trainable weights succeeds', function (done) { return __awaiter(_this, void 0, void 0, function () {
        var model, handler;
        return __generator(this, function (_a) {
            model = tfl.sequential();
            model.add(tfl.layers.dense({ units: 3, inputShape: [5], trainable: false }));
            model.add(tfl.layers.dense({ units: 2 }));
            handler = new IOHandlerForTest();
            model.save(handler, { trainableOnly: true })
                .then(function (saveResult) {
                expect(handler.savedArtifacts.modelTopology)
                    .toEqual(model.toJSON(null, false));
                expect(handler.savedArtifacts.weightSpecs.length).toEqual(2);
                expect(handler.savedArtifacts.weightSpecs[0].name.indexOf('/kernel'))
                    .toBeGreaterThan(0);
                expect(handler.savedArtifacts.weightSpecs[0].shape).toEqual([3, 2]);
                expect(handler.savedArtifacts.weightSpecs[0].dtype)
                    .toEqual('float32');
                expect(handler.savedArtifacts.weightSpecs[1].name.indexOf('/bias'))
                    .toBeGreaterThan(0);
                expect(handler.savedArtifacts.weightSpecs[1].shape).toEqual([2]);
                expect(handler.savedArtifacts.weightSpecs[1].dtype)
                    .toEqual('float32');
                done();
            })
                .catch(function (err) {
                console.error(err.stack);
            });
            return [2];
        });
    }); });
    it('Saving to a handler without save method fails', function (done) { return __awaiter(_this, void 0, void 0, function () {
        var model, handler;
        return __generator(this, function (_a) {
            model = tfl.sequential();
            model.add(tfl.layers.dense({ units: 3, inputShape: [5] }));
            handler = new EmptyIOHandler();
            model.save(handler)
                .then(function (saveResult) {
                fail('Saving with an IOHandler without `save` succeeded ' +
                    'unexpectedly.');
            })
                .catch(function (err) {
                expect(err.message)
                    .toEqual('Model.save() cannot proceed because the IOHandler ' +
                    'provided does not have the `save` attribute defined.');
                done();
            });
            return [2];
        });
    }); });
});
test_utils_1.describeMathGPU('Save-load round trips', function () {
    it('Sequential model, Local storage', function (done) {
        var model1 = tfl.sequential();
        model1.add(tfl.layers.dense({ units: 2, inputShape: [2], activation: 'relu' }));
        model1.add(tfl.layers.dense({ units: 1, useBias: false }));
        var path = "testModel" + new Date().getTime() + "_" + Math.random();
        var modelURL = "localstorage://" + path;
        model1.save(modelURL)
            .then(function (saveResult) {
            tfl.loadModel(modelURL)
                .then(function (model2) {
                expect(model2.toJSON(null, false))
                    .toEqual(model1.toJSON(null, false));
                var weights1 = model1.getWeights();
                var weights2 = model2.getWeights();
                expect(weights2.length).toEqual(weights1.length);
                for (var i = 0; i < weights1.length; ++i) {
                    test_utils_1.expectTensorsClose(weights1[i], weights2[i]);
                }
                done();
            })
                .catch(function (err) {
                done.fail(err.stack);
            });
        })
            .catch(function (err) {
            done.fail(err.stack);
        });
    });
    it('Functional model, IndexedDB', function (done) {
        var input = tfl.input({ shape: [2, 2] });
        var layer1 = tfl.layers.flatten().apply(input);
        var layer2 = tfl.layers.dense({ units: 2 }).apply(layer1);
        var model1 = tfl.model({ inputs: input, outputs: layer2 });
        var path = "testModel" + new Date().getTime() + "_" + Math.random();
        var modelURL = "indexeddb://" + path;
        model1.save(modelURL)
            .then(function (saveResult) {
            tfl.loadModel(modelURL)
                .then(function (model2) {
                expect(model2.toJSON(null, false))
                    .toEqual(model1.toJSON(null, false));
                var weights1 = model1.getWeights();
                var weights2 = model2.getWeights();
                expect(weights2.length).toEqual(weights1.length);
                for (var i = 0; i < weights1.length; ++i) {
                    test_utils_1.expectTensorsClose(weights1[i], weights2[i]);
                }
                done();
            })
                .catch(function (err) {
                done.fail(err.stack);
            });
        })
            .catch(function (err) {
            done.fail(err.stack);
        });
    });
    it('Call predict() and fit() after load: conv2d model', function (done) {
        var model = tfl.sequential();
        model.add(tfl.layers.conv2d({
            filters: 8,
            kernelSize: 4,
            inputShape: [28, 28, 1],
            padding: 'same',
            activation: 'relu'
        }));
        model.add(tfl.layers.maxPooling2d({
            poolSize: 2,
            padding: 'same',
        }));
        model.add(tfl.layers.flatten());
        model.add(tfl.layers.dense({ units: 1 }));
        var x = tfjs_core_1.randomNormal([1, 28, 28, 1]);
        var y = model.predict(x);
        var path = "testModel" + new Date().getTime() + "_" + Math.random();
        var url = "indexeddb://" + path;
        model.save(url)
            .then(function (saveResult) {
            tfl.loadModel(url)
                .then(function (modelPrime) {
                var yPrime = modelPrime.predict(x);
                test_utils_1.expectTensorsClose(y, yPrime);
                modelPrime.compile({ optimizer: 'sgd', loss: 'meanSquaredError' });
                var trainExamples = 10;
                modelPrime
                    .fit(tfjs_core_1.randomNormal([trainExamples, 28, 28, 1]), tfjs_core_1.randomNormal([trainExamples]), { epochs: 4 })
                    .then(function (history) {
                    done();
                })
                    .catch(function (err) { return done.fail(err.stack); });
            })
                .catch(function (err) { return done.fail(err.stack); });
        })
            .catch(function (err) { return done.fail(err.stack); });
    });
    it('Call predict() and fit() after load: conv1d model', function (done) {
        var model = tfl.sequential();
        model.add(tfl.layers.conv1d({
            filters: 8,
            kernelSize: 4,
            inputShape: [100, 1],
            padding: 'same',
            activation: 'relu'
        }));
        model.add(tfl.layers.maxPooling1d({
            poolSize: 2,
            padding: 'same',
        }));
        model.add(tfl.layers.flatten());
        model.add(tfl.layers.dense({ units: 1 }));
        var x = tfjs_core_1.randomNormal([1, 100, 1]);
        var y = model.predict(x);
        var path = "testModel" + new Date().getTime() + "_" + Math.random();
        var url = "indexeddb://" + path;
        model.save(url)
            .then(function (saveResult) {
            tfl.loadModel(url)
                .then(function (modelPrime) {
                var yPrime = modelPrime.predict(x);
                test_utils_1.expectTensorsClose(y, yPrime);
                modelPrime.compile({ optimizer: 'sgd', loss: 'meanSquaredError' });
                var trainExamples = 10;
                modelPrime
                    .fit(tfjs_core_1.randomNormal([trainExamples, 100, 1]), tfjs_core_1.randomNormal([trainExamples]), { epochs: 4 })
                    .then(function (history) {
                    done();
                })
                    .catch(function (err) { return done.fail(err.stack); });
            })
                .catch(function (err) { return done.fail(err.stack); });
        })
            .catch(function (err) { return done.fail(err.stack); });
    });
    it('Call predict() and fit() after load: Bidirectional LSTM', function (done) {
        var model = tfl.sequential();
        var lstmUnits = 3;
        var sequenceLength = 4;
        var inputDims = 5;
        model.add(tfl.layers.bidirectional({
            layer: tfl.layers.lstm({ units: lstmUnits }),
            mergeMode: 'concat',
            inputShape: [sequenceLength, inputDims]
        }));
        var x = tfjs_core_1.randomNormal([2, 4, 5]);
        var y = model.predict(x);
        var path = "testModel" + new Date().getTime() + "_" + Math.random();
        var url = "indexeddb://" + path;
        model.save(url)
            .then(function (saveResult) {
            tfl.loadModel(url)
                .then(function (modelPrime) {
                var yPrime = modelPrime.predict(x);
                test_utils_1.expectTensorsClose(y, yPrime);
                modelPrime.compile({ optimizer: 'sgd', loss: 'meanSquaredError' });
                var trainExamples = 2;
                modelPrime
                    .fit(tfjs_core_1.randomNormal([trainExamples, sequenceLength, inputDims]), tfjs_core_1.randomNormal([trainExamples, lstmUnits * 2]), { epochs: 2 })
                    .then(function (history) {
                    done();
                })
                    .catch(function (err) { return done.fail(err.stack); });
            })
                .catch(function (err) { return done.fail(err.stack); });
        })
            .catch(function (err) { return done.fail(err.stack); });
    });
});
//# sourceMappingURL=model_save_test.js.map