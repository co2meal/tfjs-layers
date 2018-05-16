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
var K = require("../backend/tfjs_backend");
var callbacks_1 = require("../callbacks");
var tfl = require("../index");
var generic_utils_1 = require("../utils/generic_utils");
var test_utils_1 = require("../utils/test_utils");
var training_1 = require("./training");
test_utils_1.describeMathCPU('isDataTensor', function () {
    it('Positive case', function () {
        expect(training_1.isDataTensor(tfjs_core_1.scalar(3.14))).toEqual(true);
    });
    it('Negative cases', function () {
        expect(training_1.isDataTensor([tfjs_core_1.scalar(3.14), tfjs_core_1.scalar(-3.14)])).toEqual(false);
        expect(training_1.isDataTensor({ 'Pie': tfjs_core_1.scalar(3.14) })).toEqual(false);
        expect(training_1.isDataTensor({})).toEqual(false);
    });
});
test_utils_1.describeMathCPU('isDataArray', function () {
    it('Positive case', function () {
        expect(training_1.isDataArray([tfjs_core_1.scalar(3.14), tfjs_core_1.scalar(-3.14)])).toEqual(true);
        expect(training_1.isDataArray([])).toEqual(true);
    });
    it('Negative cases', function () {
        expect(training_1.isDataArray(tfjs_core_1.scalar(3.14))).toEqual(false);
        expect(training_1.isDataArray({ 'Pie': tfjs_core_1.scalar(3.14) })).toEqual(false);
        expect(training_1.isDataArray({})).toEqual(false);
    });
});
test_utils_1.describeMathCPU('isDataDict', function () {
    it('Positive case', function () {
        expect(training_1.isDataDict({ 'Pie': tfjs_core_1.scalar(3.14) })).toEqual(true);
        expect(training_1.isDataDict({})).toEqual(true);
    });
    it('Negative cases', function () {
        expect(training_1.isDataDict(tfjs_core_1.scalar(3.14))).toEqual(false);
        expect(training_1.isDataDict([tfjs_core_1.scalar(3.14), tfjs_core_1.scalar(-3.14)])).toEqual(false);
        expect(training_1.isDataDict([])).toEqual(false);
    });
});
test_utils_1.describeMathCPU('standardizeInputData', function () {
    it('Singleton Tensor, Array of one name', function () {
        var outputs = training_1.standardizeInputData(tfjs_core_1.scalar(42), ['Foo']);
        expect(outputs.length).toEqual(1);
        test_utils_1.expectTensorsClose(outputs[0], tfjs_core_1.scalar(42));
    });
    it('Array of one Tensor, Array of one name', function () {
        var outputs = training_1.standardizeInputData([tfjs_core_1.scalar(42)], ['Foo']);
        expect(outputs.length).toEqual(1);
        test_utils_1.expectTensorsClose(outputs[0], tfjs_core_1.scalar(42));
    });
    it('Array of two Tensors, Array of two names', function () {
        var outputs = training_1.standardizeInputData([tfjs_core_1.scalar(42), tfjs_core_1.scalar(21)], ['Foo', 'Bar']);
        expect(outputs.length).toEqual(2);
        test_utils_1.expectTensorsClose(outputs[0], tfjs_core_1.scalar(42));
        test_utils_1.expectTensorsClose(outputs[1], tfjs_core_1.scalar(21));
    });
    it('Dict of two Tensors, Array of two names', function () {
        var outputs = training_1.standardizeInputData({ 'Foo': tfjs_core_1.scalar(42), 'Bar': tfjs_core_1.scalar(21) }, ['Foo', 'Bar']);
        expect(outputs.length).toEqual(2);
        test_utils_1.expectTensorsClose(outputs[0], tfjs_core_1.scalar(42));
        test_utils_1.expectTensorsClose(outputs[1], tfjs_core_1.scalar(21));
    });
    it('Unexpected data leads to exception: singleton Tensor', function () {
        expect(function () { return training_1.standardizeInputData(tfjs_core_1.scalar(42), []); })
            .toThrowError(/expected no data/);
    });
    it('Unexpected data leads to exception: Array of two Tensors', function () {
        expect(function () { return training_1.standardizeInputData([tfjs_core_1.scalar(42), tfjs_core_1.scalar(21)], []); })
            .toThrowError(/expected no data/);
    });
    it('Unexpected data leads to exception: Dict', function () {
        expect(function () { return training_1.standardizeInputData({ 'Pie': tfjs_core_1.scalar(42) }, []); })
            .toThrowError(/expected no data/);
    });
    it('Length mismatch: 1 singleton Scalar vs two names', function () {
        expect(function () { return training_1.standardizeInputData(tfjs_core_1.scalar(42), ['Foo', 'Bar']); })
            .toThrowError(/expects 2 Tensor.* but only received one/);
    });
    it('Length mismatch: Array of 2 Scalars vs one name', function () {
        expect(function () { return training_1.standardizeInputData([tfjs_core_1.scalar(42), tfjs_core_1.scalar(-42)], ['Foo']); })
            .toThrowError(/Expected to see 1 Tensor/);
    });
    it('Length mismatch: Dict of 1 Scalar vs 2 names', function () {
        expect(function () { return training_1.standardizeInputData({ 'Foo': tfjs_core_1.scalar(42) }, ['Foo', 'Bar']); })
            .toThrowError(/No data provided for \"Bar\"/);
    });
});
test_utils_1.describeMathCPU('checkArrayLengths', function () {
    it('Batch mismatch in inputs', function () {
        var inputs = [tfjs_core_1.zeros([2, 1]), tfjs_core_1.zeros([3, 1])];
        var targets = [tfjs_core_1.zeros([2, 1]), tfjs_core_1.zeros([2, 1])];
        expect(function () { return training_1.checkArrayLengths(inputs, targets); })
            .toThrowError(/All input .* should have the same number of samples/);
    });
    it('Batch mismatch in targets', function () {
        var inputs = [tfjs_core_1.zeros([2, 1]), tfjs_core_1.zeros([2, 1])];
        var targets = [tfjs_core_1.zeros([2, 1]), tfjs_core_1.zeros([3, 1])];
        expect(function () { return training_1.checkArrayLengths(inputs, targets); })
            .toThrowError(/All target .* should have the same number of samples/);
    });
    it('Batch mismatch between inputs and targets', function () {
        var inputs = [tfjs_core_1.zeros([2, 1]), tfjs_core_1.zeros([2, 1])];
        var targets = [tfjs_core_1.zeros([3, 1]), tfjs_core_1.zeros([3, 1])];
        expect(function () { return training_1.checkArrayLengths(inputs, targets); })
            .toThrowError(/Input Tensors should have the same number of samples as target/);
    });
});
test_utils_1.describeMathCPUAndGPU('sliceArraysByIndices', function () {
    it('Single 2D', function () {
        var x = tfjs_core_1.tensor2d([[1, 2], [3, 4], [5, 6]], [3, 2]);
        var y = training_1.sliceArraysByIndices(x, tfjs_core_1.tensor1d([0, 2]));
        test_utils_1.expectTensorsClose(y, tfjs_core_1.tensor2d([[1, 2], [5, 6]], [2, 2]));
    });
    it('Array of two 2Ds', function () {
        var xs = [
            tfjs_core_1.tensor2d([[1, 2], [3, 4], [5, 6]], [3, 2]),
            tfjs_core_1.tensor2d([[10, 20], [30, 40], [50, 60]], [3, 2])
        ];
        var ys = training_1.sliceArraysByIndices(xs, tfjs_core_1.tensor1d([0, 2]));
        expect(ys.length).toEqual(2);
        test_utils_1.expectTensorsClose(ys[0], tfjs_core_1.tensor2d([[1, 2], [5, 6]], [2, 2]));
        test_utils_1.expectTensorsClose(ys[1], tfjs_core_1.tensor2d([[10, 20], [50, 60]], [2, 2]));
    });
    it('Array of two 3Ds', function () {
        var xs = [
            tfjs_core_1.tensor3d([[[1]], [[2]], [[3]]], [3, 1, 1]),
            tfjs_core_1.tensor3d([[[10]], [[20]], [[30]]], [3, 1, 1]),
        ];
        var ys = training_1.sliceArraysByIndices(xs, tfjs_core_1.tensor1d([0, 2]));
        expect(ys.length).toEqual(2);
        test_utils_1.expectTensorsClose(ys[0], tfjs_core_1.tensor3d([[[1]], [[3]]], [2, 1, 1]));
        test_utils_1.expectTensorsClose(ys[1], tfjs_core_1.tensor3d([[[10]], [[30]]], [2, 1, 1]));
    });
    it('null array input', function () {
        expect(training_1.sliceArraysByIndices(null, tfjs_core_1.tensor1d([0, 2]))).toBeNull();
    });
    it('casts indices automatically', function () {
        var x = tfjs_core_1.tensor2d([[1, 2], [3, 4], [5, 6]], [3, 2]);
        var y = training_1.sliceArraysByIndices(x, tfjs_core_1.tensor1d([0.1, 2.0], 'float32'));
        test_utils_1.expectTensorsClose(y, tfjs_core_1.tensor2d([[1, 2], [5, 6]], [2, 2]));
    });
});
describe('makeBatches', function () {
    it('divisible', function () {
        expect(training_1.makeBatches(6, 3)).toEqual([[0, 3], [3, 6]]);
    });
    it('indivisible', function () {
        expect(training_1.makeBatches(7, 3)).toEqual([[0, 3], [3, 6], [6, 7]]);
        expect(training_1.makeBatches(2, 4)).toEqual([[0, 2]]);
    });
    it('empty size', function () {
        expect(training_1.makeBatches(0, 4)).toEqual([]);
    });
});
test_utils_1.describeMathCPUAndGPU('Model.predict', function () {
    it('1 input, 1 output', function () {
        var inputTensor = tfl.layers.input({ shape: [3, 4], name: 'inputLayer1', dtype: 'float32' });
        var layer = tfl.layers.reshape({ targetShape: [2, 6] });
        var output = layer.apply(inputTensor);
        var model = new tfl.Model({ inputs: [inputTensor], outputs: [output], name: 'model1x1' });
        var xs = tfjs_core_1.ones([10, 3, 4]);
        var ys = model.predict(xs, { batchSize: 4 });
        test_utils_1.expectTensorsClose(ys, tfjs_core_1.ones([10, 2, 6]));
    });
    it('1 input, 1 output, tensor as input argument', function () {
        var inputTensor = tfl.layers.input({ shape: [3, 4], name: 'inputLayer1', dtype: 'float32' });
        var layer = tfl.layers.reshape({ targetShape: [2, 6] });
        var output = layer.apply(inputTensor);
        var model = new tfl.Model({ inputs: [inputTensor], outputs: [output], name: 'model1x1' });
        var xs = tfjs_core_1.ones([10, 3, 4]);
        var ys = model.predict(xs);
        test_utils_1.expectTensorsClose(ys, tfjs_core_1.ones([10, 2, 6]));
    });
    it('1 input as Array, 1 output', function () {
        var inputTensor = tfl.layers.input({ shape: [3, 4], name: 'inputLayer1', dtype: 'float32' });
        var layer = tfl.layers.reshape({ targetShape: [2, 6] });
        var output = layer.apply(inputTensor);
        var model = new tfl.Model({ inputs: [inputTensor], outputs: [output], name: 'model1x1' });
        var xs = tfjs_core_1.ones([10, 3, 4]);
        var ys = model.predict([xs], { batchSize: 4 });
        test_utils_1.expectTensorsClose(ys, tfjs_core_1.ones([10, 2, 6]));
    });
    it('1 input, 2 outputs', function () {
        var inputTensor = tfl.layers.input({ shape: [3, 4], name: 'inputLayer2', dtype: 'float32' });
        var layer1 = tfl.layers.reshape({ targetShape: [2, 6] });
        var layer2 = tfl.layers.flatten();
        var output1 = layer1.apply(inputTensor);
        var output2 = layer2.apply(output1);
        var model = new tfl.Model({ inputs: [inputTensor], outputs: [output1, output2], name: 'model1x2' });
        var xs = tfjs_core_1.ones([10, 3, 4]);
        var ys = model.predict(xs, { batchSize: 4 });
        expect(ys.length).toEqual(2);
        test_utils_1.expectTensorsClose(ys[0], tfjs_core_1.ones([10, 2, 6]));
        test_utils_1.expectTensorsClose(ys[1], tfjs_core_1.ones([10, 12]));
    });
    it('2 inputs, 2 outputs', function () {
        var inputTensor1 = tfl.layers.input({ shape: [3, 4], name: 'inputLayer3', dtype: 'float32' });
        var inputTensor2 = tfl.layers.input({ shape: [3, 3], name: 'inputLayer4', dtype: 'float32' });
        var layer1 = tfl.layers.reshape({ targetShape: [2, 6] });
        var layer2 = tfl.layers.flatten();
        var output1 = layer1.apply(inputTensor1);
        var output2 = layer2.apply(inputTensor2);
        var model = new tfl.Model({
            inputs: [inputTensor1, inputTensor2],
            outputs: [output1, output2],
            name: 'model2x2'
        });
        var xs1 = tfjs_core_1.ones([10, 3, 4]);
        var xs2 = tfjs_core_1.ones([10, 3, 3]);
        var ys = model.predict([xs1, xs2], { batchSize: 4 });
        expect(ys.length).toEqual(2);
        test_utils_1.expectTensorsClose(ys[0], tfjs_core_1.ones([10, 2, 6]));
        test_utils_1.expectTensorsClose(ys[1], tfjs_core_1.ones([10, 9]));
    });
    it('Incorrect number of inputs leads to exception: 1 vs 2', function () {
        var inputTensor = tfl.layers.input({ shape: [3, 4], name: 'inputLayer_inc_1', dtype: 'float32' });
        var layer = tfl.layers.reshape({ targetShape: [2, 6] });
        var output = layer.apply(inputTensor);
        var model = new tfl.Model({ inputs: [inputTensor], outputs: [output], name: 'model_inc_1x1' });
        var xs1 = tfjs_core_1.ones([10, 3, 4]);
        expect(function () { return model.predict([
            xs1, xs1
        ]); }).toThrowError(/.*Expected.*1 Tensor.*got 2 Tensor.*/);
    });
    it('Incorrect number of inputs leads to exception: 2 vs 3', function () {
        var inputTensor1 = tfl.layers.input({ shape: [3, 4], name: 'inputLayer_inc_3', dtype: 'float32' });
        var inputTensor2 = tfl.layers.input({ shape: [3, 3], name: 'inputLayer_inc_4', dtype: 'float32' });
        var layer1 = tfl.layers.reshape({ targetShape: [2, 6] });
        var layer2 = tfl.layers.flatten();
        var output1 = layer1.apply(inputTensor1);
        var output2 = layer2.apply(inputTensor2);
        var model = new tfl.Model({
            inputs: [inputTensor1, inputTensor2],
            outputs: [output1, output2],
            name: 'model_inc_2x2'
        });
        var xs1 = tfjs_core_1.ones([10, 3, 4]);
        expect(function () { return model.predict([
            xs1, xs1, xs1
        ]); }).toThrowError(/.*Expected.*2 Tensor.*got 3 Tensor.*/);
    });
    it('Incorrect input shape leads to exception', function () {
        var inputTensor = tfl.layers.input({ shape: [3, 4], name: 'inputLayer_inc_1', dtype: 'float32' });
        var layer = tfl.layers.reshape({ targetShape: [2, 6] });
        var output = layer.apply(inputTensor);
        var model = new tfl.Model({ inputs: [inputTensor], outputs: [output], name: 'model_inc_1x1' });
        var xs1 = tfjs_core_1.ones([2, 4, 3]);
        expect(function () { return model.predict(xs1); })
            .toThrowError(/.*expected.* shape \[null,3,4\].*but got.*\[2,4,3\]/);
    });
});
test_utils_1.describeMathCPUAndGPU('Model.fit', function () {
    var inputSize = 4;
    var inputSize1 = 3;
    var inputSize2 = 4;
    var numSamples = 5;
    var inputTensor = tfl.layers.input({ shape: [inputSize], name: 'inputLayer1', dtype: 'float32' });
    var inputTensor1 = tfl.layers.input({ shape: [inputSize1], name: 'inputLayer1of2', dtype: 'float32' });
    var inputTensor2 = tfl.layers.input({ shape: [inputSize2], name: 'inputLayer2of2', dtype: 'float32' });
    var model;
    var inputs;
    var targets;
    var twoOutputModel;
    var inputs1;
    var inputs2;
    var targets1;
    var targets2;
    function createDenseModelAndData(useBias, kernelRegularizer, biasRegularizer) {
        if (useBias === void 0) { useBias = false; }
        var layer = tfl.layers.dense({ units: 1, useBias: useBias, kernelInitializer: 'ones', kernelRegularizer: kernelRegularizer });
        var output = layer.apply(inputTensor);
        model = new tfl.Model({ inputs: [inputTensor], outputs: [output] });
        inputs = tfjs_core_1.ones([numSamples, inputSize]);
        targets = tfjs_core_1.ones([numSamples, 1]);
    }
    function createDenseCategoricalModelAndData(useBias) {
        if (useBias === void 0) { useBias = false; }
        var layer = tfl.layers.dense({ units: 2, useBias: useBias, kernelInitializer: 'ones' });
        var output = layer.apply(inputTensor);
        model = new tfl.Model({ inputs: [inputTensor], outputs: [output] });
        inputs = tfjs_core_1.ones([numSamples, inputSize]);
        targets = K.oneHot(tfjs_core_1.ones([numSamples]), 2);
    }
    function createTwoLayerDenseModelAndData(useBias) {
        if (useBias === void 0) { useBias = false; }
        var layer1 = tfl.layers.dense({ units: 10, useBias: useBias, kernelInitializer: 'ones' });
        var layer2 = tfl.layers.dense({ units: 1, useBias: useBias, kernelInitializer: 'ones' });
        var output = layer2.apply(layer1.apply(inputTensor));
        model = new tfl.Model({ inputs: [inputTensor], outputs: [output] });
        inputs = tfjs_core_1.ones([numSamples, inputSize]);
        targets = tfjs_core_1.ones([numSamples, 1]);
        return [layer1, layer2];
    }
    function createDenseModelWithTwoOutputsAndData() {
        var layer1 = tfl.layers.dense({ units: 1, useBias: false, kernelInitializer: 'ones' });
        var layer2 = tfl.layers.dense({ units: 1, useBias: false, kernelInitializer: 'ones' });
        var output1 = layer1.apply(inputTensor1);
        var output2 = layer2.apply(inputTensor2);
        twoOutputModel = new tfl.Model({ inputs: [inputTensor1, inputTensor2], outputs: [output1, output2] });
        inputs1 = tfjs_core_1.ones([numSamples, inputSize1]);
        inputs2 = tfjs_core_1.ones([numSamples, inputSize2]);
        targets1 = tfjs_core_1.ones([numSamples, 1]);
        targets2 = tfjs_core_1.ones([numSamples, 1]);
    }
    it('1 input, 1 output, dense, 1 weight, string optimizer, 1 batch', function (done) { return __awaiter(_this, void 0, void 0, function () {
        return __generator(this, function (_a) {
            createDenseModelAndData();
            model.compile({ optimizer: 'SGD', loss: 'meanSquaredError' });
            model.fit(inputs, targets, { batchSize: numSamples, epochs: 1 })
                .then(function (history) {
                expect(history.epoch).toEqual([0]);
                var newWeightsValue = model.trainableWeights[0].read();
                var lr = 0.01;
                var expectedValueArray = generic_utils_1.pyListRepeat([1.0 - (inputSize - 1) * 2 * lr], inputSize);
                test_utils_1.expectTensorsClose(newWeightsValue, tfjs_core_1.tensor2d(expectedValueArray, [inputSize, 1]));
                done();
            })
                .catch(function (err) {
                done.fail(err.stack);
            });
            return [2];
        });
    }); });
    it('training with custom loss', function (done) { return __awaiter(_this, void 0, void 0, function () {
        var absDiffLoss;
        return __generator(this, function (_a) {
            createDenseModelAndData();
            absDiffLoss = function (x, y) { return tfjs_core_1.mean(tfjs_core_1.abs(x.sub(y))); };
            model.compile({ optimizer: 'SGD', loss: absDiffLoss });
            model
                .fit(inputs, targets, { batchSize: numSamples, epochs: 2, validationSplit: 0.2 })
                .then(function (history) {
                tfjs_core_1.test_util.expectArraysClose(history.history['loss'], [3, 2.96]);
                tfjs_core_1.test_util.expectArraysClose(history.history['val_loss'], [2.96, 2.92]);
                done();
            })
                .catch(function (err) {
                done.fail(err.stack);
            });
            return [2];
        });
    }); });
    it('Using only x and y input arguments', function (done) { return __awaiter(_this, void 0, void 0, function () {
        return __generator(this, function (_a) {
            createDenseModelAndData();
            model.compile({ optimizer: 'SGD', loss: 'meanSquaredError' });
            model.fit(inputs, targets, { epochs: 100 })
                .then(function (history) {
                expect(history.epoch.length).toEqual(100);
                for (var i = 0; i < 100; ++i) {
                    expect(history.epoch[i]).toEqual(i);
                }
                done();
            })
                .catch(function (err) {
                done.fail(err.stack);
            });
            return [2];
        });
    }); });
    it('Default Model.fit epochs is 1', function (done) { return __awaiter(_this, void 0, void 0, function () {
        return __generator(this, function (_a) {
            createDenseModelAndData();
            model.compile({ optimizer: 'SGD', loss: 'meanSquaredError' });
            model.fit(inputs, targets)
                .then(function (history) {
                expect(history.epoch.length).toEqual(1);
                expect(history.epoch[0]).toEqual(0);
                done();
            })
                .catch(function (err) {
                done.fail(err.stack);
            });
            return [2];
        });
    }); });
    it('1 input, 1 output, dense, 1 weight, string optimizer, 2 epochs', function (done) { return __awaiter(_this, void 0, void 0, function () {
        return __generator(this, function (_a) {
            createDenseModelAndData();
            model.compile({ optimizer: 'SGD', loss: 'meanSquaredError' });
            model.fit(inputs, targets, { batchSize: numSamples, epochs: 2 })
                .then(function (history) {
                expect(history.epoch).toEqual([0, 1]);
                done();
            });
            return [2];
        });
    }); });
    it('Training with Dropout layer', function (done) { return __awaiter(_this, void 0, void 0, function () {
        var inputSize, batchSize, input, dense1, dropout, dense2, output, model, x, y;
        return __generator(this, function (_a) {
            inputSize = 2;
            batchSize = 4;
            input = tfl.layers.input({ shape: [inputSize] });
            dense1 = tfl.layers.dense({ units: 2, kernelInitializer: 'ones', useBias: false });
            dropout = tfl.layers.dropout({ rate: 0.5 });
            dense2 = tfl.layers.dense({ units: 1, kernelInitializer: 'ones', useBias: false });
            output = dense2.apply(dropout.apply(dense1.apply(input)));
            model = new tfl.Model({ inputs: input, outputs: output });
            model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' });
            x = tfjs_core_1.ones([batchSize, inputSize]);
            y = tfjs_core_1.ones([batchSize, 1]);
            model.fit(x, y, { batchSize: batchSize, epochs: 1 }).then(function (history) {
                done();
            });
            return [2];
        });
    }); });
    var validationSplits = [0.2, 0.01];
    var _loop_1 = function (validationSplit) {
        var testTitle = '1 input, 1 output, dense, 1 weight, string optimizer, 2 epochs, ' +
            ("validationSplit=" + validationSplit);
        it(testTitle, function (done) { return __awaiter(_this, void 0, void 0, function () {
            return __generator(this, function (_a) {
                createDenseModelAndData();
                model.compile({ optimizer: 'SGD', loss: 'meanSquaredError' });
                model
                    .fit(inputs, targets, { batchSize: numSamples, epochs: 2, validationSplit: validationSplit })
                    .then(function (history) {
                    expect(history.epoch).toEqual([0, 1]);
                    var losses = history.history['loss'];
                    expect(losses.length).toEqual(2);
                    var valLosses = history.history['val_loss'];
                    expect(valLosses.length).toEqual(2);
                    test_utils_1.expectTensorsClose(losses, [9, 7.617599964141846]);
                    test_utils_1.expectTensorsClose(valLosses, [7.617599964141846, 6.447536945343018]);
                    done();
                });
                return [2];
            });
        }); });
    };
    for (var _i = 0, validationSplits_1 = validationSplits; _i < validationSplits_1.length; _i++) {
        var validationSplit = validationSplits_1[_i];
        _loop_1(validationSplit);
    }
    it('1 input, 1 output, dense, 1 weight, string optimizer, 2 epochs, ' +
        'use validationData', function (done) { return __awaiter(_this, void 0, void 0, function () {
        return __generator(this, function (_a) {
            createDenseModelAndData();
            model.compile({ optimizer: 'SGD', loss: 'meanSquaredError' });
            model
                .fit(inputs, targets, {
                batchSize: numSamples,
                epochs: 2,
                validationData: [tfjs_core_1.zeros(inputs.shape), targets]
            })
                .then(function (history) {
                expect(history.epoch).toEqual([0, 1]);
                var losses = history.history['loss'];
                expect(losses.length).toEqual(2);
                var valLosses = history.history['val_loss'];
                expect(valLosses.length).toEqual(2);
                test_utils_1.expectTensorsClose(losses, [9, 7.617599964141846]);
                done();
            });
            return [2];
        });
    }); });
    it('1 input, 1 output, dense, 1 weight, string optimizer, 2 epochs, ' +
        'validationSplit = 0.2, with additional metric', function (done) { return __awaiter(_this, void 0, void 0, function () {
        return __generator(this, function (_a) {
            createDenseModelAndData();
            model.compile({ optimizer: 'SGD', loss: 'meanSquaredError', metrics: ['accuracy'] });
            expect(model.metricsNames).toEqual(['loss', 'acc']);
            model
                .fit(inputs, targets, {
                batchSize: numSamples,
                epochs: 2,
                validationSplit: 0.2,
            })
                .then(function (history) {
                expect(history.epoch).toEqual([0, 1]);
                var losses = history.history['loss'];
                expect(losses.length).toEqual(2);
                var valLosses = history.history['val_loss'];
                expect(valLosses.length).toEqual(2);
                test_utils_1.expectTensorsClose(losses, [9, 7.617599964141846]);
                test_utils_1.expectTensorsClose(valLosses, [7.617599964141846, 6.447536945343018]);
                done();
            });
            return [2];
        });
    }); });
    it('Return sequences; Fit with metric', function (done) { return __awaiter(_this, void 0, void 0, function () {
        var sequenceLength, inputSize, dataSize, validationSplit, batchSize, outputSize, simpleRNN, timeDistributed, input, output, model, history;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    sequenceLength = 3;
                    inputSize = 4;
                    dataSize = 16;
                    validationSplit = 0.5;
                    batchSize = 3;
                    outputSize = 2;
                    simpleRNN = tfl.layers.simpleRNN({
                        units: outputSize,
                        kernelInitializer: 'ones',
                        recurrentInitializer: 'ones',
                        useBias: false,
                        returnSequences: true,
                    });
                    timeDistributed = tfl.layers.timeDistributed({
                        layer: tfl.layers.dense({ units: outputSize, kernelInitializer: 'ones', useBias: false })
                    });
                    input = tfl.layers.input({ shape: [sequenceLength, inputSize] });
                    output = timeDistributed.apply(simpleRNN.apply(input));
                    model = new tfl.Model({ inputs: input, outputs: output });
                    model.compile({
                        optimizer: 'sgd',
                        loss: 'categoricalCrossentropy',
                        metrics: ['accuracy'],
                    });
                    return [4, model.fit(tfjs_core_1.ones([dataSize, sequenceLength, inputSize]), tfjs_core_1.ones([dataSize, sequenceLength, outputSize]), {
                            batchSize: batchSize,
                            epochs: 1,
                            validationSplit: validationSplit,
                        })];
                case 1:
                    history = _a.sent();
                    test_utils_1.expectTensorsClose(history.history['loss'], [1.3862943649291992]);
                    test_utils_1.expectTensorsClose(history.history['val_loss'], [1.3862943649291992]);
                    test_utils_1.expectTensorsClose(history.history['acc'], [1.0]);
                    test_utils_1.expectTensorsClose(history.history['val_acc'], [1.0]);
                    done();
                    return [2];
            }
        });
    }); });
    var metricsToTest = [['acc'], ['accuracy']];
    var _loop_2 = function (metrics) {
        var testTitle = "categoricalCrossentropy model, validationSplit = 0.2, " +
            ("" + JSON.stringify(metrics));
        it(testTitle, function (done) { return __awaiter(_this, void 0, void 0, function () {
            return __generator(this, function (_a) {
                createDenseCategoricalModelAndData();
                model.compile({ optimizer: 'SGD', loss: 'categoricalCrossentropy', metrics: metrics });
                if (generic_utils_1.stringsEqual(metrics, ['acc']) ||
                    generic_utils_1.stringsEqual(metrics, ['accuracy'])) {
                    expect(model.metricsNames).toEqual(['loss', 'acc']);
                }
                else if (generic_utils_1.stringsEqual(metrics, ['acc', 'accuracy'])) {
                    expect(model.metricsNames).toEqual(['loss', 'acc', 'acc']);
                }
                model
                    .fit(inputs, targets, { batchSize: numSamples, epochs: 2, validationSplit: 0.2 })
                    .then(function (history) {
                    var losses = history.history['loss'];
                    test_utils_1.expectTensorsClose(losses, [0.6931471824645996, 0.6918979287147522]);
                    var valLosses = history.history['val_loss'];
                    test_utils_1.expectTensorsClose(valLosses, [0.6918979287147522, 0.6906517744064331]);
                    var acc = history.history['acc'];
                    test_utils_1.expectTensorsClose(acc, [0, 1]);
                    var valAcc = history.history['val_acc'];
                    test_utils_1.expectTensorsClose(valAcc, [1, 1]);
                    done();
                });
                return [2];
            });
        }); });
    };
    for (var _a = 0, metricsToTest_1 = metricsToTest; _a < metricsToTest_1.length; _a++) {
        var metrics = metricsToTest_1[_a];
        _loop_2(metrics);
    }
    it('Two layers, freeze one layer', function (done) { return __awaiter(_this, void 0, void 0, function () {
        var layers, layer1, layer2, optimizer, history, losses, valLosses;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    layers = createTwoLayerDenseModelAndData();
                    layer1 = layers[0];
                    layer2 = layers[1];
                    optimizer = new tfjs_core_1.SGDOptimizer(1e-2);
                    model.compile({ optimizer: optimizer, loss: 'meanSquaredError' });
                    return [4, model.fit(inputs, targets, { batchSize: numSamples, epochs: 2, validationSplit: 0.2 })];
                case 1:
                    history = _a.sent();
                    losses = history.history['loss'];
                    test_utils_1.expectTensorsClose(losses, [1521.0, 386.35842895507812]);
                    valLosses = history.history['val_loss'];
                    test_utils_1.expectTensorsClose(valLosses, [386.35848999023438, 1808.7342529296875]);
                    test_utils_1.expectTensorsClose(layer1.getWeights()[0], K.scalarTimesArray(tfjs_core_1.scalar(-0.61341441), tfjs_core_1.ones([4, 10])));
                    test_utils_1.expectTensorsClose(layer2.getWeights()[0], K.scalarTimesArray(tfjs_core_1.scalar(-1.77405429), tfjs_core_1.ones([10, 1])));
                    layer1.trainable = false;
                    model.compile({ optimizer: optimizer, loss: 'meanSquaredError' });
                    return [4, model.fit(inputs, targets, { batchSize: numSamples, epochs: 2, validationSplit: 0.2 })];
                case 2:
                    history = _a.sent();
                    losses = history.history['loss'];
                    test_utils_1.expectTensorsClose(losses, [1808.7342529296875, 75.336509704589844]);
                    valLosses = history.history['val_loss'];
                    test_utils_1.expectTensorsClose(valLosses, [75.336524963378906, 3.1378798484802246]);
                    test_utils_1.expectTensorsClose(layer1.getWeights()[0], K.scalarTimesArray(tfjs_core_1.scalar(-0.61341441), tfjs_core_1.ones([4, 10])));
                    test_utils_1.expectTensorsClose(layer2.getWeights()[0], K.scalarTimesArray(tfjs_core_1.scalar(-0.11295), tfjs_core_1.ones([10, 1])));
                    done();
                    return [2];
            }
        });
    }); });
    it('Unknown metric', function () {
        createDenseCategoricalModelAndData();
        expect(function () { return model.compile({
            optimizer: 'SGD',
            loss: 'categoricalCrossentropy',
            metrics: ['foo']
        }); }).toThrowError(/Unknown metric foo/);
    });
    it('1 input, 1 output, dense, 2 weights, string optimizer, 1 batch', function (done) { return __awaiter(_this, void 0, void 0, function () {
        return __generator(this, function (_a) {
            createDenseModelAndData(true);
            model.compile({ optimizer: 'SGD', loss: 'meanSquaredError' });
            model.fit(inputs, targets, { batchSize: numSamples, epochs: 1 })
                .then(function (history) {
                expect(history.epoch).toEqual([0]);
                expect(model.trainableWeights.length).toEqual(2);
                var lr = 0.01;
                var newKernelValue = model.trainableWeights[0].read();
                var expectedKernelArray = generic_utils_1.pyListRepeat([1.0 - (inputSize - 1) * 2 * lr], inputSize);
                test_utils_1.expectTensorsClose(newKernelValue, tfjs_core_1.tensor2d(expectedKernelArray, [inputSize, 1]));
                var newBiasValue = model.trainableWeights[1].read();
                var expectedBiasArray = [0.0 - (inputSize - 1) * 2 * lr];
                test_utils_1.expectTensorsClose(newBiasValue, tfjs_core_1.tensor1d(expectedBiasArray));
                done();
            });
            return [2];
        });
    }); });
    it('1 input, 1 output, dense, 1 weight, optimizer object, 1 batch', function (done) { return __awaiter(_this, void 0, void 0, function () {
        var lr;
        return __generator(this, function (_a) {
            createDenseModelAndData();
            lr = 0.025;
            model.compile({ optimizer: new tfjs_core_1.SGDOptimizer(lr), loss: 'meanSquaredError' });
            model.fit(inputs, targets, { batchSize: numSamples, epochs: 1 })
                .then(function (history) {
                expect(history.epoch).toEqual([0]);
                var newWeightsValue = model.trainableWeights[0].read();
                var expectedValueArray = generic_utils_1.pyListRepeat([1.0 - (inputSize - 1) * 2 * lr], inputSize);
                test_utils_1.expectTensorsClose(newWeightsValue, tfjs_core_1.tensor2d(expectedValueArray, [inputSize, 1]));
                done();
            });
            return [2];
        });
    }); });
    it('2 inputs, 2 outputs, dense, optimizer object, 1 batch', function (done) { return __awaiter(_this, void 0, void 0, function () {
        var lr, trainableWeights, newWeightsValue1, newWeightsValue2;
        return __generator(this, function (_a) {
            createDenseModelWithTwoOutputsAndData();
            lr = 0.01;
            twoOutputModel.compile({
                optimizer: new tfjs_core_1.SGDOptimizer(lr),
                loss: ['meanSquaredError', 'meanSquaredError']
            });
            trainableWeights = twoOutputModel.trainableWeights;
            newWeightsValue1 = trainableWeights[0].read();
            newWeightsValue2 = trainableWeights[1].read();
            twoOutputModel
                .fit([inputs1, inputs2], [targets1, targets2], { batchSize: numSamples, epochs: 1 })
                .then(function (history) {
                expect(history.epoch).toEqual([0]);
                expect(twoOutputModel.trainableWeights.length).toEqual(2);
                newWeightsValue1 = twoOutputModel.trainableWeights[0].read();
                newWeightsValue2 = twoOutputModel.trainableWeights[1].read();
                var expectedValueArray1 = generic_utils_1.pyListRepeat([1.0 - (inputSize1 - 1) * 2 * lr], inputSize1);
                test_utils_1.expectTensorsClose(newWeightsValue1, tfjs_core_1.tensor2d(expectedValueArray1, [inputSize1, 1]));
                var expectedValueArray2 = generic_utils_1.pyListRepeat([1.0 - (inputSize2 - 1) * 2 * lr], inputSize2);
                test_utils_1.expectTensorsClose(newWeightsValue2, tfjs_core_1.tensor2d(expectedValueArray2, [inputSize2, 1]));
                done();
            });
            return [2];
        });
    }); });
    var isCustomCallbackConfig = [false, true];
    var isCustomCallbackArray = [false, true];
    var _loop_3 = function (isConfig) {
        var _loop_4 = function (isArray) {
            var testTitle = "Fit with custom callback object: isConfig=" + isConfig + ", isArray=" + isArray;
            it(testTitle, function (done) { return __awaiter(_this, void 0, void 0, function () {
                var _this = this;
                var trainBeginLogs, trainEndLogs, epochBeginEpochs, epochEndEpochs, batchBeginBatches, batchEndBatches, batchEndLosses, epochEndLosses, customCallbackConfig, customCallback, i;
                return __generator(this, function (_a) {
                    switch (_a.label) {
                        case 0:
                            createDenseModelAndData();
                            trainBeginLogs = [];
                            trainEndLogs = [];
                            epochBeginEpochs = [];
                            epochEndEpochs = [];
                            batchBeginBatches = [];
                            batchEndBatches = [];
                            batchEndLosses = [];
                            epochEndLosses = [];
                            customCallbackConfig = {
                                onTrainBegin: function (logs) { return __awaiter(_this, void 0, void 0, function () {
                                    return __generator(this, function (_a) {
                                        trainBeginLogs.push(logs);
                                        return [2];
                                    });
                                }); },
                                onTrainEnd: function (logs) { return __awaiter(_this, void 0, void 0, function () {
                                    return __generator(this, function (_a) {
                                        trainEndLogs.push(logs);
                                        return [2];
                                    });
                                }); },
                                onEpochBegin: function (epoch, logs) { return __awaiter(_this, void 0, void 0, function () {
                                    return __generator(this, function (_a) {
                                        epochBeginEpochs.push(epoch);
                                        return [2];
                                    });
                                }); },
                                onEpochEnd: function (epoch, logs) { return __awaiter(_this, void 0, void 0, function () {
                                    return __generator(this, function (_a) {
                                        epochEndEpochs.push(epoch);
                                        epochEndLosses.push(logs['loss']);
                                        return [2];
                                    });
                                }); },
                                onBatchBegin: function (batch, logs) { return __awaiter(_this, void 0, void 0, function () {
                                    return __generator(this, function (_a) {
                                        batchBeginBatches.push(batch);
                                        return [2];
                                    });
                                }); },
                                onBatchEnd: function (batch, logs) { return __awaiter(_this, void 0, void 0, function () {
                                    return __generator(this, function (_a) {
                                        batchEndBatches.push(batch);
                                        batchEndLosses.push(logs['loss']);
                                        return [2];
                                    });
                                }); }
                            };
                            customCallback = isConfig ?
                                customCallbackConfig :
                                new callbacks_1.CustomCallback(customCallbackConfig);
                            model.compile({ optimizer: 'sgd', loss: 'meanSquaredError' });
                            return [4, model.fit(inputs, targets, {
                                    batchSize: 2,
                                    epochs: 2,
                                    callbacks: isArray ? [customCallback] : customCallback,
                                })];
                        case 1:
                            _a.sent();
                            expect(trainBeginLogs.length).toEqual(1);
                            expect(trainEndLogs.length).toEqual(1);
                            expect(epochBeginEpochs).toEqual([0, 1]);
                            expect(epochEndEpochs).toEqual([0, 1]);
                            expect(batchBeginBatches).toEqual([0, 1, 2, 0, 1, 2]);
                            expect(batchEndBatches).toEqual([0, 1, 2, 0, 1, 2]);
                            expect(batchEndLosses.length).toEqual(6);
                            for (i = 1; i < batchEndLosses.length; ++i) {
                                expect(batchEndLosses[i]).toBeLessThan(batchEndLosses[i - 1]);
                            }
                            expect(epochEndLosses.length).toEqual(2);
                            expect(epochEndLosses[1]).toBeLessThan(epochEndLosses[0]);
                            done();
                            return [2];
                    }
                });
            }); });
        };
        for (var _i = 0, isCustomCallbackArray_1 = isCustomCallbackArray; _i < isCustomCallbackArray_1.length; _i++) {
            var isArray = isCustomCallbackArray_1[_i];
            _loop_4(isArray);
        }
    };
    for (var _b = 0, isCustomCallbackConfig_1 = isCustomCallbackConfig; _b < isCustomCallbackConfig_1.length; _b++) {
        var isConfig = isCustomCallbackConfig_1[_b];
        _loop_3(isConfig);
    }
    it('Using custom regularizer', function (done) { return __awaiter(_this, void 0, void 0, function () {
        return __generator(this, function (_a) {
            createDenseModelAndData(false, tfl.regularizers.l1l2({ l1: 1, l2: 1 }));
            model.compile({ optimizer: 'SGD', loss: 'meanSquaredError' });
            model.fit(inputs, targets, { batchSize: numSamples, epochs: 2 })
                .then(function (history) {
                test_utils_1.expectTensorsClose(model.layers[1].getWeights()[0], tfjs_core_1.tensor2d([0.829, 0.829, 0.829, 0.829], [4, 1]));
                expect(history.history.loss.length).toEqual(2);
                expect(history.history.loss[0]).toBeCloseTo(17);
                expect(history.history.loss[1]).toBeCloseTo(13.92);
                done();
            })
                .catch(function (err) {
                done.fail(err.stack);
            });
            return [2];
        });
    }); });
    it('Using string regularizer', function (done) { return __awaiter(_this, void 0, void 0, function () {
        return __generator(this, function (_a) {
            createDenseModelAndData(false, 'l1l2');
            model.compile({ optimizer: 'SGD', loss: 'meanSquaredError' });
            model.fit(inputs, targets, { batchSize: numSamples, epochs: 2 })
                .then(function (history) {
                test_utils_1.expectTensorsClose(model.layers[1].getWeights()[0], tfjs_core_1.tensor2d([0.884, 0.884, 0.884, 0.884], [4, 1]));
                expect(history.history.loss.length).toEqual(2);
                expect(history.history.loss[0]).toBeCloseTo(9.08);
                expect(history.history.loss[1]).toBeCloseTo(7.68);
                done();
            })
                .catch(function (err) {
                done.fail(err.stack);
            });
            return [2];
        });
    }); });
    it('Invalid dict loss: nonexistent output name', function () {
        createDenseModelAndData();
        expect(function () { return model.compile({
            optimizer: 'SGD',
            loss: { 'Foo': 'meanSquaredError' }
        }); }).toThrowError(/Unknown entry in loss dictionary:.*Foo.*/);
    });
    it('Invalid Array loss: missing loss for an output', function () {
        createDenseModelWithTwoOutputsAndData();
        expect(function () { return twoOutputModel.compile({
            optimizer: 'SGD',
            loss: ['meanSquaredError']
        }); }).toThrowError(/should have one entry per model output.*has 2 output/);
    });
    it('Calling fit without compile leads to error', function () {
        createDenseModelAndData(true);
        var fitPromise = model.fit(inputs, targets, { batchSize: numSamples, epochs: 1 });
        fitPromise.catch(function (error) {
            expect(error.message).toContain('You must compile a model before');
        });
    });
});
test_utils_1.describeMathCPUAndGPU('Model.fit with training-sensitive layers', function () {
    it('Correct training arg during fit/evaluate/predict', function (done) { return __awaiter(_this, void 0, void 0, function () {
        var inputTensor, layer1, layer2, dropoutLayerTrainingFlags, recordDropoutTrainingArgHook, output, model, xs, ys, err_1;
        return __generator(this, function (_a) {
            switch (_a.label) {
                case 0:
                    inputTensor = tfl.layers.input({ shape: [1], name: 'inputLayer1', dtype: 'float32' });
                    layer1 = tfl.layers.dense({ units: 1 });
                    layer2 = tfl.layers.dropout({ rate: 0.5 });
                    dropoutLayerTrainingFlags = [];
                    recordDropoutTrainingArgHook = function (inputs, kwargs) {
                        dropoutLayerTrainingFlags.push(kwargs.training);
                    };
                    layer2.setCallHook(recordDropoutTrainingArgHook);
                    output = layer2.apply(layer1.apply(inputTensor));
                    model = new tfl.Model({ inputs: [inputTensor], outputs: [output] });
                    model.compile({ optimizer: 'sgd', loss: 'meanSquaredError' });
                    xs = tfjs_core_1.ones([4, 1]);
                    ys = tfjs_core_1.ones([4, 1]);
                    _a.label = 1;
                case 1:
                    _a.trys.push([1, 3, , 4]);
                    return [4, model.fit(xs, ys, { epochs: 2, batchSize: 4 })];
                case 2:
                    _a.sent();
                    return [3, 4];
                case 3:
                    err_1 = _a.sent();
                    done.fail(err_1.stack);
                    return [3, 4];
                case 4:
                    expect(dropoutLayerTrainingFlags).toEqual([true, true]);
                    model.evaluate(xs, ys, { batchSize: 4 });
                    expect(dropoutLayerTrainingFlags).toEqual([true, true, undefined]);
                    model.predict(xs, { batchSize: 4 });
                    expect(dropoutLayerTrainingFlags).toEqual([
                        true, true, undefined, undefined
                    ]);
                    done();
                    return [2];
            }
        });
    }); });
});
test_utils_1.describeMathCPUAndGPU('Model.evaluate', function () {
    var numExamples = 8;
    var inputSize = 2;
    var outputSize = 1;
    var model;
    var x;
    var y;
    function prepModel() {
        var input = tfl.layers.input({ shape: [inputSize] });
        var dense = tfl.layers.dense({ units: outputSize, kernelInitializer: 'ones', useBias: false });
        var output = dense.apply(input);
        model = new tfl.Model({ inputs: input, outputs: output });
    }
    function prepData() {
        x = tfjs_core_1.ones([numExamples, inputSize]);
        y = tfjs_core_1.ones([numExamples, outputSize]);
    }
    it('Calling evaluate before compile leads to error', function () {
        prepModel();
        prepData();
        expect(function () { return model.evaluate(x, y); })
            .toThrowError(/must compile a model before/);
    });
    var metricsValues = [null, ['mse']];
    var batchSizes = [null, 4, 16];
    var _loop_5 = function (metrics) {
        var _loop_6 = function (batchSize) {
            var testTitle = "metrics=" + JSON.stringify(metrics) + ", batchSize=" + batchSize;
            it(testTitle, function () {
                prepModel();
                prepData();
                model.compile({ optimizer: 'sgd', loss: 'meanSquaredError', metrics: metrics });
                var losses = model.evaluate(x, y, { batchSize: batchSize });
                if (metrics == null) {
                    test_utils_1.expectTensorsClose(losses, tfjs_core_1.scalar(1));
                }
                else {
                    var lossesArray = losses;
                    expect(lossesArray.length).toEqual(2);
                    test_utils_1.expectTensorsClose(lossesArray[0], tfjs_core_1.scalar(1));
                    test_utils_1.expectTensorsClose(lossesArray[1], tfjs_core_1.scalar(1));
                }
            });
        };
        for (var _i = 0, batchSizes_1 = batchSizes; _i < batchSizes_1.length; _i++) {
            var batchSize = batchSizes_1[_i];
            _loop_6(batchSize);
        }
    };
    for (var _i = 0, metricsValues_1 = metricsValues; _i < metricsValues_1.length; _i++) {
        var metrics = metricsValues_1[_i];
        _loop_5(metrics);
    }
});
test_utils_1.describeMathCPUAndGPU('Load weights', function () {
    it('Simple functional model', function () {
        var inputTensor = tfl.layers.input({ shape: [3], name: 'inputLayer', dtype: 'float32' });
        var denseLayer = tfl.layers.dense({ units: 2, useBias: true, name: 'denseLayer' });
        var output = denseLayer.apply(inputTensor);
        var model = new tfl.Model({
            inputs: [inputTensor],
            outputs: [output],
            name: 'modelWithWeightsToLoad',
        });
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
        model.loadWeights(weightsJSON);
        test_utils_1.expectTensorsClose(model.apply(tfjs_core_1.tensor2d([[1, 1, 1]], [1, 3])), tfjs_core_1.tensor2d([[0.8, 1.0]], [1, 2]));
    });
});
//# sourceMappingURL=training_test.js.map