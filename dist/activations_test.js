"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tfjs_core_1 = require("@tensorflow/tfjs-core");
var activations_1 = require("./activations");
var test_utils_1 = require("./utils/test_utils");
test_utils_1.describeMathCPUAndGPU('linear activation', function () {
    var initVals = new Float32Array([-1, 2, 0, 4, -5, 6]);
    var expectedVals = new Float32Array([-1, 2, 0, 4, -5, 6]);
    var linear = new activations_1.Linear().apply;
    it('1D', function () {
        var initX = tfjs_core_1.tensor1d(initVals);
        test_utils_1.expectTensorsClose(linear(initX), tfjs_core_1.tensor1d(expectedVals));
    });
    it('2D', function () {
        var initX = tfjs_core_1.tensor2d(initVals, [2, 3]);
        test_utils_1.expectTensorsClose(linear(initX), tfjs_core_1.tensor2d(expectedVals, [2, 3]));
    });
    it('3D', function () {
        var initX = tfjs_core_1.tensor3d(initVals, [1, 2, 3]);
        test_utils_1.expectTensorsClose(linear(initX), tfjs_core_1.tensor3d(expectedVals, [1, 2, 3]));
    });
    it('Does not leak', function () {
        var initX = tfjs_core_1.tensor1d(initVals);
        test_utils_1.expectNoLeakedTensors(function () { return linear(initX); }, 0);
    });
});
test_utils_1.describeMathCPUAndGPU('elu activation', function () {
    var initVals = [-1, 2, 0, 4, -5, 6];
    var expectedVals = initVals.map(function (x) { return x < 0 ? Math.exp(x) - 1 : x; });
    var elu = new activations_1.Elu().apply;
    it('1D', function () {
        var initX = tfjs_core_1.tensor1d(initVals);
        test_utils_1.expectTensorsClose(elu(initX), tfjs_core_1.tensor1d(expectedVals));
    });
    it('2D', function () {
        var initX = tfjs_core_1.tensor2d(initVals, [2, 3]);
        test_utils_1.expectTensorsClose(elu(initX), tfjs_core_1.tensor2d(expectedVals, [2, 3]));
    });
    it('3D', function () {
        var initX = tfjs_core_1.tensor3d(initVals, [1, 2, 3]);
        test_utils_1.expectTensorsClose(elu(initX), tfjs_core_1.tensor3d(expectedVals, [1, 2, 3]));
    });
    it('Does not leak', function () {
        var initX = tfjs_core_1.tensor1d(initVals);
        test_utils_1.expectNoLeakedTensors(function () { return elu(initX); }, 1);
    });
});
test_utils_1.describeMathCPUAndGPU('selu activation', function () {
    var initVals = [-1, 2, 0, 4, -5, 6];
    var alpha = 1.6732632423543772848170429916717;
    var scale = 1.0507009873554804934193349852946;
    var expectedVals = initVals.map(function (x) { return scale * (x < 0 ? (alpha * (Math.exp(x) - 1)) : x); });
    var selu = new activations_1.Selu().apply;
    it('1D', function () {
        var initX = tfjs_core_1.tensor1d(initVals);
        test_utils_1.expectTensorsClose(selu(initX), tfjs_core_1.tensor1d(expectedVals));
    });
    it('2D', function () {
        var initX = tfjs_core_1.tensor2d(initVals, [2, 3]);
        test_utils_1.expectTensorsClose(selu(initX), tfjs_core_1.tensor2d(expectedVals, [2, 3]));
    });
    it('3D', function () {
        var initX = tfjs_core_1.tensor3d(initVals, [1, 2, 3]);
        test_utils_1.expectTensorsClose(selu(initX), tfjs_core_1.tensor3d(expectedVals, [1, 2, 3]));
    });
    it('Does not leak', function () {
        var initX = tfjs_core_1.tensor1d(initVals);
        test_utils_1.expectNoLeakedTensors(function () { return selu(initX); }, 1);
    });
});
test_utils_1.describeMathCPUAndGPU('relu activation', function () {
    var initVals = new Float32Array([-1, 2, 0, 4, -5, 6]);
    var expectedVals = new Float32Array([0, 2, 0, 4, 0, 6]);
    var relu = new activations_1.Relu().apply;
    it('1D', function () {
        var initX = tfjs_core_1.tensor1d(initVals);
        test_utils_1.expectTensorsClose(relu(initX), tfjs_core_1.tensor1d(expectedVals));
    });
    it('2D', function () {
        var initX = tfjs_core_1.tensor2d(initVals, [2, 3]);
        test_utils_1.expectTensorsClose(relu(initX), tfjs_core_1.tensor2d(expectedVals, [2, 3]));
    });
    it('3D', function () {
        var initX = tfjs_core_1.tensor3d(initVals, [1, 2, 3]);
        test_utils_1.expectTensorsClose(relu(initX), tfjs_core_1.tensor3d(expectedVals, [1, 2, 3]));
    });
    it('Does not leak', function () {
        var initX = tfjs_core_1.tensor1d(initVals);
        test_utils_1.expectNoLeakedTensors(function () { return relu(initX); }, 1);
    });
});
test_utils_1.describeMathCPUAndGPU('relu6 activation', function () {
    var initVals = new Float32Array([-10, -5, 0, 1, 5, 15]);
    var expectedVals = new Float32Array([0, 0, 0, 1, 5, 6]);
    var relu6 = new activations_1.Relu6().apply;
    it('1D', function () {
        var initX = tfjs_core_1.tensor1d(initVals);
        test_utils_1.expectTensorsClose(relu6(initX), tfjs_core_1.tensor1d(expectedVals));
    });
    it('2D', function () {
        var initX = tfjs_core_1.tensor2d(initVals, [2, 3]);
        test_utils_1.expectTensorsClose(relu6(initX), tfjs_core_1.tensor2d(expectedVals, [2, 3]));
    });
    it('3D', function () {
        var initX = tfjs_core_1.tensor3d(initVals, [1, 2, 3]);
        test_utils_1.expectTensorsClose(relu6(initX), tfjs_core_1.tensor3d(expectedVals, [1, 2, 3]));
    });
    it('Does not leak', function () {
        var initX = tfjs_core_1.tensor1d(initVals);
        test_utils_1.expectNoLeakedTensors(function () { return relu6(initX); }, 1);
    });
});
test_utils_1.describeMathCPUAndGPU('sigmoid activation', function () {
    var sigmoid = new activations_1.Sigmoid().apply;
    var initVals = [-1, 2, 0, 4, -5, 6];
    it('Scalar', function () {
        test_utils_1.expectTensorsClose(sigmoid(tfjs_core_1.scalar(0)), tfjs_core_1.scalar(0.5));
    });
    it('3D', function () {
        var expectedVals = initVals.map(function (v) { return 1 / (1 + Math.exp(-v)); });
        var initX = tfjs_core_1.tensor3d(initVals, [1, 2, 3]);
        test_utils_1.expectTensorsClose(sigmoid(initX), tfjs_core_1.tensor3d(expectedVals, [1, 2, 3]));
    });
    it('Does not leak', function () {
        var initX = tfjs_core_1.tensor1d(initVals);
        test_utils_1.expectNoLeakedTensors(function () { return sigmoid(initX); }, 1);
    });
});
test_utils_1.describeMathCPUAndGPU('hardSigmoid activation', function () {
    var hardSigmoid = new activations_1.HardSigmoid().apply;
    var initVals = [-1, 2, 0, 4, -5, 6];
    it('Scalar', function () {
        test_utils_1.expectTensorsClose(hardSigmoid(tfjs_core_1.scalar(0)), tfjs_core_1.scalar(0.5));
    });
    it('3D', function () {
        var expectedVals = initVals.map(function (v) {
            var y = 0.2 * v + 0.5;
            if (y > 1) {
                return 1;
            }
            else if (y < 0) {
                return 0;
            }
            else {
                return y;
            }
        });
        var initX = tfjs_core_1.tensor3d(initVals, [1, 2, 3]);
        test_utils_1.expectTensorsClose(hardSigmoid(initX), tfjs_core_1.tensor3d(expectedVals, [1, 2, 3]));
    });
    it('Does not leak', function () {
        var initX = tfjs_core_1.tensor1d(initVals);
        test_utils_1.expectNoLeakedTensors(function () { return hardSigmoid(initX); }, 1);
    });
});
test_utils_1.describeMathCPUAndGPU('softplus activation', function () {
    var softplus = new activations_1.Softplus().apply;
    var initVals = [-1, 2, 0, 4, -5, 6];
    it('Scalar', function () {
        test_utils_1.expectTensorsClose(softplus(tfjs_core_1.scalar(0)), tfjs_core_1.scalar(Math.log(2)));
    });
    it('3D', function () {
        var expectedVals = initVals.map(function (v) { return Math.log(Math.exp(v) + 1); });
        var initX = tfjs_core_1.tensor3d(initVals, [1, 2, 3]);
        test_utils_1.expectTensorsClose(softplus(initX), tfjs_core_1.tensor3d(expectedVals, [1, 2, 3]));
    });
    it('Does not leak', function () {
        var initX = tfjs_core_1.tensor1d(initVals);
        test_utils_1.expectNoLeakedTensors(function () { return softplus(initX); }, 1);
    });
});
test_utils_1.describeMathCPUAndGPU('softsign activation', function () {
    var softsign = new activations_1.Softsign().apply;
    var initVals = [-1, 2, 0, 4, -5, 6];
    it('Scalar', function () {
        test_utils_1.expectTensorsClose(softsign(tfjs_core_1.scalar(0)), tfjs_core_1.scalar(0));
    });
    it('3D', function () {
        var expectedVals = initVals.map(function (v) { return v / (Math.abs(v) + 1); });
        var initX = tfjs_core_1.tensor3d(initVals, [1, 2, 3]);
        test_utils_1.expectTensorsClose(softsign(initX), tfjs_core_1.tensor3d(expectedVals, [1, 2, 3]));
    });
    it('Does not leak', function () {
        var initX = tfjs_core_1.tensor1d(initVals);
        test_utils_1.expectNoLeakedTensors(function () { return softsign(initX); }, 1);
    });
});
test_utils_1.describeMathCPUAndGPU('tanh activation', function () {
    var tanh = new activations_1.Tanh().apply;
    var initVals = [-1, 2, 0, 4, -5, 6];
    var expectedVals = initVals.map(function (x) { return Math.tanh(x); });
    it('1D', function () {
        var initX = tfjs_core_1.tensor1d(initVals);
        test_utils_1.expectTensorsClose(tanh(initX), tfjs_core_1.tensor1d(expectedVals));
    });
    it('2D', function () {
        var initX = tfjs_core_1.tensor2d(initVals, [2, 3]);
        test_utils_1.expectTensorsClose(tanh(initX), tfjs_core_1.tensor2d(expectedVals, [2, 3]));
    });
    it('3D', function () {
        var initX = tfjs_core_1.tensor3d(initVals, [1, 2, 3]);
        test_utils_1.expectTensorsClose(tanh(initX), tfjs_core_1.tensor3d(expectedVals, [1, 2, 3]));
    });
    it('Does not leak', function () {
        var initX = tfjs_core_1.tensor1d(initVals);
        test_utils_1.expectNoLeakedTensors(function () { return tanh(initX); }, 1);
    });
});
test_utils_1.describeMathCPUAndGPU('softmax activation', function () {
    var softmax = new activations_1.Softmax().apply;
    it('1D', function () {
        var initVals = new Float32Array([0, 1, 3, 9]);
        var expectedVals = new Float32Array([0.000, 0.000, 0.002, 0.997]);
        var initX = tfjs_core_1.tensor1d(initVals);
        test_utils_1.expectTensorsClose(softmax(initX), tfjs_core_1.tensor1d(expectedVals));
    });
    it('1D all equal', function () {
        var initVals = new Float32Array([-1, -1, -1, -1]);
        var expectedVals = new Float32Array([0.25, 0.25, 0.25, 0.25]);
        var initX = tfjs_core_1.tensor1d(initVals);
        test_utils_1.expectTensorsClose(softmax(initX), tfjs_core_1.tensor1d(expectedVals));
    });
    it('2D', function () {
        var initVals = new Float32Array([0, 1, 3, 9, 0, 1, 3, 9]);
        var expectedVals = new Float32Array([0.000, 0.000, 0.002, 0.997, 0.000, 0.000, 0.002, 0.997]);
        var initX = tfjs_core_1.tensor2d(initVals, [2, 4]);
        test_utils_1.expectTensorsClose(softmax(initX), tfjs_core_1.tensor2d(expectedVals, [2, 4]));
    });
    it('3D', function () {
        var initVals = new Float32Array([0, 1, 3, 9, 0, 1, 3, 9]);
        var expectedVals = new Float32Array([0.000, 0.000, 0.002, 0.997, 0.000, 0.000, 0.002, 0.997]);
        var initX = tfjs_core_1.tensor3d(initVals, [1, 2, 4]);
        test_utils_1.expectTensorsClose(softmax(initX), tfjs_core_1.tensor3d(expectedVals, [1, 2, 4]));
    });
    it('Does not leak', function () {
        var initVals = new Float32Array([0, 1, 3, 9]);
        var initX = tfjs_core_1.tensor1d(initVals);
        test_utils_1.expectNoLeakedTensors(function () { return softmax(initX); }, 1);
    });
});
//# sourceMappingURL=activations_test.js.map