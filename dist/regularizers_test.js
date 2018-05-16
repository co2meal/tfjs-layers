"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tfjs_core_1 = require("@tensorflow/tfjs-core");
var tfl = require("./index");
var regularizers_1 = require("./regularizers");
var test_utils_1 = require("./utils/test_utils");
test_utils_1.describeMathCPU('Built-in Regularizers', function () {
    it('l1_l2', function () {
        var x = tfjs_core_1.tensor1d([1, -2, 3, -4]);
        var regularizer = tfl.regularizers.l1l2();
        var score = regularizer.apply(x);
        test_utils_1.expectTensorsClose(score, tfjs_core_1.scalar(0.01 * (1 + 2 + 3 + 4) + 0.01 * (1 + 4 + 9 + 16)));
    });
    it('l1', function () {
        var x = tfjs_core_1.tensor1d([1, -2, 3, -4]);
        var regularizer = tfl.regularizers.l1();
        var score = regularizer.apply(x);
        test_utils_1.expectTensorsClose(score, tfjs_core_1.scalar(0.01 * (1 + 2 + 3 + 4)));
    });
    it('l2', function () {
        var x = tfjs_core_1.tensor1d([1, -2, 3, -4]);
        var regularizer = tfl.regularizers.l2();
        var score = regularizer.apply(x);
        test_utils_1.expectTensorsClose(score, tfjs_core_1.scalar(0.01 * (1 + 4 + 9 + 16)));
    });
    it('l1_l2 non default', function () {
        var x = tfjs_core_1.tensor1d([1, -2, 3, -4]);
        var regularizer = tfl.regularizers.l1l2({ l1: 1, l2: 2 });
        var score = regularizer.apply(x);
        test_utils_1.expectTensorsClose(score, tfjs_core_1.scalar(1 * (1 + 2 + 3 + 4) + 2 * (1 + 4 + 9 + 16)));
    });
});
test_utils_1.describeMathCPU('regularizers.get', function () {
    var x;
    beforeEach(function () {
        x = tfjs_core_1.tensor1d([1, -2, 3, -4]);
    });
    it('by string - lower camel', function () {
        var regularizer = regularizers_1.getRegularizer('l1l2');
        test_utils_1.expectTensorsClose(regularizer.apply(x), tfl.regularizers.l1l2().apply(x));
    });
    it('by string - upper camel', function () {
        var regularizer = regularizers_1.getRegularizer('L1L2');
        test_utils_1.expectTensorsClose(regularizer.apply(x), tfl.regularizers.l1l2().apply(x));
    });
    it('by existing object', function () {
        var origReg = tfl.regularizers.l1l2({ l1: 1, l2: 2 });
        var regularizer = regularizers_1.getRegularizer(origReg);
        expect(regularizer).toEqual(origReg);
    });
    it('by config dict', function () {
        var origReg = tfl.regularizers.l1l2({ l1: 1, l2: 2 });
        var regularizer = regularizers_1.getRegularizer(regularizers_1.serializeRegularizer(origReg));
        test_utils_1.expectTensorsClose(regularizer.apply(x), origReg.apply(x));
    });
});
test_utils_1.describeMathCPU('Regularizer Serialization', function () {
    it('Built-ins', function () {
        var regularizer = tfl.regularizers.l1l2({ l1: 1, l2: 2 });
        var config = regularizers_1.serializeRegularizer(regularizer);
        var reconstituted = regularizers_1.deserializeRegularizer(config);
        var roundTripConfig = regularizers_1.serializeRegularizer(reconstituted);
        expect(roundTripConfig.className).toEqual('L1L2');
        var nestedConfig = roundTripConfig.config;
        expect(nestedConfig.l1).toEqual(1);
        expect(nestedConfig.l2).toEqual(2);
    });
});
//# sourceMappingURL=regularizers_test.js.map