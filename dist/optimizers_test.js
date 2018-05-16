"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tfjs_core_1 = require("@tensorflow/tfjs-core");
var optimizers_1 = require("./optimizers");
var test_utils_1 = require("./utils/test_utils");
test_utils_1.describeMathCPU('getOptimizer', function () {
    it("can instantiate SGD", function () {
        var optimizer = optimizers_1.getOptimizer('SGD');
        expect(optimizer instanceof tfjs_core_1.SGDOptimizer).toBe(true);
    });
    it("can instantiate sgd", function () {
        var optimizer = optimizers_1.getOptimizer('sgd');
        expect(optimizer instanceof tfjs_core_1.SGDOptimizer).toBe(true);
    });
    it("can instantiate Adam", function () {
        var optimizer = optimizers_1.getOptimizer('Adam');
        expect(optimizer instanceof tfjs_core_1.AdamOptimizer).toBe(true);
    });
    it("can instantiate adam", function () {
        var optimizer = optimizers_1.getOptimizer('adam');
        expect(optimizer instanceof tfjs_core_1.AdamOptimizer).toBe(true);
    });
    it("can instantiate RMSProp", function () {
        var optimizer = optimizers_1.getOptimizer('RMSProp');
        expect(optimizer instanceof tfjs_core_1.RMSPropOptimizer).toBe(true);
    });
    it("can instantiate rmsprop", function () {
        var optimizer = optimizers_1.getOptimizer('rmsprop');
        expect(optimizer instanceof tfjs_core_1.RMSPropOptimizer).toBe(true);
    });
    it("can instantiate Adagrad", function () {
        var optimizer = optimizers_1.getOptimizer('Adagrad');
        expect(optimizer instanceof tfjs_core_1.AdagradOptimizer).toBe(true);
    });
    it("can instantiate adagrad", function () {
        var optimizer = optimizers_1.getOptimizer('adagrad');
        expect(optimizer instanceof tfjs_core_1.AdagradOptimizer).toBe(true);
    });
    it("can instantiate Adadelta", function () {
        var optimizer = optimizers_1.getOptimizer('Adadelta');
        expect(optimizer instanceof tfjs_core_1.AdadeltaOptimizer).toBe(true);
    });
    it("can instantiate adadelta", function () {
        var optimizer = optimizers_1.getOptimizer('adadelta');
        expect(optimizer instanceof tfjs_core_1.AdadeltaOptimizer).toBe(true);
    });
    it("can instantiate Adamax", function () {
        var optimizer = optimizers_1.getOptimizer('Adamax');
        expect(optimizer instanceof tfjs_core_1.AdamaxOptimizer).toBe(true);
    });
    it("can instantiate adamax", function () {
        var optimizer = optimizers_1.getOptimizer('adamax');
        expect(optimizer instanceof tfjs_core_1.AdamaxOptimizer).toBe(true);
    });
    it('throws for non-existent optimizer', function () {
        expect(function () { return optimizers_1.getOptimizer('not an optimizer'); })
            .toThrowError(/Unknown Optimizer/);
    });
});
//# sourceMappingURL=optimizers_test.js.map