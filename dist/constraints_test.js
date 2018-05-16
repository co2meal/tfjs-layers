"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tfjs_core_1 = require("@tensorflow/tfjs-core");
var constraints_1 = require("./constraints");
var tfl = require("./index");
var test_utils_1 = require("./utils/test_utils");
test_utils_1.describeMathCPU('Built-in Constraints', function () {
    var initVals;
    beforeEach(function () {
        initVals = tfjs_core_1.tensor1d(new Float32Array([-1, 2, 0, 4, -5, 6]));
    });
    it('NonNeg', function () {
        var constraint = constraints_1.getConstraint('NonNeg');
        var postConstraint = constraint.apply(initVals);
        test_utils_1.expectTensorsClose(postConstraint, tfjs_core_1.tensor1d(new Float32Array([0, 2, 0, 4, 0, 6])));
        test_utils_1.expectNoLeakedTensors(function () { return constraint.apply(initVals); }, 1);
    });
    it('MaxNorm', function () {
        var constraint = constraints_1.getConstraint('MaxNorm');
        var postConstraint = constraint.apply(initVals);
        test_utils_1.expectTensorsClose(postConstraint, tfjs_core_1.tensor1d(new Float32Array([
            -0.2208630521, 0.4417261043, 0, 0.8834522086,
            -1.104315261, 1.325178313
        ])));
        test_utils_1.expectNoLeakedTensors(function () { return constraint.apply(initVals); }, 1);
    });
    it('UnitNorm', function () {
        var constraint = constraints_1.getConstraint('UnitNorm');
        var postConstraint = constraint.apply(initVals);
        test_utils_1.expectTensorsClose(postConstraint, tfjs_core_1.tensor1d(new Float32Array([
            -0.2208630521 / 2, 0.4417261043 / 2, 0,
            0.8834522086 / 2, -1.104315261 / 2, 1.325178313 / 2
        ])));
        test_utils_1.expectNoLeakedTensors(function () { return constraint.apply(initVals); }, 1);
    });
    it('MinMaxNorm', function () {
        var constraint = constraints_1.getConstraint('MinMaxNorm');
        var postConstraint = constraint.apply(initVals);
        test_utils_1.expectTensorsClose(postConstraint, tfjs_core_1.tensor1d(new Float32Array([
            -0.2208630521 / 2, 0.4417261043 / 2, 0,
            0.8834522086 / 2, -1.104315261 / 2, 1.325178313 / 2
        ])));
        test_utils_1.expectNoLeakedTensors(function () { return constraint.apply(initVals); }, 1);
    });
    it('nonNeg', function () {
        var constraint = constraints_1.getConstraint('nonNeg');
        var postConstraint = constraint.apply(initVals);
        test_utils_1.expectTensorsClose(postConstraint, tfjs_core_1.tensor1d(new Float32Array([0, 2, 0, 4, 0, 6])));
    });
    it('maxNorm', function () {
        var constraint = constraints_1.getConstraint('maxNorm');
        var postConstraint = constraint.apply(initVals);
        test_utils_1.expectTensorsClose(postConstraint, tfjs_core_1.tensor1d(new Float32Array([
            -0.2208630521, 0.4417261043, 0, 0.8834522086,
            -1.104315261, 1.325178313
        ])));
    });
    it('unitNorm', function () {
        var constraint = constraints_1.getConstraint('unitNorm');
        var postConstraint = constraint.apply(initVals);
        test_utils_1.expectTensorsClose(postConstraint, tfjs_core_1.tensor1d(new Float32Array([
            -0.2208630521 / 2, 0.4417261043 / 2, 0,
            0.8834522086 / 2, -1.104315261 / 2, 1.325178313 / 2
        ])));
    });
    it('minMaxNorm', function () {
        var constraint = constraints_1.getConstraint('minMaxNorm');
        var postConstraint = constraint.apply(initVals);
        test_utils_1.expectTensorsClose(postConstraint, tfjs_core_1.tensor1d(new Float32Array([
            -0.2208630521 / 2, 0.4417261043 / 2, 0,
            0.8834522086 / 2, -1.104315261 / 2, 1.325178313 / 2
        ])));
    });
});
test_utils_1.describeMathCPU('constraints.get', function () {
    it('by string', function () {
        var constraint = constraints_1.getConstraint('maxNorm');
        var config = constraints_1.serializeConstraint(constraint);
        var nestedConfig = config.config;
        expect(nestedConfig.maxValue).toEqual(2);
        expect(nestedConfig.axis).toEqual(0);
    });
    it('by string, upper case', function () {
        var constraint = constraints_1.getConstraint('maxNorm');
        var config = constraints_1.serializeConstraint(constraint);
        var nestedConfig = config.config;
        expect(nestedConfig.maxValue).toEqual(2);
        expect(nestedConfig.axis).toEqual(0);
    });
    it('by existing object', function () {
        var origConstraint = tfl.constraints.nonNeg();
        expect(constraints_1.getConstraint(origConstraint)).toEqual(origConstraint);
    });
    it('by config dict', function () {
        var origConstraint = tfl.constraints.minMaxNorm({ minValue: 0, maxValue: 2, rate: 3, axis: 4 });
        var constraint = constraints_1.getConstraint(constraints_1.serializeConstraint(origConstraint));
        expect(constraints_1.serializeConstraint(constraint))
            .toEqual(constraints_1.serializeConstraint(origConstraint));
    });
});
describe('Constraints Serialization', function () {
    it('Built-ins', function () {
        var constraints = [
            'maxNorm', 'nonNeg', 'unitNorm', 'minMaxNorm', 'MaxNorm', 'NonNeg',
            'UnitNorm', 'MinMaxNorm'
        ];
        for (var _i = 0, constraints_2 = constraints; _i < constraints_2.length; _i++) {
            var name_1 = constraints_2[_i];
            var constraint = constraints_1.getConstraint(name_1);
            var config = constraints_1.serializeConstraint(constraint);
            var reconstituted = constraints_1.deserializeConstraint(config);
            expect(reconstituted).toEqual(constraint);
        }
    });
});
//# sourceMappingURL=constraints_test.js.map