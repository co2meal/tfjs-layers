"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tfjs_core_1 = require("@tensorflow/tfjs-core");
var variables_1 = require("../variables");
var variable_utils = require("./variable_utils");
describe('countParamsInWeights', function () {
    it('Zero weights', function () {
        expect(variable_utils.countParamsInWeights([])).toEqual(0);
    });
    it('One float32 weight', function () {
        var weight1 = new variables_1.LayerVariable(tfjs_core_1.zeros([2, 3]));
        expect(variable_utils.countParamsInWeights([weight1])).toEqual(6);
    });
    it('One float32 scalar weight', function () {
        var weight1 = new variables_1.LayerVariable(tfjs_core_1.scalar(42));
        expect(variable_utils.countParamsInWeights([weight1])).toEqual(1);
    });
    it('One int32 weight', function () {
        var weight1 = new variables_1.LayerVariable(tfjs_core_1.zeros([1, 3, 4], 'int32'), 'int32');
        expect(variable_utils.countParamsInWeights([weight1])).toEqual(12);
    });
    it('Two weights, mixed types and shapes', function () {
        var weight1 = new variables_1.LayerVariable(tfjs_core_1.scalar(42));
        var weight2 = new variables_1.LayerVariable(tfjs_core_1.zeros([2, 3]));
        var weight3 = new variables_1.LayerVariable(tfjs_core_1.zeros([1, 3, 4], 'int32'), 'int32');
        expect(variable_utils.countParamsInWeights([
            weight1, weight2, weight3
        ])).toEqual(19);
    });
});
//# sourceMappingURL=variable_utils_test.js.map