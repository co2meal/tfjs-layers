"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tfjs_core_1 = require("@tensorflow/tfjs-core");
var tfl = require("./index");
var initializers_1 = require("./initializers");
var math_utils = require("./utils/math_utils");
var test_utils_1 = require("./utils/test_utils");
test_utils_1.describeMathCPU('Zeros initializer', function () {
    it('1D', function () {
        var init = initializers_1.getInitializer('zeros');
        var weights = init.apply([3], 'float32');
        expect(weights.shape).toEqual([3]);
        expect(weights.dtype).toEqual('float32');
        expect(weights.dataSync()).toEqual(new Float32Array([0, 0, 0]));
    });
    it('1D, upper case', function () {
        var init = initializers_1.getInitializer('Zeros');
        var weights = init.apply([3], 'float32');
        expect(weights.shape).toEqual([3]);
        expect(weights.dtype).toEqual('float32');
        expect(weights.dataSync()).toEqual(new Float32Array([0, 0, 0]));
    });
    it('2D', function () {
        var init = initializers_1.getInitializer('zeros');
        var weights = init.apply([2, 2], 'float32');
        expect(weights.shape).toEqual([2, 2]);
        expect(weights.dtype).toEqual('float32');
        expect(weights.dataSync()).toEqual(new Float32Array([0, 0, 0, 0]));
    });
    it('Does not leak', function () {
        test_utils_1.expectNoLeakedTensors(function () { return initializers_1.getInitializer('zeros').apply([3]); }, 1);
    });
});
test_utils_1.describeMathCPU('Ones initializer', function () {
    it('1D', function () {
        var init = initializers_1.getInitializer('ones');
        var weights = init.apply([3], 'float32');
        expect(weights.shape).toEqual([3]);
        expect(weights.dtype).toEqual('float32');
        expect(weights.dataSync()).toEqual(new Float32Array([1, 1, 1]));
    });
    it('1D, upper case', function () {
        var init = initializers_1.getInitializer('Ones');
        var weights = init.apply([3], 'float32');
        expect(weights.shape).toEqual([3]);
        expect(weights.dtype).toEqual('float32');
        expect(weights.dataSync()).toEqual(new Float32Array([1, 1, 1]));
    });
    it('2D', function () {
        var init = initializers_1.getInitializer('ones');
        var weights = init.apply([2, 2], 'float32');
        expect(weights.shape).toEqual([2, 2]);
        expect(weights.dtype).toEqual('float32');
        expect(weights.dataSync()).toEqual(new Float32Array([1, 1, 1, 1]));
    });
    it('Does not leak', function () {
        test_utils_1.expectNoLeakedTensors(function () { return initializers_1.getInitializer('ones').apply([3]); }, 1);
    });
});
test_utils_1.describeMathCPU('Constant initializer', function () {
    it('1D, from config dict', function () {
        var initializerConfig = { className: 'Constant', config: { value: 5 } };
        var init = initializers_1.getInitializer(initializerConfig);
        var weights = init.apply([3], 'float32');
        expect(weights.shape).toEqual([3]);
        expect(weights.dtype).toEqual('float32');
        expect(weights.dataSync()).toEqual(new Float32Array([5, 5, 5]));
    });
    it('1D, from builder function', function () {
        var init = tfl.initializers.constant({ value: 5 });
        var weights = init.apply([3], 'float32');
        expect(weights.shape).toEqual([3]);
        expect(weights.dtype).toEqual('float32');
        expect(weights.dataSync()).toEqual(new Float32Array([5, 5, 5]));
    });
    it('1D, from builder function: passing a direct value throws error', function () {
        expect(function () { return tfl.initializers.constant(5); })
            .toThrowError(/Expected.*ConstantConfig/);
    });
    it('2D, from config dict', function () {
        var initializerConfig = { className: 'Constant', config: { value: 5 } };
        var init = initializers_1.getInitializer(initializerConfig);
        var weights = init.apply([2, 2], 'float32');
        expect(weights.shape).toEqual([2, 2]);
        expect(weights.dtype).toEqual('float32');
        expect(weights.dataSync()).toEqual(new Float32Array([5, 5, 5, 5]));
    });
    it('Does not leak', function () {
        var initializerConfig = { className: 'Constant', config: { value: 5 } };
        test_utils_1.expectNoLeakedTensors(function () { return initializers_1.getInitializer(initializerConfig).apply([3]); }, 1);
    });
});
test_utils_1.describeMathCPU('Identity initializer', function () {
    it('1D', function () {
        var initializerConfig = { className: 'Identity', config: { gain: 5 } };
        var init = initializers_1.getInitializer(initializerConfig);
        expect(function () { return init.apply([4]); }).toThrowError(/2D square/);
    });
    it('1D, from config', function () {
        var initializerConfig = { className: 'Identity', config: { gain: 5 } };
        var init = initializers_1.getInitializer(initializerConfig);
        expect(function () { return init.apply([4]); }).toThrowError(/2D square/);
    });
    it('2D', function () {
        var initializerConfig = { className: 'Identity', config: { gain: 5 } };
        var init = initializers_1.getInitializer(initializerConfig);
        var weights = init.apply([2, 2], 'float32');
        expect(weights.shape).toEqual([2, 2]);
        expect(weights.dtype).toEqual('float32');
        test_utils_1.expectTensorsClose(weights, tfjs_core_1.tensor2d([5, 0, 0, 5], [2, 2]));
    });
});
test_utils_1.describeMathCPU('RandomUniform initializer', function () {
    var shape = [7, 2];
    it('default', function () {
        var init = initializers_1.getInitializer('randomUniform');
        var weights = init.apply(shape, 'float32');
        expect(weights.shape).toEqual(shape);
        expect(weights.dtype).toEqual('float32');
        test_utils_1.expectTensorsValuesInRange(weights, -0.05, 0.05);
    });
    it('default, upper case', function () {
        var init = initializers_1.getInitializer('RandomUniform');
        var weights = init.apply(shape, 'float32');
        expect(weights.shape).toEqual(shape);
        expect(weights.dtype).toEqual('float32');
        test_utils_1.expectTensorsValuesInRange(weights, -0.05, 0.05);
    });
    it('with configured min max val', function () {
        var initializerConfig = {
            className: 'RandomUniform',
            config: { minval: 17, maxval: 47 }
        };
        var init = initializers_1.getInitializer(initializerConfig);
        var weights = init.apply(shape, 'float32');
        expect(weights.shape).toEqual(shape);
        expect(weights.dtype).toEqual('float32');
        test_utils_1.expectTensorsValuesInRange(weights, 17, 47);
    });
    it('Does not leak', function () {
        test_utils_1.expectNoLeakedTensors(function () { return initializers_1.getInitializer('RandomUniform').apply([3]); }, 1);
    });
});
test_utils_1.describeMathCPU('RandomNormal initializer', function () {
    var shape = [7, 2];
    it('default', function () {
        var init = initializers_1.getInitializer('randomNormal');
        var weights = init.apply(shape, 'float32');
        expect(weights.shape).toEqual(shape);
        expect(weights.dtype).toEqual('float32');
    });
    it('default, upper case', function () {
        var init = initializers_1.getInitializer('RandomNormal');
        var weights = init.apply(shape, 'float32');
        expect(weights.shape).toEqual(shape);
        expect(weights.dtype).toEqual('float32');
    });
    it('with configured min max val', function () {
        var initializerConfig = {
            className: 'RandomNormal',
            config: { mean: 1.0, stddev: 0.001 }
        };
        var init = initializers_1.getInitializer(initializerConfig);
        var weights = init.apply(shape, 'float32');
        expect(weights.shape).toEqual(shape);
        expect(weights.dtype).toEqual('float32');
    });
    it('Does not leak', function () {
        test_utils_1.expectNoLeakedTensors(function () { return initializers_1.getInitializer('RandomNormal').apply([3]); }, 1);
    });
});
test_utils_1.describeMathCPU('HeNormal initializer', function () {
    var shape = [7, 2];
    var stddev = Math.sqrt(2 / shape[0]);
    it('default', function () {
        var init = initializers_1.getInitializer('heNormal');
        var weights = init.apply(shape, 'float32');
        expect(weights.shape).toEqual(shape);
        expect(weights.dtype).toEqual('float32');
        test_utils_1.expectTensorsValuesInRange(weights, -2 * stddev, 2 * stddev);
        expect(init.getClassName()).toEqual(initializers_1.VarianceScaling.className);
    });
    it('default, upper case', function () {
        var init = initializers_1.getInitializer('HeNormal');
        var weights = init.apply(shape, 'float32');
        expect(weights.shape).toEqual(shape);
        expect(weights.dtype).toEqual('float32');
        test_utils_1.expectTensorsValuesInRange(weights, -2 * stddev, 2 * stddev);
    });
    it('Does not leak', function () {
        test_utils_1.expectNoLeakedTensors(function () { return initializers_1.getInitializer('HeNormal').apply([3]); }, 1);
    });
});
test_utils_1.describeMathCPU('LecunNormal initializer', function () {
    var shape = [7, 2];
    var stddev = Math.sqrt(1 / shape[0]);
    it('default', function () {
        var init = initializers_1.getInitializer('leCunNormal');
        var weights = init.apply(shape, 'float32');
        expect(weights.shape).toEqual(shape);
        expect(weights.dtype).toEqual('float32');
        test_utils_1.expectTensorsValuesInRange(weights, -2 * stddev, 2 * stddev);
        expect(init.getClassName()).toEqual(initializers_1.VarianceScaling.className);
    });
    it('default, upper case', function () {
        var init = initializers_1.getInitializer('LeCunNormal');
        var weights = init.apply(shape, 'float32');
        expect(weights.shape).toEqual(shape);
        expect(weights.dtype).toEqual('float32');
        test_utils_1.expectTensorsValuesInRange(weights, -2 * stddev, 2 * stddev);
    });
    it('Does not leak', function () {
        test_utils_1.expectNoLeakedTensors(function () { return initializers_1.getInitializer('LeCunNormal').apply([3]); }, 1);
    });
});
test_utils_1.describeMathCPU('TruncatedNormal initializer', function () {
    var shape = [7, 2];
    it('default', function () {
        var init = initializers_1.getInitializer('truncatedNormal');
        var weights = init.apply(shape, 'float32');
        expect(weights.shape).toEqual(shape);
        expect(weights.dtype).toEqual('float32');
        test_utils_1.expectTensorsValuesInRange(weights, -0.1, 0.1);
    });
    it('default, upper case', function () {
        var init = initializers_1.getInitializer('TruncatedNormal');
        var weights = init.apply(shape, 'float32');
        expect(weights.shape).toEqual(shape);
        expect(weights.dtype).toEqual('float32');
        test_utils_1.expectTensorsValuesInRange(weights, -0.1, 0.1);
    });
    it('with configured min max val', function () {
        var initializerConfig = {
            className: 'TruncatedNormal',
            config: { mean: 1.0, stddev: 0.5 }
        };
        var init = initializers_1.getInitializer(initializerConfig);
        var weights = init.apply(shape, 'float32');
        expect(weights.shape).toEqual(shape);
        expect(weights.dtype).toEqual('float32');
        test_utils_1.expectTensorsValuesInRange(weights, 0.0, 2.0);
    });
    it('Does not leak', function () {
        test_utils_1.expectNoLeakedTensors(function () { return initializers_1.getInitializer('TruncatedNormal').apply([3]); }, 1);
    });
});
test_utils_1.describeMathCPU('Glorot uniform initializer', function () {
    ['glorotUniform', 'GlorotUniform'].forEach(function (initializer) {
        it('1D ' + initializer, function () {
            var init = initializers_1.getInitializer(initializer);
            var weights = init.apply([3], 'float32');
            expect(weights.shape).toEqual([3]);
            expect(weights.dtype).toEqual('float32');
            var scale = 1 / ((Math.sqrt(3) + Math.sqrt(3)) / 2);
            var limit = Math.sqrt(3 * scale);
            expect(math_utils.max(weights.dataSync()))
                .toBeLessThan(limit);
            expect(math_utils.min(weights.dataSync()))
                .toBeGreaterThan(-limit);
            weights = init.apply([30], 'float32');
            expect(weights.shape).toEqual([30]);
            expect(weights.dtype).toEqual('float32');
            scale = 1 / ((Math.sqrt(30) + Math.sqrt(30)) / 2);
            limit = Math.sqrt(3 * scale);
            expect(math_utils.max(weights.dataSync()))
                .toBeLessThan(limit);
            expect(math_utils.min(weights.dataSync()))
                .toBeGreaterThan(-limit);
            expect(init.getClassName()).toEqual(initializers_1.VarianceScaling.className);
        });
        it('2D ' + initializer, function () {
            var init = initializers_1.getInitializer(initializer);
            var weights = init.apply([2, 2], 'float32');
            expect(weights.shape).toEqual([2, 2]);
            expect(weights.dtype).toEqual('float32');
            var scale = 1 / ((Math.sqrt(2) + Math.sqrt(2)) / 2);
            var limit = Math.sqrt(3 * scale);
            expect(math_utils.max(weights.dataSync()))
                .toBeLessThan(limit);
            expect(math_utils.min(weights.dataSync()))
                .toBeGreaterThan(-limit);
            weights = init.apply([20, 20], 'float32');
            expect(weights.shape).toEqual([20, 20]);
            expect(weights.dtype).toEqual('float32');
            scale = 1 / ((Math.sqrt(20) + Math.sqrt(20)) / 2);
            limit = Math.sqrt(3 * scale);
            expect(math_utils.max(weights.dataSync()))
                .toBeLessThan(limit);
            expect(math_utils.min(weights.dataSync()))
                .toBeGreaterThan(-limit);
        });
    });
    it('Does not leak', function () {
        test_utils_1.expectNoLeakedTensors(function () { return initializers_1.getInitializer('GlorotUniform').apply([3]); }, 1);
    });
});
test_utils_1.describeMathCPU('Glorot normal initializer', function () {
    ['glorotNormal', 'GlorotNormal'].forEach(function (initializer) {
        it('1D ' + initializer, function () {
            var init = initializers_1.getInitializer(initializer);
            var NUM_TRIALS = 4;
            var varianceArr1 = [];
            var varianceArr2 = [];
            for (var i = 0; i < NUM_TRIALS; ++i) {
                var weights = init.apply([30], 'float32');
                expect(weights.shape).toEqual([30]);
                expect(weights.dtype).toEqual('float32');
                varianceArr1.push(math_utils.variance(weights.dataSync()));
                weights = init.apply([1200], 'float32');
                expect(weights.shape).toEqual([1200]);
                expect(weights.dtype).toEqual('float32');
                varianceArr2.push(math_utils.variance(weights.dataSync()));
                expect(init.getClassName()).toEqual(initializers_1.VarianceScaling.className);
            }
            var variance1 = math_utils.median(varianceArr1);
            var variance2 = math_utils.median(varianceArr2);
            expect(variance2).toBeLessThan(variance1);
        });
        it('2D ' + initializer, function () {
            var init = initializers_1.getInitializer(initializer);
            var NUM_TRIALS = 4;
            var varianceArr1 = [];
            var varianceArr2 = [];
            for (var i = 0; i < NUM_TRIALS; ++i) {
                var weights = init.apply([5, 6], 'float32');
                expect(weights.shape).toEqual([5, 6]);
                expect(weights.dtype).toEqual('float32');
                varianceArr1.push(math_utils.variance(weights.dataSync()));
                weights = init.apply([30, 50], 'float32');
                expect(weights.shape).toEqual([30, 50]);
                expect(weights.dtype).toEqual('float32');
                varianceArr2.push(math_utils.variance(weights.dataSync()));
            }
            var variance1 = math_utils.median(varianceArr1);
            var variance2 = math_utils.median(varianceArr2);
            expect(variance2).toBeLessThan(variance1);
        });
    });
    it('Does not leak', function () {
        test_utils_1.expectNoLeakedTensors(function () { return initializers_1.getInitializer('GlorotNormal').apply([3]); }, 1);
    });
});
test_utils_1.describeMathCPU('initializers.get', function () {
    it('by string', function () {
        var initializer = initializers_1.getInitializer('glorotNormal');
        var config = initializers_1.serializeInitializer(initializer);
        var nestedConfig = config.config;
        expect(nestedConfig.scale).toEqual(1.0);
        expect(nestedConfig.mode).toEqual('fanAvg');
        expect(nestedConfig.distribution).toEqual('normal');
    });
    it('by existing object', function () {
        var origInit = tfl.initializers.zeros();
        var initializer = initializers_1.getInitializer(origInit);
        expect(initializer).toEqual(origInit);
    });
    it('by config dict', function () {
        var origInit = tfl.initializers.glorotUniform({ seed: 10 });
        var initializer = initializers_1.getInitializer(initializers_1.serializeInitializer(origInit));
        expect(initializers_1.serializeInitializer(initializer))
            .toEqual(initializers_1.serializeInitializer(origInit));
    });
});
describe('Invalid intializer identifier', function () {
    it('Throws exception', function () {
        expect(function () {
            initializers_1.getInitializer('invalid_initializer_id');
        }).toThrowError();
    });
});
describe('checkFanMode', function () {
    it('Valid values', function () {
        var extendedValues = initializers_1.VALID_FAN_MODE_VALUES.concat([undefined, null]);
        for (var _i = 0, extendedValues_1 = extendedValues; _i < extendedValues_1.length; _i++) {
            var validValue = extendedValues_1[_i];
            initializers_1.checkFanMode(validValue);
        }
    });
    it('Invalid values', function () {
        expect(function () { return initializers_1.checkFanMode('foo'); }).toThrowError(/foo/);
        try {
            initializers_1.checkFanMode('bad');
        }
        catch (e) {
            expect(e).toMatch('FanMode');
            for (var _i = 0, VALID_FAN_MODE_VALUES_1 = initializers_1.VALID_FAN_MODE_VALUES; _i < VALID_FAN_MODE_VALUES_1.length; _i++) {
                var validValue = VALID_FAN_MODE_VALUES_1[_i];
                expect(e).toMatch(validValue);
            }
        }
    });
});
describe('checkDistribution', function () {
    it('Valid values', function () {
        var extendedValues = initializers_1.VALID_DISTRIBUTION_VALUES.concat([undefined, null]);
        for (var _i = 0, extendedValues_2 = extendedValues; _i < extendedValues_2.length; _i++) {
            var validValue = extendedValues_2[_i];
            initializers_1.checkDistribution(validValue);
        }
    });
    it('Invalid values', function () {
        expect(function () { return initializers_1.checkDistribution('foo'); }).toThrowError(/foo/);
        try {
            initializers_1.checkDistribution('bad');
        }
        catch (e) {
            expect(e).toMatch('Distribution');
            for (var _i = 0, VALID_DISTRIBUTION_VALUES_1 = initializers_1.VALID_DISTRIBUTION_VALUES; _i < VALID_DISTRIBUTION_VALUES_1.length; _i++) {
                var validValue = VALID_DISTRIBUTION_VALUES_1[_i];
                expect(e).toMatch(validValue);
            }
        }
    });
});
test_utils_1.describeMathCPUAndGPU('Orthogonal Initializer', function () {
    it('2x2', function () {
        var init = initializers_1.getInitializer('Orthogonal');
        var w = init.apply([2, 2], 'float32');
        expect(w.shape).toEqual([2, 2]);
        expect(w.dtype).toEqual('float32');
        test_utils_1.expectTensorsClose(w.transpose().matMul(w), tfjs_core_1.eye(2));
    });
    it('1x1 with gain', function () {
        var init = tfl.initializers.orthogonal({ gain: 3 });
        var w = init.apply([1, 1], 'float32');
        expect(w.shape).toEqual([1, 1]);
        expect(w.dtype).toEqual('float32');
        test_utils_1.expectTensorsClose(w.transpose().matMul(w), tfjs_core_1.tensor2d([[9]], [1, 1]));
    });
    it('4x2', function () {
        var init = initializers_1.getInitializer('Orthogonal');
        var w = init.apply([4, 2], 'float32');
        expect(w.shape).toEqual([4, 2]);
        expect(w.dtype).toEqual('float32');
        test_utils_1.expectTensorsClose(w.transpose().matMul(w), tfjs_core_1.eye(2));
    });
    it('2x4', function () {
        var init = initializers_1.getInitializer('Orthogonal');
        var w = init.apply([2, 4], 'float32');
        expect(w.shape).toEqual([2, 4]);
        expect(w.dtype).toEqual('float32');
        test_utils_1.expectTensorsClose(w.matMul(w.transpose()), tfjs_core_1.eye(2));
    });
    it('64x64', function () {
        spyOn(console, 'warn').and.callFake(function (message) { });
        var n = 64;
        var init = initializers_1.getInitializer('Orthogonal');
        var w = init.apply([n, n], 'float32');
        expect(w.shape).toEqual([n, n]);
        expect(w.dtype).toEqual('float32');
        test_utils_1.expectTensorsClose(w.matMul(w.transpose()), tfjs_core_1.eye(n));
    });
    it('Does not leak', function () {
        var init = initializers_1.getInitializer('Orthogonal');
        test_utils_1.expectNoLeakedTensors(function () { return init.apply([3, 3]); }, 1);
    });
});
//# sourceMappingURL=initializers_test.js.map