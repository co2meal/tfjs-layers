"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tfc = require("@tensorflow/tfjs-core");
var tfjs_core_1 = require("@tensorflow/tfjs-core");
var K = require("./backend/tfjs_backend");
var common_1 = require("./common");
var tfl = require("./index");
var test_utils_1 = require("./utils/test_utils");
var V = require("./variables");
test_utils_1.describeMathCPU('Variable', function () {
    it('Variable constructor: no explicit name', function () {
        var v1 = new V.LayerVariable(tfjs_core_1.zeros([2]));
        expect(v1.name.indexOf('Variable')).toEqual(0);
        expect(v1.dtype).toEqual('float32');
        expect(v1.shape).toEqual([2]);
        expect(v1.trainable).toEqual(true);
        expect(v1.read().dataSync()).toEqual(new Float32Array([0, 0]));
        var v2 = new V.LayerVariable(tfjs_core_1.zeros([2, 2]));
        expect(v2.name.indexOf('Variable')).toEqual(0);
        expect(v2.dtype).toEqual('float32');
        expect(v2.shape).toEqual([2, 2]);
        expect(v2.read().dataSync()).toEqual(new Float32Array([0, 0, 0, 0]));
        expect(v2.name === v1.name).toBe(false);
    });
    it('Variable constructor: explicit name', function () {
        var v1 = new V.LayerVariable(tfjs_core_1.zeros([]), undefined, 'foo');
        expect(v1.name.indexOf('foo')).toEqual(0);
        expect(v1.dtype).toEqual('float32');
        expect(v1.shape).toEqual([]);
        expect(v1.trainable).toEqual(true);
        expect(v1.read().dataSync()).toEqual(new Float32Array([0]));
        var v2 = new V.LayerVariable(tfjs_core_1.zeros([2, 2, 1]));
        expect(v1.name.indexOf('foo')).toEqual(0);
        expect(v2.dtype).toEqual('float32');
        expect(v2.shape).toEqual([2, 2, 1]);
        expect(v2.read().dataSync()).toEqual(new Float32Array([0, 0, 0, 0]));
        expect(v2.name.length).toBeGreaterThan(0);
        expect(v2.name === v1.name).toBe(false);
    });
    it('Variable constructor: explicit name with name scope', function () {
        var v1;
        common_1.nameScope('barScope', function () {
            common_1.nameScope('bazScope', function () {
                v1 = new V.LayerVariable(tfjs_core_1.scalar(0), undefined, 'foo');
            });
        });
        expect(v1.name.indexOf('barScope/bazScope/foo')).toEqual(0);
        expect(v1.dtype).toEqual('float32');
        expect(v1.shape).toEqual([]);
        expect(v1.trainable).toEqual(true);
        expect(v1.read().dataSync()).toEqual(new Float32Array([0]));
    });
    it('Variable trainable property', function () {
        var v1 = new V.LayerVariable(tfjs_core_1.zeros([]), null, 'foo', false);
        expect(v1.trainable).toEqual(false);
    });
    it('Variable works if name is null or undefined', function () {
        expect((new V.LayerVariable(tfjs_core_1.zeros([]), null)).name.indexOf('Variable'))
            .toEqual(0);
        expect((new V.LayerVariable(tfjs_core_1.zeros([]), undefined)).name.indexOf('Variable'))
            .toEqual(0);
    });
    it('int32 dtype', function () {
        expect(new V.LayerVariable(tfjs_core_1.zeros([]), 'int32').dtype).toEqual('int32');
    });
    it('bool dtype', function () {
        expect(new V.LayerVariable(tfjs_core_1.zeros([]), 'bool').dtype).toEqual('bool');
    });
    it('Read value', function () {
        var v1 = new V.LayerVariable(tfjs_core_1.scalar(10), null, 'foo');
        expect(v1.read().dataSync()).toEqual(new Float32Array([10]));
    });
    it('Update value: Compatible shape', function () {
        var v = new V.LayerVariable(tfjs_core_1.tensor1d([10, -10]), null, 'bar');
        expect(v.name.indexOf('bar')).toEqual(0);
        expect(v.shape).toEqual([2]);
        expect(v.read().dataSync()).toEqual(new Float32Array([10, -10]));
        v.write(tfjs_core_1.tensor1d([10, 50]));
        expect(v.name.indexOf('bar')).toEqual(0);
        expect(v.shape).toEqual([2]);
        expect(v.read().dataSync()).toEqual(new Float32Array([10, 50]));
    });
    it('Update value: w/ constraint', function () {
        var v = new V.LayerVariable(tfjs_core_1.tensor1d([10, -10]), null, 'bar', true, tfl.constraints.nonNeg());
        v.write(tfjs_core_1.tensor1d([-10, 10]));
        expect(v.read().dataSync()).toEqual(new Float32Array([0, 10]));
    });
    it('Update value: Incompatible shape', function () {
        var v = new V.LayerVariable(tfjs_core_1.zeros([2, 2]), null, 'qux');
        expect(function () {
            v.write(tfjs_core_1.zeros([4]));
        }).toThrowError();
    });
    it('Generates unique ID', function () {
        var v1 = new V.LayerVariable(tfjs_core_1.scalar(1), null, 'foo');
        var v2 = new V.LayerVariable(tfjs_core_1.scalar(1), null, 'foo');
        expect(v1.id).not.toEqual(v2.id);
    });
    it('Generates unique IDs for Tensors and Variables', function () {
        var v1 = tfjs_core_1.scalar(1);
        var v2 = new V.LayerVariable(tfjs_core_1.scalar(1), null, 'foo');
        expect(v1.id).not.toEqual(v2.id);
    });
    it('dispose() frees memory', function () {
        var v = new V.LayerVariable(tfjs_core_1.tensor1d([10, -10]), null, 'gralk');
        var numTensors0 = tfc.memory().numTensors;
        v.dispose();
        expect(tfc.memory().numTensors).toEqual(numTensors0 - 1);
    });
    it('Disposing LayersVariable twices leads to Error', function () {
        var v = new V.LayerVariable(tfjs_core_1.tensor1d([10, -10]), null, 'gralk');
        v.dispose();
        expect(function () { return v.dispose(); }).toThrowError(/LayersVariable .*gralk.* disposed/);
    });
    it('read() after dispose() leads to Error', function () {
        var v = new V.LayerVariable(tfjs_core_1.tensor1d([10, -10]), null, 'gralk');
        v.dispose();
        expect(function () { return v.read(); }).toThrowError(/LayersVariable .*gralk.* disposed/);
    });
    it('write() after dispose() leads to Error', function () {
        var v = new V.LayerVariable(tfjs_core_1.tensor1d([10, -10]), null, 'gralk');
        v.dispose();
        expect(function () { return v.write(tfjs_core_1.tensor1d([20, -20])); })
            .toThrowError(/LayersVariable .*gralk.* disposed/);
    });
});
test_utils_1.describeMathCPUAndGPU('Create Variable', function () {
    it('From Tensor, no explicit name', function () {
        var v = V.variable(tfjs_core_1.zeros([2, 2]));
        expect(v.name.indexOf('Variable')).toEqual(0);
        expect(v.shape).toEqual([2, 2]);
        expect(v.read().dataSync()).toEqual(new Float32Array([0, 0, 0, 0]));
    });
    it('From Tensor, no explicit name', function () {
        var v = V.variable(tfjs_core_1.zeros([3]));
        expect(v.name.indexOf('Variable')).toEqual(0);
        expect(v.shape).toEqual([3]);
        expect(v.read().dataSync()).toEqual(new Float32Array([0, 0, 0]));
    });
    it('From Tensor, explicit name', function () {
        var v = V.variable(tfjs_core_1.zeros([3]), undefined, 'Var1');
        expect(v.name.indexOf('Var1')).toEqual(0);
        expect(v.shape).toEqual([3]);
        expect(v.read().dataSync()).toEqual(new Float32Array([0, 0, 0]));
    });
});
test_utils_1.describeMathCPUAndGPU('ZerosVariable', function () {
    it('Scalar', function () {
        var s = V.zerosVariable([], 'float32', 'Scalar');
        expect(s.name.indexOf('Scalar')).toEqual(0);
        expect(s.read().shape).toEqual([]);
        expect(s.read().dataSync()).toEqual(new Float32Array([0]));
    });
    it('Vector', function () {
        var v = V.zerosVariable([3], 'float32', 'Vector');
        expect(v.name.indexOf('Vector')).toEqual(0);
        expect(v.read().shape).toEqual([3]);
        expect(v.read().dataSync()).toEqual(new Float32Array([0, 0, 0]));
    });
    it('Matrix', function () {
        var m = V.zerosVariable([2, 2], 'float32', 'Matrix');
        expect(m.name.indexOf('Matrix')).toEqual(0);
        expect(m.read().shape).toEqual([2, 2]);
        expect(m.read().dataSync()).toEqual(new Float32Array([0, 0, 0, 0]));
    });
    it('3D', function () {
        var t = V.zerosVariable([2, 2, 2], 'float32', 'Tertiary');
        expect(t.name.indexOf('Tertiary')).toEqual(0);
        expect(t.read().shape).toEqual([2, 2, 2]);
        expect(t.read().dataSync()).toEqual(new Float32Array([
            0, 0, 0, 0, 0, 0, 0, 0
        ]));
    });
    it('4D', function () {
        var q = V.zerosVariable([1, 2, 1, 3], 'float32', 'Quaternary');
        expect(q.name.indexOf('Quaternary')).toEqual(0);
        expect(q.read().shape).toEqual([1, 2, 1, 3]);
        expect(q.read().dataSync()).toEqual(new Float32Array([0, 0, 0, 0, 0, 0]));
    });
});
test_utils_1.describeMathCPUAndGPU('OnesVariable', function () {
    it('Scalar', function () {
        var s = V.onesVariable([], 'float32', 'Scalar');
        expect(s.name.indexOf('Scalar')).toEqual(0);
        expect(s.read().shape).toEqual([]);
        expect(s.read().dataSync()).toEqual(new Float32Array([1]));
    });
    it('Vector', function () {
        var v = V.onesVariable([3], 'float32', 'Vector');
        expect(v.name.indexOf('Vector')).toEqual(0);
        expect(v.read().shape).toEqual([3]);
        expect(v.read().dataSync()).toEqual(new Float32Array([1, 1, 1]));
    });
    it('Matrix', function () {
        var m = V.onesVariable([2, 2], 'float32', 'Matrix');
        expect(m.name.indexOf('Matrix')).toEqual(0);
        expect(m.read().shape).toEqual([2, 2]);
        expect(m.read().dataSync()).toEqual(new Float32Array([1, 1, 1, 1]));
    });
    it('3D', function () {
        var t = V.onesVariable([2, 2, 2], 'float32', 'Tertiary');
        expect(t.name.indexOf('Tertiary')).toEqual(0);
        expect(t.read().shape).toEqual([2, 2, 2]);
        expect(t.read().dataSync()).toEqual(new Float32Array([
            1, 1, 1, 1, 1, 1, 1, 1
        ]));
    });
    it('4D', function () {
        var q = V.onesVariable([1, 2, 1, 3], 'float32', 'Quaternary');
        expect(q.name.indexOf('Quaternary')).toEqual(0);
        expect(q.read().shape).toEqual([1, 2, 1, 3]);
        expect(q.read().dataSync()).toEqual(new Float32Array([1, 1, 1, 1, 1, 1]));
    });
});
test_utils_1.describeMathCPUAndGPU('ZerosLike', function () {
    it('Scalar', function () {
        var s = V.zerosLike(tfjs_core_1.randomUniform([], -10, 10));
        expect(s.read().shape).toEqual([]);
        expect(s.read().dataSync()).toEqual(new Float32Array([0]));
    });
    it('Vector', function () {
        var v = V.zerosLike(tfjs_core_1.randomUniform([3], -10, 10));
        expect(v.read().shape).toEqual([3]);
        expect(v.read().dataSync()).toEqual(new Float32Array([0, 0, 0]));
    });
    it('Matrix', function () {
        var m = V.zerosLike(tfjs_core_1.randomUniform([2, 2], -10, 10));
        expect(m.read().shape).toEqual([2, 2]);
        expect(m.read().dataSync()).toEqual(new Float32Array([0, 0, 0, 0]));
    });
    it('3D', function () {
        var t = V.zerosLike(tfjs_core_1.randomUniform([2, 2, 2], -10, 10));
        expect(t.read().shape).toEqual([2, 2, 2]);
        expect(t.read().dataSync()).toEqual(new Float32Array([
            0, 0, 0, 0, 0, 0, 0, 0
        ]));
    });
    it('4D', function () {
        var q = V.zerosLike(tfjs_core_1.randomUniform([1, 2, 1, 3], -10, 10));
        expect(q.read().shape).toEqual([1, 2, 1, 3]);
        expect(q.read().dataSync()).toEqual(new Float32Array([0, 0, 0, 0, 0, 0]));
    });
});
test_utils_1.describeMathCPUAndGPU('OnesLike', function () {
    it('Scalar', function () {
        var s = V.onesLike(tfjs_core_1.randomUniform([], -10, 10));
        expect(s.read().shape).toEqual([]);
        expect(s.read().dataSync()).toEqual(new Float32Array([1]));
    });
    it('Vector', function () {
        var v = V.onesLike(tfjs_core_1.randomUniform([3], -10, 10));
        expect(v.read().shape).toEqual([3]);
        expect(v.read().dataSync()).toEqual(new Float32Array([1, 1, 1]));
    });
    it('Matrix', function () {
        var m = V.onesLike(tfjs_core_1.randomUniform([2, 2], -10, 10));
        expect(m.read().shape).toEqual([2, 2]);
        expect(m.read().dataSync()).toEqual(new Float32Array([1, 1, 1, 1]));
    });
    it('3D', function () {
        var t = V.onesLike(tfjs_core_1.randomUniform([2, 2, 2], -10, 10));
        expect(t.read().shape).toEqual([2, 2, 2]);
        expect(t.read().dataSync()).toEqual(new Float32Array([
            1, 1, 1, 1, 1, 1, 1, 1
        ]));
    });
    it('4D', function () {
        var q = V.onesLike(tfjs_core_1.randomUniform([1, 2, 1, 3], -10, 10));
        expect(q.read().shape).toEqual([1, 2, 1, 3]);
        expect(q.read().dataSync()).toEqual(new Float32Array([1, 1, 1, 1, 1, 1]));
    });
});
test_utils_1.describeMathCPUAndGPU('eye (I-matrix builder)', function () {
    it('Variable Zero sized 2D matrix works', function () {
        var v = V.eyeVariable(0);
        expect(v.shape).toEqual([0, 0]);
    });
    it('Variable 1 sized 2D matrix', function () {
        var I = V.eyeVariable(1);
        expect(I.shape).toEqual([1, 1]);
        expect(I.read().dataSync()).toEqual(new Float32Array([1]));
    });
    it('Variable 2 sized 2D matrix', function () {
        var I = V.eyeVariable(2);
        expect(I.shape).toEqual([2, 2]);
        expect(I.read().dataSync()).toEqual(new Float32Array([1, 0, 0, 1]));
    });
});
test_utils_1.describeMathCPUAndGPU('Variable update', function () {
    it('Update', function () {
        var v = new V.LayerVariable(tfjs_core_1.scalar(10.0));
        V.update(v, tfjs_core_1.scalar(20.0));
        test_utils_1.expectTensorsClose(v.read(), tfjs_core_1.scalar(20.0));
    });
    it('Update: Incompatible shape', function () {
        var v = new V.LayerVariable(tfjs_core_1.tensor1d([10.0, 20.0]));
        var x = tfjs_core_1.tensor1d([10.0, 20.0, 30.0]);
        expect(function () { return V.update(v, x); }).toThrowError();
    });
    it('UpdateAdd', function () {
        var v = new V.LayerVariable(tfjs_core_1.scalar(10.0));
        V.updateAdd(v, tfjs_core_1.scalar(20.0));
        test_utils_1.expectTensorsClose(v.read(), tfjs_core_1.scalar(30.0));
    });
    it('UpdateAdd: Incompatible shape', function () {
        var v = new V.LayerVariable(tfjs_core_1.tensor1d([10.0, 20.0]));
        var x = tfjs_core_1.tensor1d([0.0, 10.0, 20.0]);
        expect(function () { return V.updateAdd(v, x); }).toThrowError();
    });
    it('UpdateSub', function () {
        var v = new V.LayerVariable(tfjs_core_1.scalar(10.0));
        V.updateSub(v, tfjs_core_1.scalar(20.0));
        var vNew = v.read();
        test_utils_1.expectTensorsClose(vNew, tfjs_core_1.scalar(-10.0));
    });
    it('UpdateSub: Incompatible shape', function () {
        var v = new V.LayerVariable(tfjs_core_1.tensor1d([10.0, 20.0]));
        var x = tfjs_core_1.tensor1d([0.0, 10.0, 20.0]);
        expect(function () { return V.updateSub(v, x); }).toThrowError();
    });
});
test_utils_1.describeMathCPUAndGPU('batchGetValue', function () {
    it('Legnth-3 Array, Mixed Tensor and Variable', function () {
        var v1 = V.variable(tfjs_core_1.zeros([]));
        var v2 = V.variable(tfjs_core_1.zeros([2]));
        var v3 = V.variable(tfjs_core_1.zeros([2, 2]));
        var values = V.batchGetValue([v1, v2, v3]);
        expect(values.length).toEqual(3);
        expect(values[0].shape).toEqual([]);
        expect(values[0].dataSync()).toEqual(new Float32Array([0]));
        expect(values[1].shape).toEqual([2]);
        expect(values[1].dataSync()).toEqual(new Float32Array([0, 0]));
        expect(values[2].shape).toEqual([2, 2]);
        expect(values[2].dataSync()).toEqual(new Float32Array([0, 0, 0, 0]));
    });
});
test_utils_1.describeMathCPUAndGPU('batchSetValue', function () {
    it('Update using Tensor values', function () {
        var v1 = V.randomUniformVariable([2], 0, 1);
        var v2 = V.randomUniformVariable([2, 2], 0, 1);
        V.batchSetValue([[v1, tfjs_core_1.zeros([2])], [v2, tfjs_core_1.zeros([2, 2])]]);
        expect(v1.shape).toEqual([2]);
        expect(v1.read().dataSync()).toEqual(new Float32Array([0, 0]));
        expect(v2.shape).toEqual([2, 2]);
        expect(v2.read().dataSync()).toEqual(new Float32Array([0, 0, 0, 0]));
    });
    it('Update using Tensor values', function () {
        var v1 = V.randomUniformVariable([], 0, 1);
        var v2 = V.randomUniformVariable([2, 2, 1], 0, 1);
        V.batchSetValue([[v1, tfjs_core_1.zeros([])], [v2, tfjs_core_1.zeros([2, 2, 1])]]);
        expect(v1.shape).toEqual([]);
        expect(v1.read().dataSync()).toEqual(new Float32Array([0]));
        expect(v2.shape).toEqual([2, 2, 1]);
        expect(v2.read().dataSync()).toEqual(new Float32Array([0, 0, 0, 0]));
    });
    it('Update empty Array', function () {
        V.batchSetValue([]);
    });
});
test_utils_1.describeMathCPUAndGPU('gradients', function () {
    it('Simple mean: 1 variable', function () {
        var var1 = new V.LayerVariable(tfc.mul(tfjs_core_1.scalar(2.0), tfc.ones([2, 2])));
        var gradients = V.gradients(function () { return tfc.mean(var1.read()); }, [var1]);
        expect(gradients.length).toEqual(1);
        test_utils_1.expectTensorsClose(tfjs_core_1.tensor2d([[0.25, 0.25], [0.25, 0.25]], [2, 2]), gradients[0]);
    });
    it('Simple matmul and mean: 2 variables', function () {
        var var1 = new V.LayerVariable(tfjs_core_1.tensor2d([[1, 0], [0, 0]], [2, 2]));
        var var2 = new V.LayerVariable(tfjs_core_1.tensor2d([[1, 0], [0, 1]], [2, 2]));
        var gradients = V.gradients(function () { return tfc.mean(K.dot(var1.read(), var2.read())); }, [var1, var2]);
        expect(gradients.length).toEqual(2);
        test_utils_1.expectTensorsClose(tfjs_core_1.tensor2d([[0.25, 0.25], [0.25, 0.25]], [2, 2]), gradients[0]);
        test_utils_1.expectTensorsClose(tfjs_core_1.tensor2d([[0.25, 0.25], [0, 0]], [2, 2]), gradients[1]);
    });
});
//# sourceMappingURL=variables_test.js.map