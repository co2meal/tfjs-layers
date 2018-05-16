"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var serialization_utils_1 = require("./serialization_utils");
describe('convertPythonToTs', function () {
    it('primitives', function () {
        expect(serialization_utils_1.convertPythonicToTs(null)).toEqual(null);
        expect(serialization_utils_1.convertPythonicToTs(true)).toEqual(true);
        expect(serialization_utils_1.convertPythonicToTs(false)).toEqual(false);
        expect(serialization_utils_1.convertPythonicToTs(4)).toEqual(4);
    });
    it('strings w/o name tags', function () {
        expect(serialization_utils_1.convertPythonicToTs('abc')).toEqual('abc');
        expect(serialization_utils_1.convertPythonicToTs('ABC')).toEqual('ABC');
        expect(serialization_utils_1.convertPythonicToTs('one_two')).toEqual('oneTwo');
        expect(serialization_utils_1.convertPythonicToTs('OneTwo')).toEqual('OneTwo');
    });
    it('simple arrays', function () {
        expect(serialization_utils_1.convertPythonicToTs([])).toEqual([]);
        expect(serialization_utils_1.convertPythonicToTs([null])).toEqual([null]);
        expect(serialization_utils_1.convertPythonicToTs(['one_two'])).toEqual(['oneTwo']);
        expect(serialization_utils_1.convertPythonicToTs([null, true, false, 4, 'abc'])).toEqual([
            null, true, false, 4, 'abc'
        ]);
        expect(serialization_utils_1.convertPythonicToTs([[[]]])).toEqual([[[]]]);
    });
    it('layer tuple (array) with key', function () {
        for (var _i = 0, _a = ['inboundNodes', 'inputLayers', 'outputLayers']; _i < _a.length; _i++) {
            var key = _a[_i];
            expect(serialization_utils_1.convertPythonicToTs(['layer_name', 'meta_data', 0], key)).toEqual([
                'layer_name', 'metaData', 0
            ]);
        }
    });
    it('dictionary', function () {
        expect(serialization_utils_1.convertPythonicToTs({})).toEqual({});
        expect(serialization_utils_1.convertPythonicToTs({ key: null })).toEqual({ key: null });
        expect(serialization_utils_1.convertPythonicToTs({ key_one: 4 })).toEqual({ keyOne: 4 });
        expect(serialization_utils_1.convertPythonicToTs({ key_two: 'abc_def' })).toEqual({
            keyTwo: 'abcDef'
        });
        expect(serialization_utils_1.convertPythonicToTs({ key_one: true, key_two: false }))
            .toEqual({ keyOne: true, keyTwo: false });
        expect(serialization_utils_1.convertPythonicToTs({ name: 'layer_name' })).toEqual({
            name: 'layer_name'
        });
    });
    it('dictionary keys are passed down the stack', function () {
        var dict = { inbound_nodes: ['DoNotChange_Me', 0, null] };
        expect(serialization_utils_1.convertPythonicToTs(dict)).toEqual({
            inboundNodes: ['DoNotChange_Me', 0, null]
        });
    });
    it('enum promotion', function () {
        expect(serialization_utils_1.convertPythonicToTs({ mode: 'fan_out' })).toEqual({ mode: 'fanOut' });
        expect(serialization_utils_1.convertPythonicToTs({ distribution: 'normal' })).toEqual({
            distribution: 'normal'
        });
        expect(serialization_utils_1.convertPythonicToTs({ data_format: 'channels_last' })).toEqual({
            dataFormat: 'channelsLast'
        });
        expect(serialization_utils_1.convertPythonicToTs({ padding: 'valid' })).toEqual({ padding: 'valid' });
    });
});
describe('convertTsToPythonic', function () {
    it('primitives', function () {
        expect(serialization_utils_1.convertTsToPythonic(null)).toEqual(null);
        expect(serialization_utils_1.convertTsToPythonic(true)).toEqual(true);
        expect(serialization_utils_1.convertTsToPythonic(false)).toEqual(false);
        expect(serialization_utils_1.convertTsToPythonic(4)).toEqual(4);
    });
    it('strings w/o name tags', function () {
        expect(serialization_utils_1.convertTsToPythonic('abc')).toEqual('abc');
        expect(serialization_utils_1.convertTsToPythonic('ABC')).toEqual('abc');
        expect(serialization_utils_1.convertTsToPythonic('oneTwo')).toEqual('one_two');
        expect(serialization_utils_1.convertTsToPythonic('OneTwo')).toEqual('one_two');
    });
    it('simple arrays', function () {
        expect(serialization_utils_1.convertTsToPythonic([])).toEqual([]);
        expect(serialization_utils_1.convertTsToPythonic([null])).toEqual([null]);
        expect(serialization_utils_1.convertTsToPythonic(['oneTwo'])).toEqual(['one_two']);
        expect(serialization_utils_1.convertTsToPythonic([null, true, false, 4, 'abc'])).toEqual([
            null, true, false, 4, 'abc'
        ]);
        expect(serialization_utils_1.convertTsToPythonic([[[]]])).toEqual([[[]]]);
    });
    it('layer tuple (array) with key', function () {
        for (var _i = 0, _a = ['inboundNodes', 'inputLayers', 'outputLayers']; _i < _a.length; _i++) {
            var key = _a[_i];
            expect(serialization_utils_1.convertTsToPythonic(['layerName', 'metaData', 0], key)).toEqual([
                'layerName', 'meta_data', 0
            ]);
        }
    });
    it('dictionary', function () {
        expect(serialization_utils_1.convertTsToPythonic({})).toEqual({});
        expect(serialization_utils_1.convertTsToPythonic({ key: null })).toEqual({ key: null });
        expect(serialization_utils_1.convertTsToPythonic({ keyOne: 4 })).toEqual({ key_one: 4 });
        expect(serialization_utils_1.convertTsToPythonic({ keyTwo: 'abcDef' })).toEqual({
            key_two: 'abc_def'
        });
        expect(serialization_utils_1.convertTsToPythonic({ keyOne: true, keyTwo: false }))
            .toEqual({ key_one: true, key_two: false });
        expect(serialization_utils_1.convertTsToPythonic({ name: 'layerName' })).toEqual({
            name: 'layerName'
        });
    });
    it('dictionary keys are passed down the stack', function () {
        var dict = {
            inboundNodes: ['DoNotChange_Me', 0, null]
        };
        expect(serialization_utils_1.convertTsToPythonic(dict)).toEqual({
            inbound_nodes: ['DoNotChange_Me', 0, null]
        });
    });
    it('enum promotion', function () {
        expect(serialization_utils_1.convertTsToPythonic({ mode: 'fanOut' })).toEqual({ mode: 'fan_out' });
        expect(serialization_utils_1.convertTsToPythonic({ distribution: 'normal' })).toEqual({
            distribution: 'normal'
        });
        expect(serialization_utils_1.convertTsToPythonic({ dataFormat: 'channelsLast' })).toEqual({
            data_format: 'channels_last'
        });
        expect(serialization_utils_1.convertTsToPythonic({ padding: 'valid' })).toEqual({ padding: 'valid' });
    });
});
//# sourceMappingURL=serialization_utils_test.js.map