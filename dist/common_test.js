"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var common_1 = require("./common");
describe('checkDataFormat', function () {
    it('Valid values', function () {
        var extendedValues = common_1.VALID_DATA_FORMAT_VALUES.concat([undefined, null]);
        for (var _i = 0, extendedValues_1 = extendedValues; _i < extendedValues_1.length; _i++) {
            var validValue = extendedValues_1[_i];
            common_1.checkDataFormat(validValue);
        }
    });
    it('Invalid values', function () {
        expect(function () { return common_1.checkDataFormat('foo'); }).toThrowError(/foo/);
        try {
            common_1.checkDataFormat('bad');
        }
        catch (e) {
            expect(e).toMatch('DataFormat');
            for (var _i = 0, VALID_DATA_FORMAT_VALUES_1 = common_1.VALID_DATA_FORMAT_VALUES; _i < VALID_DATA_FORMAT_VALUES_1.length; _i++) {
                var validValue = VALID_DATA_FORMAT_VALUES_1[_i];
                expect(e).toMatch(validValue);
            }
        }
    });
});
describe('checkPaddingMode', function () {
    it('Valid values', function () {
        var extendedValues = common_1.VALID_PADDING_MODE_VALUES.concat([undefined, null]);
        for (var _i = 0, extendedValues_2 = extendedValues; _i < extendedValues_2.length; _i++) {
            var validValue = extendedValues_2[_i];
            common_1.checkPaddingMode(validValue);
        }
    });
    it('Invalid values', function () {
        expect(function () { return common_1.checkPaddingMode('foo'); }).toThrowError(/foo/);
        try {
            common_1.checkPaddingMode('bad');
        }
        catch (e) {
            expect(e).toMatch('PaddingMode');
            for (var _i = 0, VALID_PADDING_MODE_VALUES_1 = common_1.VALID_PADDING_MODE_VALUES; _i < VALID_PADDING_MODE_VALUES_1.length; _i++) {
                var validValue = VALID_PADDING_MODE_VALUES_1[_i];
                expect(e).toMatch(validValue);
            }
        }
    });
});
describe('checkPoolMode', function () {
    it('Valid values', function () {
        var extendedValues = common_1.VALID_POOL_MODE_VALUES.concat([undefined, null]);
        for (var _i = 0, extendedValues_3 = extendedValues; _i < extendedValues_3.length; _i++) {
            var validValue = extendedValues_3[_i];
            common_1.checkPoolMode(validValue);
        }
    });
    it('Invalid values', function () {
        expect(function () { return common_1.checkPoolMode('foo'); }).toThrowError(/foo/);
        try {
            common_1.checkPoolMode('bad');
        }
        catch (e) {
            expect(e).toMatch('PoolMode');
            for (var _i = 0, VALID_POOL_MODE_VALUES_1 = common_1.VALID_POOL_MODE_VALUES; _i < VALID_POOL_MODE_VALUES_1.length; _i++) {
                var validValue = VALID_POOL_MODE_VALUES_1[_i];
                expect(e).toMatch(validValue);
            }
        }
    });
});
describe('isValidTensorName', function () {
    it('Valid tensor names', function () {
        expect(common_1.isValidTensorName('a')).toEqual(true);
        expect(common_1.isValidTensorName('A')).toEqual(true);
        expect(common_1.isValidTensorName('foo1')).toEqual(true);
        expect(common_1.isValidTensorName('Foo2')).toEqual(true);
        expect(common_1.isValidTensorName('n_1')).toEqual(true);
        expect(common_1.isValidTensorName('n.1')).toEqual(true);
        expect(common_1.isValidTensorName('n_1_2')).toEqual(true);
        expect(common_1.isValidTensorName('n.1.2')).toEqual(true);
        expect(common_1.isValidTensorName('a/B/c')).toEqual(true);
        expect(common_1.isValidTensorName('z_1/z_2/z.3')).toEqual(true);
    });
    it('Invalid tensor names: empty', function () {
        expect(common_1.isValidTensorName('')).toEqual(false);
    });
    it('Invalid tensor names: whitespaces', function () {
        expect(common_1.isValidTensorName('a b')).toEqual(false);
        expect(common_1.isValidTensorName('ab ')).toEqual(false);
    });
    it('Invalid tensor names: forbidden characters', function () {
        expect(common_1.isValidTensorName('foo1-2')).toEqual(false);
        expect(common_1.isValidTensorName('bar3!4')).toEqual(false);
    });
    it('Invalid tensor names: invalid first characters', function () {
        expect(common_1.isValidTensorName('/foo/bar')).toEqual(false);
        expect(common_1.isValidTensorName('.baz')).toEqual(false);
        expect(common_1.isValidTensorName('_baz')).toEqual(false);
        expect(common_1.isValidTensorName('1Qux')).toEqual(false);
    });
    it('Invalid tensor names: non-ASCII', function () {
        expect(common_1.isValidTensorName('フ')).toEqual(false);
        expect(common_1.isValidTensorName('ξ')).toEqual(false);
    });
});
describe('getUniqueTensorName', function () {
    it('Adds unique suffixes to tensor names', function () {
        expect(common_1.getUniqueTensorName('xx')).toEqual('xx');
        expect(common_1.getUniqueTensorName('xx')).toEqual('xx_1');
        expect(common_1.getUniqueTensorName('xx')).toEqual('xx_2');
        expect(common_1.getUniqueTensorName('xx')).toEqual('xx_3');
    });
    it('Correctly handles preexisting unique suffixes on tensor names', function () {
        expect(common_1.getUniqueTensorName('yy')).toEqual('yy');
        expect(common_1.getUniqueTensorName('yy')).toEqual('yy_1');
        expect(common_1.getUniqueTensorName('yy_1')).toEqual('yy_1_1');
        expect(common_1.getUniqueTensorName('yy')).toEqual('yy_2');
        expect(common_1.getUniqueTensorName('yy_1')).toEqual('yy_1_2');
        expect(common_1.getUniqueTensorName('yy_2')).toEqual('yy_2_1');
        expect(common_1.getUniqueTensorName('yy')).toEqual('yy_3');
        expect(common_1.getUniqueTensorName('yy_1_1')).toEqual('yy_1_1_1');
    });
});
//# sourceMappingURL=common_test.js.map