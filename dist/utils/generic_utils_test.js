"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var utils = require("./generic_utils");
describe('pyListRepeat() ', function () {
    it('creates an empty array for 0 numValues', function () {
        expect(utils.pyListRepeat(null, 0)).toEqual([]);
    });
    it('creates an array with 1 value for 1 numValues', function () {
        var value = 'a';
        expect(utils.pyListRepeat(value, 1)).toEqual([value]);
    });
    it('creates an array with 3 values for 3 numValues', function () {
        var value = 'a';
        var numValues = 3;
        var expectedValue = [value, value, value];
        expect(utils.pyListRepeat(value, numValues)).toEqual(expectedValue);
    });
    it('throws an exception when numValues <0', function () {
        var fillFn = function () { return utils.pyListRepeat(null, -1); };
        expect(fillFn).toThrowError();
    });
    it('takes an existing array and replicates its contents.', function () {
        var value = [1, 2];
        var numValues = 2;
        var expectedValue = [1, 2, 1, 2];
        expect(utils.pyListRepeat(value, numValues)).toEqual(expectedValue);
    });
});
describe('assert', function () {
    var _loop_1 = function (x) {
        it('throws error for false conditions', function () {
            expect(function () { return utils.assert(x); }).toThrowError();
        });
    };
    for (var _i = 0, _a = [false, null, undefined]; _i < _a.length; _i++) {
        var x = _a[_i];
        _loop_1(x);
    }
    it('doesn\'t throw error for true conditions', function () {
        expect(function () { return utils.assert(true); }).not.toThrowError();
    });
});
describe('count', function () {
    it('string array, non-empty', function () {
        var array = ['foo', 'bar', 'foo'];
        expect(utils.count(array, 'foo')).toEqual(2);
        expect(utils.count(array, 'bar')).toEqual(1);
        expect(utils.count(array, 'baz')).toEqual(0);
        expect(utils.count(array, '')).toEqual(0);
    });
    it('number array, non-empty', function () {
        var array = [-1, 1, 3, 3, 7, -1, 1.337, -1];
        expect(utils.count(array, 1)).toEqual(1);
        expect(utils.count(array, 3)).toEqual(2);
        expect(utils.count(array, 1.337)).toEqual(1);
        expect(utils.count(array, -1)).toEqual(3);
        expect(utils.count(array, 0)).toEqual(0);
    });
    it('string array, empty', function () {
        var array = [];
        expect(utils.count(array, 'foo')).toEqual(0);
        expect(utils.count(array, 'bar')).toEqual(0);
        expect(utils.count(array, 'baz')).toEqual(0);
        expect(utils.count(array, '')).toEqual(0);
    });
});
describe('Compare functions', function () {
    var inputs = [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]];
    var _loop_2 = function (input) {
        it('cmp sorts numbers in ascending order', function () {
            var expected = [1, 2, 3];
            expect(input.slice().sort(utils.numberCompare)).toEqual(expected);
        });
    };
    for (var _i = 0, inputs_1 = inputs; _i < inputs_1.length; _i++) {
        var input = inputs_1[_i];
        _loop_2(input);
    }
    var _loop_3 = function (input) {
        it('reverseCmp sorts numbers in ascending order', function () {
            var expected = [3, 2, 1];
            expect(input.slice().sort(utils.reverseNumberCompare)).toEqual(expected);
        });
    };
    for (var _a = 0, inputs_2 = inputs; _a < inputs_2.length; _a++) {
        var input = inputs_2[_a];
        _loop_3(input);
    }
});
describe('toList', function () {
    it('creates array from non-array.', function () {
        var value = 1;
        expect(utils.toList(value)).toEqual([value]);
    });
    it('returns array if passed an array.', function () {
        var value = [1];
        expect(utils.toList(value)).toEqual(value);
    });
});
describe('toSnakeCase', function () {
    var _loop_4 = function (inputString, expectedOutput) {
        it('creates expected output', function () {
            expect(utils.toSnakeCase(inputString)).toEqual(expectedOutput);
        });
    };
    for (var _i = 0, _a = [
        ['', ''], ['A', 'a'], ['AA', 'aa'], ['AAA', 'aaa'], ['AAa', 'a_aa'],
        ['AA0', 'a_a0'], ['aB', 'a_b'], ['aBC', 'a_bc'], ['aBc', 'a_bc'],
        ['_', 'private_'], ['a', 'a'], ['_a', 'private_a']
    ]; _i < _a.length; _i++) {
        var _b = _a[_i], inputString = _b[0], expectedOutput = _b[1];
        _loop_4(inputString, expectedOutput);
    }
});
describe('toCamelCase', function () {
    var _loop_5 = function (inputString, expectedOutput) {
        it('creates expected output', function () {
            expect(utils.toCamelCase(inputString)).toEqual(expectedOutput);
        });
    };
    for (var _i = 0, _a = [
        ['', ''], ['A', 'A'], ['aa', 'aa'], ['a_a', 'aA'],
        ['a_aa', 'aAa']
    ]; _i < _a.length; _i++) {
        var _b = _a[_i], inputString = _b[0], expectedOutput = _b[1];
        _loop_5(inputString, expectedOutput);
    }
});
describe('stringsEqual', function () {
    it('null and undefined', function () {
        expect(utils.stringsEqual(null, null)).toEqual(true);
        expect(utils.stringsEqual(undefined, undefined)).toEqual(true);
        expect(utils.stringsEqual(undefined, null)).toEqual(false);
        expect(utils.stringsEqual(undefined, [])).toEqual(false);
        expect(utils.stringsEqual(null, [])).toEqual(false);
        expect(utils.stringsEqual(null, ['a'])).toEqual(false);
    });
    it('Empty arrays', function () {
        expect(utils.stringsEqual([], [])).toEqual(true);
        expect(utils.stringsEqual([], ['a'])).toEqual(false);
    });
    it('Non-empty arrays', function () {
        expect(utils.stringsEqual(['a', 'b', 'c', null], [
            'a', 'b', 'c', null
        ])).toEqual(true);
        expect(utils.stringsEqual(['a', 'b', 'c', ''], [
            'a', 'b', 'c', ''
        ])).toEqual(true);
        expect(utils.stringsEqual(['a', 'b', 'c', null], [
            'a', 'b', 'c', undefined
        ])).toEqual(false);
        expect(utils.stringsEqual(['a', 'b', 'c', ''], [
            'a', 'c', 'b', ''
        ])).toEqual(false);
    });
});
describe('unique', function () {
    it('null or undefined', function () {
        expect(utils.unique(null)).toEqual(null);
        expect(utils.unique(undefined)).toEqual(undefined);
    });
    it('empty array', function () {
        expect(utils.unique([])).toEqual([]);
    });
    it('Non-empty array: string', function () {
        expect(utils.unique(['foo', 'bar', 'foo'])).toEqual(['foo', 'bar']);
        expect(utils.unique(['foo', 'bar', ''])).toEqual(['foo', 'bar', '']);
        expect(utils.unique(['foo', 'bar', null, ''])).toEqual([
            'foo', 'bar', null, ''
        ]);
    });
    it('Non-empty array: number', function () {
        expect(utils.unique([1, 2, -1, 2])).toEqual([1, 2, -1]);
        expect(utils.unique([2, 3, 2, null])).toEqual([2, 3, null]);
    });
});
describe('isObjectEmpty', function () {
    it('null or undefined', function () {
        expect(function () { return utils.isObjectEmpty(null); }).toThrowError();
        expect(function () { return utils.isObjectEmpty(undefined); }).toThrowError();
    });
    it('empty object', function () {
        expect(utils.isObjectEmpty({})).toEqual(true);
    });
    it('Non-empty object', function () {
        expect(utils.isObjectEmpty({ 'a': 12 })).toEqual(false);
        expect(utils.isObjectEmpty({ 'a': 12, 'b': 34 })).toEqual(false);
    });
});
describe('checkArrayTypeAndLength', function () {
    it('checks types', function () {
        expect(utils.checkArrayTypeAndLength([1, 2, 3], 'number')).toEqual(true);
        expect(utils.checkArrayTypeAndLength(['hello', 'world'], 'string'))
            .toEqual(true);
        expect(utils.checkArrayTypeAndLength([1, 2, [3]], 'number')).toEqual(false);
    });
    it('checks lengths', function () {
        expect(utils.checkArrayTypeAndLength([1, 2, 3], 'number', 1)).toEqual(true);
        expect(utils.checkArrayTypeAndLength([1, 2, 3], 'number', 1, 3))
            .toEqual(true);
        expect(utils.checkArrayTypeAndLength([1, 2, 3, 4, 5], 'number', 1, 3))
            .toEqual(false);
        expect(utils.checkArrayTypeAndLength([1, 2, 3, 4, 5], 'number', 7, 10))
            .toEqual(false);
        expect(utils.checkArrayTypeAndLength([], 'does_not_matter', 0, 0))
            .toEqual(true);
    });
    it('rejects negative length limits', function () {
        expect(function () { return utils.checkArrayTypeAndLength([1, 2, 3], 'number', -1); })
            .toThrowError();
    });
    it('rejects maxLength < minLength', function () {
        expect(function () { return utils.checkArrayTypeAndLength([1, 2, 3], 'number', 100, 2); })
            .toThrowError();
    });
});
//# sourceMappingURL=generic_utils_test.js.map