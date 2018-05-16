"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var types_utils = require("./types_utils");
describe('isArrayOfShapes', function () {
    it('returns false for a single non-empty shape', function () {
        expect(types_utils.isArrayOfShapes([1, 2, 3])).toEqual(false);
    });
    it('returns false for a single empty shape', function () {
        expect(types_utils.isArrayOfShapes([])).toEqual(false);
    });
    it('returns true for an array of shapes', function () {
        expect(types_utils.isArrayOfShapes([[1], [2, 3]])).toEqual(true);
    });
    it('returns true for an array of shapes that includes empty shapes', function () {
        expect(types_utils.isArrayOfShapes([[], [2, 3]])).toEqual(true);
        expect(types_utils.isArrayOfShapes([[]])).toEqual(true);
        expect(types_utils.isArrayOfShapes([[], []])).toEqual(true);
    });
});
describe('normalizeShapeList', function () {
    it('returns an empty list if an empty list is passed in.', function () {
        expect(types_utils.normalizeShapeList([])).toEqual([]);
    });
    it('returns a list of shapes if a single shape is passed in.', function () {
        expect(types_utils.normalizeShapeList([1])).toEqual([[1]]);
    });
    it('returns a list of shapes if an empty shape is passed in.', function () {
        expect(types_utils.normalizeShapeList([[]])).toEqual([[]]);
    });
    it('returns a list of shapes if a list of shapes is passed in.', function () {
        expect(types_utils.normalizeShapeList([[1]])).toEqual([[1]]);
    });
});
describe('getExactlyOneShape', function () {
    it('single instance', function () {
        expect(types_utils.getExactlyOneShape([1, 2, 3])).toEqual([1, 2, 3]);
        expect(types_utils.getExactlyOneShape([null, 8])).toEqual([null, 8]);
        expect(types_utils.getExactlyOneShape([])).toEqual([]);
    });
    it('Array of length 1', function () {
        expect(types_utils.getExactlyOneShape([[1, 2]])).toEqual([1, 2]);
        expect(types_utils.getExactlyOneShape([[]])).toEqual([]);
    });
    it('Array of length 2: ValueError', function () {
        expect(function () { return types_utils.getExactlyOneShape([
            [1], [2]
        ]); }).toThrowError(/Expected exactly 1 Shape; got 2/);
    });
});
//# sourceMappingURL=types_utils_test.js.map