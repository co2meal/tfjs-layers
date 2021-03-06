"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var topology_1 = require("./engine/topology");
describe('SymbolicTensor Test', function () {
    it('Correct dtype and shape properties', function () {
        var st1 = new topology_1.SymbolicTensor('float32', [4, 6], null, [], {});
        expect(st1.dtype).toEqual('float32');
        expect(st1.shape).toEqual([4, 6]);
        expect(st1.rank).toEqual(2);
    });
    it('Correct when operating on scalars', function () {
        var scalar = new topology_1.SymbolicTensor('float32', [], null, [], {});
        expect(scalar.dtype).toEqual('float32');
        expect(scalar.shape).toEqual([]);
        expect(scalar.rank).toEqual(0);
    });
    it('Correct names and ids', function () {
        var st1 = new topology_1.SymbolicTensor('float32', [2, 2], null, [], {}, 'TestSymbolicTensor');
        var st2 = new topology_1.SymbolicTensor('float32', [2, 2], null, [], {}, 'TestSymbolicTensor');
        expect(st1.name.indexOf('TestSymbolicTensor')).toEqual(0);
        expect(st2.name.indexOf('TestSymbolicTensor')).toEqual(0);
        expect(st1 === st2).toBe(false);
        expect(st1.id).toBeGreaterThanOrEqual(0);
        expect(st2.id).toBeGreaterThanOrEqual(0);
        expect(st1.id === st2.id).toBe(false);
    });
    it('Invalid tensor name leads to error', function () {
        expect(function () { return new topology_1.SymbolicTensor('float32', [2, 2], null, [], {}, '!'); })
            .toThrowError();
    });
});
//# sourceMappingURL=types_test.js.map