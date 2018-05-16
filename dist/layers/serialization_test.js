"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var initializers_1 = require("../initializers");
var serialization_1 = require("./serialization");
describe('Deserialization', function () {
    it('Zeros Initialzer', function () {
        var config = { className: 'Zeros', config: {} };
        var initializer = serialization_1.deserialize(config);
        expect(initializer instanceof (initializers_1.Zeros)).toEqual(true);
    });
    it('Ones Initialzer', function () {
        var config = { className: 'Ones', config: {} };
        var initializer = serialization_1.deserialize(config);
        expect(initializer instanceof (initializers_1.Ones)).toEqual(true);
    });
});
//# sourceMappingURL=serialization_test.js.map