"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var packageJSON = require('../package.json');
var index_1 = require("./index");
describe('tfjs-core version consistency', function () {
    it('dev-peer match', function () {
        var tfjsCoreDevDepVersion = packageJSON.devDependencies['@tensorflow/tfjs-core'];
        var tfjsCorePeerDepVersion = packageJSON.peerDependencies['@tensorflow/tfjs-core'];
        expect(tfjsCoreDevDepVersion).toEqual(tfjsCorePeerDepVersion);
    });
    it('version.ts matches package version', function () {
        var expected = require('../package.json').version;
        expect(index_1.version_layers).toBe(expected);
    });
});
//# sourceMappingURL=version_test.js.map