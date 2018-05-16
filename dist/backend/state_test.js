"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var state_1 = require("../backend/state");
describe('getUID ', function () {
    it('second UID is different.', function () {
        var name = 'def';
        var firstUID = state_1.getUid(name);
        var secondUID = state_1.getUid(name);
        expect(secondUID).not.toEqual(firstUID);
    });
    it('with no prefix works and returns different UIDs.', function () {
        var firstUID = state_1.getUid();
        var secondUID = state_1.getUid();
        expect(firstUID).not.toEqual(secondUID);
    });
});
//# sourceMappingURL=state_test.js.map