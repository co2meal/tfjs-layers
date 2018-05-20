"use strict";
var __extends = (this && this.__extends) || (function () {
    var extendStatics = Object.setPrototypeOf ||
        ({ __proto__: [] } instanceof Array && function (d, b) { d.__proto__ = b; }) ||
        function (d, b) { for (var p in b) if (b.hasOwnProperty(p)) d[p] = b[p]; };
    return function (d, b) {
        extendStatics(d, b);
        function __() { this.constructor = d; }
        d.prototype = b === null ? Object.create(b) : (__.prototype = b.prototype, new __());
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
var tfjs_core_1 = require("@tensorflow/tfjs-core");
var topology_1 = require("../engine/topology");
var generic_utils = require("../utils/generic_utils");
var Upsampling = (function (_super) {
    __extends(Upsampling, _super);
    function Upsampling(config) {
        var _this = _super.call(this, config) || this;
        _this.newSize = config.newSize;
        return _this;
    }
    Upsampling.prototype.call = function (inputs, kwargs) {
        var _a = this.newSize, newHeight = _a[0], newWidth = _a[1];
        var input = generic_utils.getExactlyOneTensor(inputs);
        var resized = tfjs_core_1.image.resizeBilinear(input, [newHeight, newWidth], true);
        return resized;
    };
    Upsampling.prototype.computeOutputShape = function (inputShape) {
        return [null, this.newSize[0], this.newSize[1], inputShape[3]];
    };
    Upsampling.prototype.getConfig = function () {
        var config = {
            newSize: this.newSize,
        };
        var baseConfig = _super.prototype.getConfig.call(this);
        Object.assign(config, baseConfig);
        return config;
    };
    Upsampling.className = 'Upsampling';
    return Upsampling;
}(topology_1.Layer));
exports.Upsampling = Upsampling;
tfjs_core_1.serialization.SerializationMap.register(Upsampling);
//# sourceMappingURL=upsampling.js.map