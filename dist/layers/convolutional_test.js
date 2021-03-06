"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tfc = require("@tensorflow/tfjs-core");
var tfjs_core_1 = require("@tensorflow/tfjs-core");
var tfl = require("../index");
var test_utils_1 = require("../utils/test_utils");
var convolutional_1 = require("./convolutional");
test_utils_1.describeMathCPUAndGPU('conv1dWithBias', function () {
    var xLength4Data = [10, 20, 40, 80];
    var kernelLength2Data = [1, -1];
    var biasScalarData = 2.2;
    var outChannelsArray = [1, 2];
    var dataFormats = [undefined, 'channelsFirst', 'channelsLast'];
    var paddingModes = [undefined, 'same', 'valid'];
    var stride = 1;
    var _loop_1 = function (outChannels) {
        var _loop_2 = function (dataFormat) {
            var _loop_3 = function (paddingMode) {
                var testTitle = "outChannels=" + outChannels + ", stride=" + stride + ", " +
                    (paddingMode + ", " + dataFormat);
                it(testTitle, function () {
                    var x = tfjs_core_1.tensor3d(xLength4Data, [1, 4, 1]);
                    if (dataFormat === 'channelsFirst') {
                        x = tfc.transpose(x, [0, 2, 1]);
                    }
                    var kernelData = [];
                    var biasData = [];
                    for (var i = 0; i < outChannels; ++i) {
                        kernelData = kernelData.concat(kernelLength2Data);
                        biasData = biasData.concat([biasScalarData + i]);
                    }
                    var kernel = tfc.transpose(tfjs_core_1.tensor3d(kernelData, [1, outChannels, 2]), [2, 0, 1]);
                    var bias = tfjs_core_1.tensor1d(biasData);
                    var y = convolutional_1.conv1dWithBias(x, kernel, bias, stride, paddingMode, dataFormat);
                    var yExpectedShape;
                    var yExpectedData;
                    if (paddingMode === 'valid' || paddingMode === undefined) {
                        if (outChannels === 1) {
                            yExpectedShape = [1, 3, 1];
                            yExpectedData = [-7.8, -17.8, -37.8];
                        }
                        else if (outChannels === 2) {
                            yExpectedShape = [1, 3, 2];
                            yExpectedData = [-7.8, -6.8, -17.8, -16.8, -37.8, -36.8];
                        }
                    }
                    else if (paddingMode === 'same') {
                        if (outChannels === 1) {
                            yExpectedShape = [1, 4, 1];
                            yExpectedData = [-7.8, -17.8, -37.8, 82.2];
                        }
                        else if (outChannels === 2) {
                            yExpectedShape = [1, 4, 2];
                            yExpectedData =
                                [-7.8, -6.8, -17.8, -16.8, -37.8, -36.8, 82.2, 83.2];
                        }
                    }
                    test_utils_1.expectTensorsClose(y, tfjs_core_1.tensor3d(yExpectedData, yExpectedShape));
                });
            };
            for (var _i = 0, paddingModes_1 = paddingModes; _i < paddingModes_1.length; _i++) {
                var paddingMode = paddingModes_1[_i];
                _loop_3(paddingMode);
            }
        };
        for (var _i = 0, dataFormats_1 = dataFormats; _i < dataFormats_1.length; _i++) {
            var dataFormat = dataFormats_1[_i];
            _loop_2(dataFormat);
        }
    };
    for (var _i = 0, outChannelsArray_1 = outChannelsArray; _i < outChannelsArray_1.length; _i++) {
        var outChannels = outChannelsArray_1[_i];
        _loop_1(outChannels);
    }
});
test_utils_1.describeMathCPUAndGPU('conv1d', function () {
    var xLength4Data = [10, 20, 40, 80];
    var kernelLength2Data = [1, -1];
    var outChannels = 2;
    var dataFormat = 'channelsLast';
    var paddingMode = 'valid';
    var strides = [2, 1];
    var dilations = [1, 2];
    var expectations = [[-10, -10, -40, -40], [-30, -30, -60, -60]];
    var _loop_4 = function (i) {
        var stride = strides[i];
        var dilationRate = dilations[i];
        var expectation = expectations[i];
        var testTitle = "outChannels=" + outChannels + ", stride=" + stride + ", " +
            (paddingMode + ", dilationRate=" + dilationRate + ", " + dataFormat);
        it(testTitle, function () {
            var x = tfjs_core_1.tensor3d(xLength4Data, [1, 4, 1]);
            var kernelData = [];
            for (var i_1 = 0; i_1 < outChannels; ++i_1) {
                kernelData = kernelData.concat(kernelLength2Data);
            }
            var kernel = tfc.transpose(tfjs_core_1.tensor3d(kernelData, [1, outChannels, 2]), [2, 0, 1]);
            var y = convolutional_1.conv1d(x, kernel, stride, paddingMode, dataFormat, dilationRate);
            test_utils_1.expectTensorsClose(y, tfjs_core_1.tensor3d(expectation, [1, 2, 2]));
        });
    };
    for (var i = 0; i < strides.length; ++i) {
        _loop_4(i);
    }
});
test_utils_1.describeMathCPUAndGPU('conv2d', function () {
    var x4by4Data = [[[
                [10, 30, 50, 70], [20, 40, 60, 80], [-10, -30, -50, -70],
                [-20, -40, -60, -80]
            ]]];
    var kernel2by2Data = [1, 0, 0, -1];
    var dataFormats = [undefined, 'channelsFirst', 'channelsLast'];
    var paddingModes = [undefined, 'same', 'valid'];
    var stridesArray = [1, 2];
    var _loop_5 = function (dataFormat) {
        for (var _i = 0, paddingModes_2 = paddingModes; _i < paddingModes_2.length; _i++) {
            var paddingMode = paddingModes_2[_i];
            var _loop_6 = function (stride) {
                var testTitle = "stride=" + stride + ", " + paddingMode + ", " +
                    ("" + dataFormat);
                it(testTitle, function () {
                    var x = tfjs_core_1.tensor4d(x4by4Data, [1, 1, 4, 4]);
                    if (dataFormat !== 'channelsFirst') {
                        x = tfc.transpose(x, [0, 2, 3, 1]);
                    }
                    var kernel = tfjs_core_1.tensor4d(kernel2by2Data, [2, 2, 1, 1]);
                    var y = convolutional_1.conv2d(x, kernel, [stride, stride], 'valid', dataFormat);
                    var yExpected;
                    if (stride === 1) {
                        yExpected = tfjs_core_1.tensor4d([[[[-30, -30, -30], [50, 90, 130], [30, 30, 30]]]], [1, 1, 3, 3]);
                    }
                    else if (stride === 2) {
                        yExpected = tfjs_core_1.tensor4d([[[[-30, -30], [30, 30]]]], [1, 1, 2, 2]);
                    }
                    if (dataFormat !== 'channelsFirst') {
                        yExpected = tfc.transpose(yExpected, [0, 2, 3, 1]);
                    }
                    test_utils_1.expectTensorsClose(y, yExpected);
                });
            };
            for (var _a = 0, stridesArray_1 = stridesArray; _a < stridesArray_1.length; _a++) {
                var stride = stridesArray_1[_a];
                _loop_6(stride);
            }
        }
    };
    for (var _i = 0, dataFormats_2 = dataFormats; _i < dataFormats_2.length; _i++) {
        var dataFormat = dataFormats_2[_i];
        _loop_5(dataFormat);
    }
});
test_utils_1.describeMathCPUAndGPU('conv2dWithBias', function () {
    var x4by4Data = [[[
                [10, 30, 50, 70], [20, 40, 60, 80], [-10, -30, -50, -70],
                [-20, -40, -60, -80]
            ]]];
    var kernel2by2Data = [1, 0, 0, -1];
    var biasScalarData = [2.2];
    var outChannelsArray = [2, 3];
    var dataFormats = [undefined, 'channelsFirst', 'channelsLast'];
    var paddingModes = [undefined, 'same', 'valid'];
    var stridesArray = [1, 2];
    var _loop_7 = function (outChannels) {
        var _loop_8 = function (dataFormat) {
            for (var _i = 0, paddingModes_3 = paddingModes; _i < paddingModes_3.length; _i++) {
                var paddingMode = paddingModes_3[_i];
                var _loop_9 = function (stride) {
                    var testTitle = "outChannels=" + outChannels + ", stride=" + stride + ", " +
                        (paddingMode + ", " + dataFormat);
                    it(testTitle, function () {
                        var x = tfjs_core_1.tensor4d(x4by4Data, [1, 1, 4, 4]);
                        if (dataFormat !== 'channelsFirst') {
                            x = tfc.transpose(x, [0, 2, 3, 1]);
                        }
                        var kernelData = [];
                        var biasData = [];
                        for (var i = 0; i < outChannels; ++i) {
                            kernelData = kernelData.concat(kernel2by2Data);
                            biasData = biasData.concat(biasScalarData);
                        }
                        var kernel = tfc.transpose(tfjs_core_1.tensor4d(kernelData, [outChannels, 2, 2, 1]), [1, 2, 3, 0]);
                        var bias = tfjs_core_1.tensor1d(biasData);
                        var y = convolutional_1.conv2dWithBias(x, kernel, bias, [stride, stride], 'valid', dataFormat);
                        var yExpectedShape;
                        var yExpectedDataPerChannel;
                        if (stride === 1) {
                            yExpectedShape = [1, outChannels, 3, 3];
                            yExpectedDataPerChannel =
                                [-30, -30, -30, 50, 90, 130, 30, 30, 30];
                        }
                        else if (stride === 2) {
                            yExpectedShape = [1, outChannels, 2, 2];
                            yExpectedDataPerChannel = [-30, -30, 30, 30];
                        }
                        for (var i = 0; i < yExpectedDataPerChannel.length; ++i) {
                            yExpectedDataPerChannel[i] += biasScalarData[0];
                        }
                        var yExpectedData = [];
                        for (var i = 0; i < outChannels; ++i) {
                            yExpectedData = yExpectedData.concat(yExpectedDataPerChannel);
                        }
                        var yExpected = tfjs_core_1.tensor4d(yExpectedData, yExpectedShape);
                        if (dataFormat !== 'channelsFirst') {
                            yExpected = tfc.transpose(yExpected, [0, 2, 3, 1]);
                        }
                        test_utils_1.expectTensorsClose(y, yExpected);
                    });
                };
                for (var _a = 0, stridesArray_2 = stridesArray; _a < stridesArray_2.length; _a++) {
                    var stride = stridesArray_2[_a];
                    _loop_9(stride);
                }
            }
        };
        for (var _i = 0, dataFormats_3 = dataFormats; _i < dataFormats_3.length; _i++) {
            var dataFormat = dataFormats_3[_i];
            _loop_8(dataFormat);
        }
    };
    for (var _i = 0, outChannelsArray_2 = outChannelsArray; _i < outChannelsArray_2.length; _i++) {
        var outChannels = outChannelsArray_2[_i];
        _loop_7(outChannels);
    }
});
test_utils_1.describeMathCPU('Conv2D Layers: Symbolic', function () {
    var filtersArray = [1, 64];
    var paddingModes = [undefined, 'valid', 'same'];
    var dataFormats = ['channelsFirst', 'channelsLast'];
    var kernelSizes = [[2, 2], [3, 4]];
    var stridesArray = [undefined, 1];
    var _loop_10 = function (filters) {
        var _loop_11 = function (padding) {
            var _loop_12 = function (dataFormat) {
                var _loop_13 = function (kernelSize) {
                    var _loop_14 = function (stride) {
                        var strides = stride || kernelSize;
                        var testTitle = "filters=" + filters + ", kernelSize=" + JSON.stringify(kernelSize) + ", " +
                            ("strides=" + JSON.stringify(strides) + ", ") +
                            (dataFormat + ", " + padding);
                        it(testTitle, function () {
                            var inputShape = dataFormat === 'channelsFirst' ?
                                [2, 16, 11, 9] :
                                [2, 11, 9, 16];
                            var symbolicInput = new tfl.SymbolicTensor('float32', inputShape, null, [], null);
                            var conv2dLayer = tfl.layers.conv2d({
                                filters: filters,
                                kernelSize: kernelSize,
                                strides: strides,
                                padding: padding,
                                dataFormat: dataFormat,
                            });
                            var output = conv2dLayer.apply(symbolicInput);
                            var outputRows;
                            var outputCols;
                            if (stride === undefined) {
                                outputRows = kernelSize[0] === 2 ? 5 : 3;
                                if (padding === 'same') {
                                    outputRows++;
                                }
                                outputCols = kernelSize[1] === 2 ? 4 : 2;
                                if (padding === 'same') {
                                    outputCols++;
                                }
                            }
                            else {
                                outputRows = kernelSize[0] === 2 ? 10 : 9;
                                if (padding === 'same') {
                                    outputRows += kernelSize[0] - 1;
                                }
                                outputCols = kernelSize[1] === 2 ? 8 : 6;
                                if (padding === 'same') {
                                    outputCols += kernelSize[1] - 1;
                                }
                            }
                            var expectedShape;
                            if (dataFormat === 'channelsFirst') {
                                expectedShape = [2, filters, outputRows, outputCols];
                            }
                            else {
                                expectedShape = [2, outputRows, outputCols, filters];
                            }
                            expect(output.shape).toEqual(expectedShape);
                            expect(output.dtype).toEqual(symbolicInput.dtype);
                        });
                    };
                    for (var _i = 0, stridesArray_3 = stridesArray; _i < stridesArray_3.length; _i++) {
                        var stride = stridesArray_3[_i];
                        _loop_14(stride);
                    }
                };
                for (var _i = 0, kernelSizes_1 = kernelSizes; _i < kernelSizes_1.length; _i++) {
                    var kernelSize = kernelSizes_1[_i];
                    _loop_13(kernelSize);
                }
            };
            for (var _i = 0, dataFormats_4 = dataFormats; _i < dataFormats_4.length; _i++) {
                var dataFormat = dataFormats_4[_i];
                _loop_12(dataFormat);
            }
        };
        for (var _i = 0, paddingModes_4 = paddingModes; _i < paddingModes_4.length; _i++) {
            var padding = paddingModes_4[_i];
            _loop_11(padding);
        }
    };
    for (var _i = 0, filtersArray_1 = filtersArray; _i < filtersArray_1.length; _i++) {
        var filters = filtersArray_1[_i];
        _loop_10(filters);
    }
    it('missing config.kernelSize throws exception', function () {
        expect(function (filters) { return tfl.layers.conv2d({ filters: 1 }); })
            .toThrowError(/kernelSize/);
    });
    it('bad config.kernelSize shape throws exception', function () {
        expect(function () { return tfl.layers.conv2d({ filters: 1, kernelSize: [1, 1, 1] }); })
            .toThrowError(/kernelSize to be number or number\[\] with length 1 or 2/);
    });
    it('missing config.filters throws exception', function () {
        expect(function () { return tfl.layers.conv2d({ kernelSize: 1 }); })
            .toThrowError(/filters to be a 'number' > 0/);
    });
    it('bad config.filters value throws exception', function () {
        expect(function () { return tfl.layers.conv2d({ kernelSize: 1, filters: 0 }); })
            .toThrowError(/filters to be a 'number' > 0/);
    });
});
test_utils_1.describeMathCPUAndGPU('Conv2D Layer: Tensor', function () {
    var x4by4Data = [[[
                [10, 30, 50, 70], [20, 40, 60, 80], [-10, -30, -50, -70],
                [-20, -40, -60, -80]
            ]]];
    var useBiases = [false, true];
    var biasInitializers = ['zeros', 'ones'];
    var activations = [null, 'linear', 'relu'];
    var _loop_15 = function (useBias) {
        var _loop_16 = function (biasInitializer) {
            var _loop_17 = function (activation) {
                var testTitle = "useBias=" + useBias + ", biasInitializer=" + biasInitializer + ", " +
                    ("activation=" + activation);
                it(testTitle, function () {
                    var x = tfjs_core_1.tensor4d(x4by4Data, [1, 1, 4, 4]);
                    var conv2dLayer = tfl.layers.conv2d({
                        filters: 1,
                        kernelSize: [2, 2],
                        strides: [2, 2],
                        dataFormat: 'channelsFirst',
                        useBias: useBias,
                        kernelInitializer: 'ones',
                        biasInitializer: biasInitializer,
                        activation: activation
                    });
                    var y = conv2dLayer.apply(x);
                    var yExpectedData = [100, 260, -100, -260];
                    if (useBias && biasInitializer === 'ones') {
                        yExpectedData = yExpectedData.map(function (element) { return element + 1; });
                    }
                    if (activation === 'relu') {
                        yExpectedData =
                            yExpectedData.map(function (element) { return element >= 0 ? element : 0; });
                    }
                    var yExpected = tfjs_core_1.tensor4d(yExpectedData, [1, 1, 2, 2]);
                    test_utils_1.expectTensorsClose(y, yExpected);
                });
            };
            for (var _i = 0, activations_1 = activations; _i < activations_1.length; _i++) {
                var activation = activations_1[_i];
                _loop_17(activation);
            }
        };
        for (var _i = 0, biasInitializers_1 = biasInitializers; _i < biasInitializers_1.length; _i++) {
            var biasInitializer = biasInitializers_1[_i];
            _loop_16(biasInitializer);
        }
    };
    for (var _i = 0, useBiases_1 = useBiases; _i < useBiases_1.length; _i++) {
        var useBias = useBiases_1[_i];
        _loop_15(useBias);
    }
    it('CHANNEL_LAST', function () {
        var x = tfc.transpose(tfjs_core_1.tensor4d(x4by4Data, [1, 1, 4, 4]), [0, 2, 3, 1]);
        var conv2dLayer = tfl.layers.conv2d({
            filters: 1,
            kernelSize: [2, 2],
            strides: [2, 2],
            dataFormat: 'channelsLast',
            useBias: false,
            kernelInitializer: 'ones',
            activation: 'linear'
        });
        var y = conv2dLayer.apply(x);
        var yExpected = tfjs_core_1.tensor4d([100, 260, -100, -260], [1, 2, 2, 1]);
        test_utils_1.expectTensorsClose(y, yExpected);
    });
    var dilationRateValues = [2, [2, 2]];
    var _loop_18 = function (dilationRate) {
        it("CHANNEL_LAST, dilationRate=" + dilationRate, function () {
            var x = tfjs_core_1.tensor4d([[
                    [
                        [0.89240986], [0.54892443], [0.24670805], [0.03983783],
                        [0.56602233]
                    ],
                    [
                        [0.21421895], [0.58529864], [0.60060781], [0.66895784],
                        [0.08855761]
                    ],
                    [
                        [0.56657235], [0.25803428], [0.17971111], [0.65166403],
                        [0.70492866]
                    ],
                    [
                        [0.46641512], [0.05765411], [0.52517211], [0.62557303],
                        [0.30612501]
                    ],
                    [
                        [0.8406994], [0.56932724], [0.96028134], [0.34666753],
                        [0.04458038]
                    ]
                ]], [1, 5, 5, 1]);
            var conv2dLayer = tfl.layers.conv2d({
                filters: 1,
                kernelSize: [2, 2],
                strides: 1,
                dataFormat: 'channelsLast',
                useBias: false,
                kernelInitializer: 'ones',
                activation: 'linear',
                dilationRate: dilationRate
            });
            var y = conv2dLayer.apply(x);
            var yExpected = tfjs_core_1.tensor4d([[
                    [[1.8854014], [1.4984605], [1.6973702]],
                    [[1.8064139], [1.9374835], [1.5204625]],
                    [[2.547264], [1.8256931], [1.8895016]]
                ]], [1, 3, 3, 1]);
            test_utils_1.expectTensorsClose(y, yExpected);
        });
    };
    for (var _a = 0, dilationRateValues_1 = dilationRateValues; _a < dilationRateValues_1.length; _a++) {
        var dilationRate = dilationRateValues_1[_a];
        _loop_18(dilationRate);
    }
    var explicitDefaultDilations = [1, [1, 1]];
    var _loop_19 = function (explicitDefaultDilation) {
        var testTitle = 'Explicit default dilation rate: ' +
            JSON.stringify(explicitDefaultDilation);
        it(testTitle, function () {
            var conv2dLayer = tfl.layers.conv2d({
                filters: 1,
                kernelSize: [2, 2],
                strides: [2, 2],
                dataFormat: 'channelsFirst',
                useBias: false,
                kernelInitializer: 'ones',
                dilationRate: explicitDefaultDilation
            });
            var x = tfjs_core_1.tensor4d(x4by4Data, [1, 1, 4, 4]);
            var y = conv2dLayer.apply(x);
            var yExpected = tfjs_core_1.tensor4d([100, 260, -100, -260], [1, 1, 2, 2]);
            test_utils_1.expectTensorsClose(y, yExpected);
        });
    };
    for (var _b = 0, explicitDefaultDilations_1 = explicitDefaultDilations; _b < explicitDefaultDilations_1.length; _b++) {
        var explicitDefaultDilation = explicitDefaultDilations_1[_b];
        _loop_19(explicitDefaultDilation);
    }
});
test_utils_1.describeMathCPU('Conv2DTranspose: Symbolic', function () {
    var filtersArray = [1, 64];
    var paddingModes = [undefined, 'valid', 'same'];
    var kernelSizes = [2, [2, 2], [3, 4]];
    var stridesArray = [undefined, 2];
    var _loop_20 = function (filters) {
        var _loop_21 = function (padding) {
            var _loop_22 = function (kernelSize) {
                var _loop_23 = function (strides) {
                    var testTitle = "filters=" + filters + ", paddingMode=" + padding + "," +
                        ("kernelSize=" + JSON.stringify(kernelSize) + ", strides=" + strides);
                    it(testTitle, function () {
                        var inputShape = [2, 11, 9, 16];
                        var x = new tfl.SymbolicTensor('float32', inputShape, null, [], null);
                        var layer = tfl.layers.conv2dTranspose({ filters: filters, kernelSize: kernelSize, padding: padding, strides: strides });
                        var y = layer.apply(x);
                        var expectedShape;
                        if (strides === undefined) {
                            if (padding === 'valid' || padding === undefined) {
                                if (kernelSize === 2 ||
                                    tfjs_core_1.util.arraysEqual(kernelSize, [2, 2])) {
                                    expectedShape = [2, 12, 10, filters];
                                }
                                else if (tfjs_core_1.util.arraysEqual(kernelSize, [3, 4])) {
                                    expectedShape = [2, 13, 12, filters];
                                }
                            }
                            else if (padding === 'same') {
                                expectedShape = [2, 11, 9, filters];
                            }
                        }
                        else {
                            if (padding === 'valid' || padding === undefined) {
                                if (kernelSize === 2 ||
                                    tfjs_core_1.util.arraysEqual(kernelSize, [2, 2])) {
                                    expectedShape = [2, 22, 18, filters];
                                }
                                else if (tfjs_core_1.util.arraysEqual(kernelSize, [3, 4])) {
                                    expectedShape = [2, 23, 20, filters];
                                }
                            }
                            else if (padding === 'same') {
                                expectedShape = [2, 22, 18, filters];
                            }
                        }
                        expect(y.shape).toEqual(expectedShape);
                    });
                };
                for (var _i = 0, stridesArray_4 = stridesArray; _i < stridesArray_4.length; _i++) {
                    var strides = stridesArray_4[_i];
                    _loop_23(strides);
                }
            };
            for (var _i = 0, kernelSizes_2 = kernelSizes; _i < kernelSizes_2.length; _i++) {
                var kernelSize = kernelSizes_2[_i];
                _loop_22(kernelSize);
            }
        };
        for (var _i = 0, paddingModes_5 = paddingModes; _i < paddingModes_5.length; _i++) {
            var padding = paddingModes_5[_i];
            _loop_21(padding);
        }
    };
    for (var _i = 0, filtersArray_2 = filtersArray; _i < filtersArray_2.length; _i++) {
        var filters = filtersArray_2[_i];
        _loop_20(filters);
    }
    it('Correct weight names', function () {
        var x = new tfl.SymbolicTensor('float32', [1, 2, 3, 4], null, [], null);
        var layer = tfl.layers.conv2dTranspose({ filters: 2, kernelSize: [3, 3] });
        layer.apply(x);
        expect(layer.weights.length).toEqual(2);
        expect(layer.weights[0].name.indexOf('/kernel')).toBeGreaterThan(0);
        expect(layer.weights[1].name.indexOf('/bias')).toBeGreaterThan(0);
    });
});
test_utils_1.describeMathCPUAndGPU('Conv2DTranspose: Tensor', function () {
    var dataFormats = ['channelsFirst', 'channelsLast'];
    var stridesArray = [2, [2, 2]];
    var _loop_24 = function (dataFormat) {
        for (var _i = 0, stridesArray_5 = stridesArray; _i < stridesArray_5.length; _i++) {
            var strides = stridesArray_5[_i];
            var testTitle = "filters=8, kernelSize=[2,2], padding=valid, strides=" + strides +
                ("dataFormat=" + dataFormat);
            it(testTitle, function () {
                var filters = 8;
                var kernelSize = [2, 2];
                var padding = 'valid';
                var strides = 2;
                var layer = tfl.layers.conv2dTranspose({
                    filters: filters,
                    kernelSize: kernelSize,
                    padding: padding,
                    strides: strides,
                    dataFormat: dataFormat,
                    kernelInitializer: 'ones',
                    biasInitializer: 'ones'
                });
                var x = tfc.ones([2, 3, 4, 2]);
                var y = layer.apply(x);
                if (dataFormat === 'channelsLast') {
                    test_utils_1.expectTensorsClose(y, tfc.ones([2, 6, 8, 8]).mul(tfjs_core_1.scalar(3)));
                }
                else {
                    test_utils_1.expectTensorsClose(y, tfc.ones([2, 8, 8, 4]).mul(tfjs_core_1.scalar(4)));
                }
            });
        }
    };
    for (var _i = 0, dataFormats_5 = dataFormats; _i < dataFormats_5.length; _i++) {
        var dataFormat = dataFormats_5[_i];
        _loop_24(dataFormat);
    }
});
test_utils_1.describeMathCPU('Conv1D Layers: Symbolic', function () {
    var filtersArray = [1, 4];
    var paddingModes = [undefined, 'valid', 'same'];
    var stridesArray = [undefined, 1];
    var _loop_25 = function (filters) {
        var _loop_26 = function (padding) {
            var _loop_27 = function (strides) {
                var testTitle = "filters=" + filters + ", padding=" + padding + ", " +
                    ("strides=" + strides);
                it(testTitle, function () {
                    var inputShape = [2, 8, 3];
                    var symbolicInput = new tfl.SymbolicTensor('float32', inputShape, null, [], null);
                    var conv1dLayer = tfl.layers.conv1d({
                        filters: filters,
                        kernelSize: 2,
                        strides: strides,
                        padding: padding,
                        dataFormat: 'channelsLast',
                    });
                    var output = conv1dLayer.apply(symbolicInput);
                    var expectedShape = [2, 7, filters];
                    if (padding === 'same') {
                        expectedShape[1] = 8;
                    }
                    expect(output.shape).toEqual(expectedShape);
                    expect(output.dtype).toEqual(symbolicInput.dtype);
                });
            };
            for (var _i = 0, stridesArray_6 = stridesArray; _i < stridesArray_6.length; _i++) {
                var strides = stridesArray_6[_i];
                _loop_27(strides);
            }
        };
        for (var _i = 0, paddingModes_6 = paddingModes; _i < paddingModes_6.length; _i++) {
            var padding = paddingModes_6[_i];
            _loop_26(padding);
        }
    };
    for (var _i = 0, filtersArray_3 = filtersArray; _i < filtersArray_3.length; _i++) {
        var filters = filtersArray_3[_i];
        _loop_25(filters);
    }
    it('bad config.kernelSize shape throws exception', function () {
        expect(function () { return tfl.layers.conv1d({ filters: 1, kernelSize: [1, 1] }); })
            .toThrowError(/kernelSize.*1/);
    });
});
test_utils_1.describeMathCPUAndGPU('Conv1D Layer: Tensor', function () {
    var xLength4Data = [10, -30, -50, 70];
    var stridesValues = [1, 2];
    var activations = ['linear', 'relu'];
    var _loop_28 = function (strides) {
        var _loop_29 = function (activation) {
            var testTitle = "useBias=true, biasInitializer=ones, " +
                ("activation=" + activation + "; strides=" + strides);
            it(testTitle, function () {
                var x = tfjs_core_1.tensor3d(xLength4Data, [1, 4, 1]);
                var conv1dLayer = tfl.layers.conv1d({
                    filters: 1,
                    kernelSize: 2,
                    strides: strides,
                    dataFormat: 'channelsLast',
                    useBias: true,
                    kernelInitializer: 'ones',
                    biasInitializer: 'ones',
                    activation: activation
                });
                var y = conv1dLayer.apply(x);
                var yExpectedShape;
                var yExpectedData;
                if (strides === 1) {
                    yExpectedShape = [1, 3, 1];
                    yExpectedData = [-19, -79, 21];
                }
                else {
                    yExpectedShape = [1, 2, 1];
                    yExpectedData = [-19, 21];
                }
                if (activation === 'relu') {
                    yExpectedData = yExpectedData.map(function (x) { return x > 0 ? x : 0; });
                }
                var yExpected = tfjs_core_1.tensor3d(yExpectedData, yExpectedShape);
                test_utils_1.expectTensorsClose(y, yExpected);
            });
        };
        for (var _i = 0, activations_2 = activations; _i < activations_2.length; _i++) {
            var activation = activations_2[_i];
            _loop_29(activation);
        }
    };
    for (var _i = 0, stridesValues_1 = stridesValues; _i < stridesValues_1.length; _i++) {
        var strides = stridesValues_1[_i];
        _loop_28(strides);
    }
    var dilationRates = [2, [2]];
    var _loop_30 = function (dilationRate) {
        it("dilationRate = " + dilationRate, function () {
            var x = tfjs_core_1.tensor3d([
                0.0024236, 0.54829558, 0.47628448, 0.2971449, 0.7984293, 0.71802861,
                0.53109141, 0.85882819
            ], [1, 8, 1]);
            var conv1dLayer = tfl.layers.conv1d({
                filters: 1,
                kernelSize: 2,
                strides: 1,
                useBias: true,
                kernelInitializer: 'ones',
                biasInitializer: 'ones',
                dilationRate: dilationRate,
            });
            var y = conv1dLayer.apply(x);
            var yExpected = tfjs_core_1.tensor3d([1.478708, 1.8454404, 2.2747138, 2.0151734, 2.3295207, 2.5768569], [1, 6, 1]);
            test_utils_1.expectTensorsClose(y, yExpected);
        });
    };
    for (var _a = 0, dilationRates_1 = dilationRates; _a < dilationRates_1.length; _a++) {
        var dilationRate = dilationRates_1[_a];
        _loop_30(dilationRate);
    }
    it('missing config.kernelSize throws exception', function () {
        expect(function (filters) { return tfl.layers.conv1d({ filters: 1 }); })
            .toThrowError(/required key 'kernelSize' not in config/);
    });
    it('bad config.kernelSize throws exception', function () {
        expect(function () { return tfl.layers.conv1d({ filters: 1, kernelSize: [1, 1, 1] }); })
            .toThrowError(/kernelSize to be number or number\[\] with length 1/);
    });
    it('missing config.filters throws exception', function () {
        expect(function () { return tfl.layers.conv1d({ kernelSize: 1 }); })
            .toThrowError(/filters to be a 'number' > 0/);
    });
    it('bad config.filters throws exception', function () {
        expect(function () { return tfl.layers.conv1d({ kernelSize: 1, filters: 0 }); })
            .toThrowError(/filters to be a 'number' > 0/);
    });
});
test_utils_1.describeMathCPU('SeparableConv2D Layers: Symbolic', function () {
    var filtersArray = [1, 8];
    var paddingModes = [undefined, 'valid', 'same'];
    var dataFormats = ['channelsFirst', 'channelsLast'];
    var kernelSizes = [[2, 2], [3, 4]];
    var stridesArray = [undefined, 1];
    var dilationRates = [undefined, 2];
    var _loop_31 = function (filters) {
        var _loop_32 = function (padding) {
            var _loop_33 = function (dataFormat) {
                var _loop_34 = function (kernelSize) {
                    var _loop_35 = function (stride) {
                        var _loop_36 = function (dilationRate) {
                            var strides = stride || kernelSize;
                            var testTitle = "filters=" + filters + ", kernelSize=" + JSON.stringify(kernelSize) + ", " +
                                ("strides=" + JSON.stringify(strides) + ", ") +
                                ("dataFormat=" + dataFormat + ", padding=" + padding + ", ") +
                                ("dilationRate=" + dilationRate);
                            it(testTitle, function () {
                                var inputShape = dataFormat === 'channelsFirst' ?
                                    [2, 16, 11, 9] :
                                    [2, 11, 9, 16];
                                var symbolicInput = new tfl.SymbolicTensor('float32', inputShape, null, [], null);
                                var layer = tfl.layers.separableConv2d({
                                    filters: filters,
                                    kernelSize: kernelSize,
                                    strides: strides,
                                    padding: padding,
                                    dataFormat: dataFormat,
                                    dilationRate: dilationRate,
                                });
                                var output = layer.apply(symbolicInput);
                                var outputRows;
                                var outputCols;
                                if (dilationRate == null) {
                                    if (stride === undefined) {
                                        outputRows = kernelSize[0] === 2 ? 5 : 3;
                                        if (padding === 'same') {
                                            outputRows++;
                                        }
                                        outputCols = kernelSize[1] === 2 ? 4 : 2;
                                        if (padding === 'same') {
                                            outputCols++;
                                        }
                                    }
                                    else {
                                        outputRows = kernelSize[0] === 2 ? 10 : 9;
                                        if (padding === 'same') {
                                            outputRows += kernelSize[0] - 1;
                                        }
                                        outputCols = kernelSize[1] === 2 ? 8 : 6;
                                        if (padding === 'same') {
                                            outputCols += kernelSize[1] - 1;
                                        }
                                    }
                                }
                                else {
                                    if (padding === 'same') {
                                        if (stride === undefined) {
                                            outputRows = kernelSize[0] === 2 ? 6 : 4;
                                            outputCols = kernelSize[1] === 2 ? 5 : 3;
                                        }
                                        else {
                                            outputRows = 11;
                                            outputCols = 9;
                                        }
                                    }
                                    else {
                                        if (stride === undefined) {
                                            outputRows = kernelSize[0] === 2 ? 5 : 3;
                                            outputCols = kernelSize[1] === 2 ? 4 : 1;
                                        }
                                        else {
                                            outputRows = kernelSize[0] === 2 ? 9 : 7;
                                            outputCols = kernelSize[1] === 2 ? 7 : 3;
                                        }
                                    }
                                }
                                var expectedShape;
                                if (dataFormat === 'channelsFirst') {
                                    expectedShape = [2, filters, outputRows, outputCols];
                                }
                                else {
                                    expectedShape = [2, outputRows, outputCols, filters];
                                }
                                expect(output.shape).toEqual(expectedShape);
                                expect(output.dtype).toEqual(symbolicInput.dtype);
                            });
                        };
                        for (var _i = 0, dilationRates_2 = dilationRates; _i < dilationRates_2.length; _i++) {
                            var dilationRate = dilationRates_2[_i];
                            _loop_36(dilationRate);
                        }
                    };
                    for (var _i = 0, stridesArray_7 = stridesArray; _i < stridesArray_7.length; _i++) {
                        var stride = stridesArray_7[_i];
                        _loop_35(stride);
                    }
                };
                for (var _i = 0, kernelSizes_3 = kernelSizes; _i < kernelSizes_3.length; _i++) {
                    var kernelSize = kernelSizes_3[_i];
                    _loop_34(kernelSize);
                }
            };
            for (var _i = 0, dataFormats_6 = dataFormats; _i < dataFormats_6.length; _i++) {
                var dataFormat = dataFormats_6[_i];
                _loop_33(dataFormat);
            }
        };
        for (var _i = 0, paddingModes_7 = paddingModes; _i < paddingModes_7.length; _i++) {
            var padding = paddingModes_7[_i];
            _loop_32(padding);
        }
    };
    for (var _i = 0, filtersArray_4 = filtersArray; _i < filtersArray_4.length; _i++) {
        var filters = filtersArray_4[_i];
        _loop_31(filters);
    }
    it('Incorrect input rank throws error', function () {
        var layer = tfl.layers.separableConv2d({
            filters: 1,
            kernelSize: [2, 2],
            strides: 1,
        });
        var symbolicInput = new tfl.SymbolicTensor('float32', [2, 3, 4], null, [], null);
        expect(function () { return layer.apply(symbolicInput); }).toThrowError(/rank 4/);
    });
    it('Undefined channel axis throws error', function () {
        var layer = tfl.layers.separableConv2d({
            filters: 1,
            kernelSize: [2, 2],
            strides: 1,
        });
        var symbolicInput = new tfl.SymbolicTensor('float32', [1, , 2, 3, null], null, [], null);
        expect(function () { return layer.apply(symbolicInput); })
            .toThrowError(/channel dimension .* should be defined/);
    });
});
test_utils_1.describeMathGPU('SeparableConv2D Layer: Tensor', function () {
    var x5by5Data = [
        1, 3, 5, 7, 9, 2, 4, 6, 8, 10, -1, -3, -5,
        -7, -9, -2, -4, -6, -8, -10, -1, 1, -1, 1, -1
    ];
    var dataFormats = ['channelsFirst', 'channelsLast'];
    var dilationRates = [undefined, 2];
    var useBiases = [false, true];
    var biasInitializers = ['zeros', 'ones'];
    var activations = [null, 'linear', 'relu'];
    var _loop_37 = function (dataFormat) {
        var _loop_38 = function (dilationRate) {
            var _loop_39 = function (useBias) {
                var _loop_40 = function (biasInitializer) {
                    var _loop_41 = function (activation) {
                        var testTitle = "dataFormat=" + dataFormat + ", " +
                            ("dilationRate=" + dilationRate + ", ") +
                            ("useBias=" + useBias + ", biasInitializer=" + biasInitializer + ", ") +
                            ("activation=" + activation);
                        it(testTitle, function () {
                            var x = tfjs_core_1.tensor4d(x5by5Data, [1, 5, 5, 1]);
                            if (dataFormat === 'channelsFirst') {
                                x = tfc.transpose(x, [0, 3, 1, 2]);
                            }
                            var conv2dLayer = tfl.layers.separableConv2d({
                                depthMultiplier: 1,
                                filters: 1,
                                kernelSize: [2, 2],
                                strides: 1,
                                dilationRate: dilationRate,
                                dataFormat: dataFormat,
                                useBias: useBias,
                                depthwiseInitializer: 'ones',
                                pointwiseInitializer: 'ones',
                                biasInitializer: biasInitializer,
                                activation: activation
                            });
                            var y = conv2dLayer.apply(x);
                            var yExpectedData;
                            if (dilationRate === 2) {
                                yExpectedData = [0, 0, 0, 0, 0, 0, -8, -8, -16];
                            }
                            else {
                                yExpectedData = [
                                    10, 18, 26, 34, 2, 2, 2, 2, -10, -18, -26, -34, -6, -10, -14,
                                    -18
                                ];
                            }
                            if (useBias && biasInitializer === 'ones') {
                                yExpectedData = yExpectedData.map(function (element) { return element + 1; });
                            }
                            if (activation === 'relu') {
                                yExpectedData =
                                    yExpectedData.map(function (element) { return element >= 0 ? element : 0; });
                            }
                            var yExpected = dilationRate === 2 ?
                                tfjs_core_1.tensor4d(yExpectedData, [1, 3, 3, 1]) :
                                tfjs_core_1.tensor4d(yExpectedData, [1, 4, 4, 1]);
                            if (dataFormat === 'channelsFirst') {
                                yExpected = tfc.transpose(yExpected, [0, 3, 1, 2]);
                            }
                            test_utils_1.expectTensorsClose(y, yExpected);
                        });
                    };
                    for (var _i = 0, activations_3 = activations; _i < activations_3.length; _i++) {
                        var activation = activations_3[_i];
                        _loop_41(activation);
                    }
                };
                for (var _i = 0, biasInitializers_2 = biasInitializers; _i < biasInitializers_2.length; _i++) {
                    var biasInitializer = biasInitializers_2[_i];
                    _loop_40(biasInitializer);
                }
            };
            for (var _i = 0, useBiases_2 = useBiases; _i < useBiases_2.length; _i++) {
                var useBias = useBiases_2[_i];
                _loop_39(useBias);
            }
        };
        for (var _i = 0, dilationRates_3 = dilationRates; _i < dilationRates_3.length; _i++) {
            var dilationRate = dilationRates_3[_i];
            _loop_38(dilationRate);
        }
    };
    for (var _i = 0, dataFormats_7 = dataFormats; _i < dataFormats_7.length; _i++) {
        var dataFormat = dataFormats_7[_i];
        _loop_37(dataFormat);
    }
    it('missing config.kernelSize throws exception', function () {
        expect(function (filters) { return tfl.layers.separableConv2d({ filters: 1 }); })
            .toThrowError(/kernelSize/);
    });
    it('bad config.kernelSize throws exception', function () {
        expect(function () { return tfl.layers.separableConv2d({ filters: 1, kernelSize: [1, 1, 1] }); })
            .toThrowError(/kernelSize/);
    });
    it('missing config.filters throws exception', function () {
        expect(function () { return tfl.layers.separableConv2d({ kernelSize: 1 }); })
            .toThrowError(/filters/);
    });
    it('bad config.filters throws exception', function () {
        expect(function () { return tfl.layers.separableConv2d({ kernelSize: 1, filters: 0 }); })
            .toThrowError(/filters/);
    });
});
describe('Cropping2D Layer', function () {
    it('check with undefined channels type', function () {
        var layer = tfl.layers.cropping2D({ cropping: [[1, 0], [1, 0]] });
        var x = tfjs_core_1.tensor4d([
            [[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]],
        ], [1, 3, 3, 1]);
        var y = tfjs_core_1.tensor4d([
            [[[5], [6]], [[8], [9]]],
        ], [1, 2, 2, 1]);
        test_utils_1.expectTensorsClose(layer.apply(x, null), y);
    });
    it('check with channels last', function () {
        var layer = tfl.layers.cropping2D({ cropping: [[1, 1], [1, 1]], dataFormat: 'channelsLast' });
        var x = tfjs_core_1.tensor4d([
            [[[1], [2], [3]], [[4], [5], [6]], [[7], [8], [9]]],
        ], [1, 3, 3, 1]);
        var y = tfjs_core_1.tensor4d([
            [[[5]]],
        ], [1, 1, 1, 1]);
        test_utils_1.expectTensorsClose(layer.apply(x, null), y);
    });
    it('check with channels first', function () {
        var layer = tfl.layers.cropping2D({ cropping: [[1, 1], [1, 1]], dataFormat: 'channelsFirst' });
        var x = tfjs_core_1.tensor4d([
            [[[1, 2, 3], [3, 4, 5], [6, 7, 8]]],
        ], [1, 1, 3, 3]);
        var y = tfjs_core_1.tensor4d([
            [[[4]]],
        ], [1, 1, 1, 1]);
        test_utils_1.expectTensorsClose(layer.apply(x, null), y);
    });
});
test_utils_1.describeMathCPU('UpSampling2D Layer: Symbolic', function () {
    var dataFormats = ['channelsFirst', 'channelsLast'];
    var sizes = [undefined, [2, 2]];
    var undeterminedDimensionArray = [null, 'height', 'both'];
    var _loop_42 = function (dataFormat) {
        var _loop_43 = function (size) {
            var _loop_44 = function (undeterminedDimension) {
                var testTitle = "size = " + size + ", " + dataFormat +
                    ("undetermined dimension = " + JSON.stringify(undeterminedDimension));
                it(testTitle, function () {
                    var inputShape;
                    if (undeterminedDimension == null) {
                        inputShape = dataFormat === 'channelsFirst' ? [2, 16, 11, 9] :
                            [2, 11, 9, 16];
                    }
                    else if (undeterminedDimension === 'height') {
                        inputShape = dataFormat === 'channelsFirst' ? [2, 16, null, 9] :
                            [2, null, 9, 16];
                    }
                    else if (undeterminedDimension === 'both') {
                        inputShape = dataFormat === 'channelsFirst' ? [2, 16, null, null] :
                            [2, null, null, 16];
                    }
                    var symbolicInput = new tfl.SymbolicTensor('float32', inputShape, null, [], null);
                    var upSampling2dLayer = tfl.layers.upSampling2d({
                        size: size,
                        dataFormat: dataFormat,
                    });
                    var output = upSampling2dLayer.apply(symbolicInput);
                    var outputRows;
                    var outputCols;
                    if (size === undefined) {
                        outputRows = 2;
                        outputCols = 2;
                    }
                    else {
                        outputRows = size[0];
                        outputCols = size[1];
                    }
                    var expectedShape;
                    if (undeterminedDimension == null) {
                        if (dataFormat === 'channelsFirst') {
                            outputRows *= inputShape[2];
                            outputCols *= inputShape[3];
                            expectedShape = [2, 16, outputRows, outputCols];
                        }
                        else {
                            outputRows *= inputShape[1];
                            outputCols *= inputShape[2];
                            expectedShape = [2, outputRows, outputCols, 16];
                        }
                    }
                    else if (undeterminedDimension === 'height') {
                        if (dataFormat === 'channelsFirst') {
                            outputCols *= inputShape[3];
                            expectedShape = [2, 16, null, outputCols];
                        }
                        else {
                            outputCols *= inputShape[2];
                            expectedShape = [2, null, outputCols, 16];
                        }
                    }
                    else if (undeterminedDimension === 'both') {
                        if (dataFormat === 'channelsFirst') {
                            expectedShape = [2, 16, null, null];
                        }
                        else {
                            outputCols *= inputShape[2];
                            expectedShape = [2, null, null, 16];
                        }
                    }
                    expect(output.shape).toEqual(expectedShape);
                });
            };
            for (var _i = 0, undeterminedDimensionArray_1 = undeterminedDimensionArray; _i < undeterminedDimensionArray_1.length; _i++) {
                var undeterminedDimension = undeterminedDimensionArray_1[_i];
                _loop_44(undeterminedDimension);
            }
        };
        for (var _i = 0, sizes_1 = sizes; _i < sizes_1.length; _i++) {
            var size = sizes_1[_i];
            _loop_43(size);
        }
    };
    for (var _i = 0, dataFormats_8 = dataFormats; _i < dataFormats_8.length; _i++) {
        var dataFormat = dataFormats_8[_i];
        _loop_42(dataFormat);
    }
});
describe('UpSampling2D Layer', function () {
    it('check with default values', function () {
        var layer = tfl.layers.upSampling2d({});
        var x = tfjs_core_1.tensor4d([
            [[[1], [2]], [[3], [4]]],
        ], [1, 2, 2, 1]);
        var y = tfjs_core_1.tensor4d([
            [
                [[1], [1], [2], [2]], [[1], [1], [2], [2]], [[3], [3], [4], [4]],
                [[3], [3], [4], [4]]
            ],
        ], [1, 4, 4, 1]);
        test_utils_1.expectTensorsClose(layer.apply(x, null), y);
    });
    it('channels last', function () {
        var layer = tfl.layers.upSampling2d({ size: [2, 2], dataFormat: 'channelsLast' });
        var x = tfjs_core_1.tensor4d([
            [[[1], [2]], [[3], [4]]],
        ], [1, 2, 2, 1]);
        var y = tfjs_core_1.tensor4d([
            [
                [[1], [1], [2], [2]], [[1], [1], [2], [2]], [[3], [3], [4], [4]],
                [[3], [3], [4], [4]]
            ],
        ], [1, 4, 4, 1]);
        test_utils_1.expectTensorsClose(layer.apply(x, null), y);
    });
    it('channels first', function () {
        var layer = tfl.layers.upSampling2d({ size: [2, 2], dataFormat: 'channelsFirst' });
        var x = tfjs_core_1.tensor4d([
            [[[1, 2], [3, 4]]],
        ], [1, 1, 2, 2]);
        var y = tfjs_core_1.tensor4d([
            [[[1, 1, 2, 2], [1, 1, 2, 2], [3, 3, 4, 4], [3, 3, 4, 4]]],
        ], [1, 1, 4, 4]);
        test_utils_1.expectTensorsClose(layer.apply(x, null), y);
    });
    it('varying input image sizes', function () {
        var layer = tfl.layers.upSampling2d({ size: [2, 2], dataFormat: 'channelsLast' });
        var x1 = tfjs_core_1.tensor4d([
            [[[1], [2]], [[3], [4]]],
        ], [1, 2, 2, 1]);
        var y1 = tfjs_core_1.tensor4d([
            [
                [[1], [1], [2], [2]], [[1], [1], [2], [2]], [[3], [3], [4], [4]],
                [[3], [3], [4], [4]]
            ],
        ], [1, 4, 4, 1]);
        test_utils_1.expectTensorsClose(layer.apply(x1, null), y1);
        var x2 = tfjs_core_1.tensor4d([
            [[[1], [2]]],
        ], [1, 1, 2, 1]);
        var y2 = tfjs_core_1.tensor4d([
            [
                [[1], [1], [2], [2]],
                [[1], [1], [2], [2]],
            ],
        ], [1, 2, 4, 1]);
        test_utils_1.expectTensorsClose(layer.apply(x2, null), y2);
    });
});
//# sourceMappingURL=convolutional_test.js.map