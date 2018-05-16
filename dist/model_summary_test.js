"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
var tfl = require("./index");
var test_utils_1 = require("./utils/test_utils");
function getRandomLayerOrModelName(length) {
    if (length === void 0) { length = 12; }
    return 'L' + Math.random().toFixed(length - 1).slice(2);
}
test_utils_1.describeMathCPU('Model.summary', function () {
    var consoleLogHistory;
    beforeEach(function () {
        consoleLogHistory = [];
        spyOn(console, 'log').and.callFake(function (message) {
            consoleLogHistory.push(message);
        });
    });
    afterEach(function () {
        consoleLogHistory = [];
    });
    it('Sequential model: one layer', function () {
        var layerName = getRandomLayerOrModelName();
        var model = tfl.sequential({
            layers: [tfl.layers.dense({ units: 3, inputShape: [10], name: layerName })]
        });
        model.summary();
        expect(consoleLogHistory).toEqual([
            '_________________________________________________________________',
            'Layer (type)                 Output shape              Param #   ',
            '=================================================================',
            layerName + " (Dense)         [null,3]                  33        ",
            '=================================================================',
            'Total params: 33', 'Trainable params: 33', 'Non-trainable params: 0',
            '_________________________________________________________________'
        ]);
    });
    it('Sequential model: one layer: custom lineLength', function () {
        var layerName = getRandomLayerOrModelName();
        var model = tfl.sequential({
            layers: [tfl.layers.dense({ units: 3, inputShape: [10], name: layerName })]
        });
        var lineLength = 70;
        model.summary(lineLength);
        expect(consoleLogHistory).toEqual([
            '______________________________________________________________________',
            'Layer (type)                   Output shape                Param #    ',
            '======================================================================',
            layerName + " (Dense)           [null,3]                    33         ",
            '======================================================================',
            'Total params: 33', 'Trainable params: 33', 'Non-trainable params: 0',
            '______________________________________________________________________'
        ]);
    });
    it('Sequential model: one layer: custom positions', function () {
        var layerName = getRandomLayerOrModelName();
        var model = tfl.sequential({
            layers: [tfl.layers.dense({ units: 3, inputShape: [10], name: layerName })]
        });
        var lineLength = 70;
        var positions = [0.5, 0.8, 1.0];
        model.summary(lineLength, positions);
        expect(consoleLogHistory).toEqual([
            '______________________________________________________________________',
            'Layer (type)                       Output shape         Param #       ',
            '======================================================================',
            layerName + " (Dense)               [null,3]             33            ",
            '======================================================================',
            'Total params: 33', 'Trainable params: 33', 'Non-trainable params: 0',
            '______________________________________________________________________'
        ]);
    });
    it('Sequential model: one layer: custom printFn', function () {
        var layerName = getRandomLayerOrModelName();
        var model = tfl.sequential({
            layers: [tfl.layers.dense({ units: 3, inputShape: [10], name: layerName })]
        });
        var messages = [];
        function rerouteLog(message) {
            var optionalParams = [];
            for (var _i = 1; _i < arguments.length; _i++) {
                optionalParams[_i - 1] = arguments[_i];
            }
            messages.push(message);
        }
        model.summary(null, null, rerouteLog);
        expect(messages).toEqual([
            '_________________________________________________________________',
            'Layer (type)                 Output shape              Param #   ',
            '=================================================================',
            layerName + " (Dense)         [null,3]                  33        ",
            '=================================================================',
            'Total params: 33', 'Trainable params: 33', 'Non-trainable params: 0',
            '_________________________________________________________________'
        ]);
        expect(consoleLogHistory).toEqual([]);
    });
    it('Sequential model: three layers', function () {
        var lyrName01 = getRandomLayerOrModelName();
        var lyrName02 = getRandomLayerOrModelName();
        var lyrName03 = getRandomLayerOrModelName();
        var model = tfl.sequential({
            layers: [
                tfl.layers.flatten({ inputShape: [2, 5], name: lyrName01 }),
                tfl.layers.dense({ units: 3, name: lyrName02 }),
                tfl.layers.dense({ units: 1, name: lyrName03 }),
            ]
        });
        model.summary();
        expect(consoleLogHistory).toEqual([
            '_________________________________________________________________',
            'Layer (type)                 Output shape              Param #   ',
            '=================================================================',
            lyrName01 + " (Flatten)       [null,10]                 0         ",
            '_________________________________________________________________',
            lyrName02 + " (Dense)         [null,3]                  33        ",
            '_________________________________________________________________',
            lyrName03 + " (Dense)         [null,1]                  4         ",
            '=================================================================',
            'Total params: 37',
            'Trainable params: 37',
            'Non-trainable params: 0',
            '_________________________________________________________________',
        ]);
    });
    it('Sequential model: with non-trainable layers', function () {
        var lyrName01 = getRandomLayerOrModelName();
        var lyrName02 = getRandomLayerOrModelName();
        var lyrName03 = getRandomLayerOrModelName();
        var model = tfl.sequential({
            layers: [
                tfl.layers.flatten({ inputShape: [2, 5], name: lyrName01 }),
                tfl.layers.dense({ units: 3, name: lyrName02, trainable: false }),
                tfl.layers.dense({ units: 1, name: lyrName03 }),
            ]
        });
        model.summary();
        expect(consoleLogHistory).toEqual([
            '_________________________________________________________________',
            'Layer (type)                 Output shape              Param #   ',
            '=================================================================',
            lyrName01 + " (Flatten)       [null,10]                 0         ",
            '_________________________________________________________________',
            lyrName02 + " (Dense)         [null,3]                  33        ",
            '_________________________________________________________________',
            lyrName03 + " (Dense)         [null,1]                  4         ",
            '=================================================================',
            'Total params: 37',
            'Trainable params: 4',
            'Non-trainable params: 33',
            '_________________________________________________________________',
        ]);
    });
    it('Sequential model with Embedding layer', function () {
        var lyrName01 = getRandomLayerOrModelName();
        var lyrName02 = getRandomLayerOrModelName();
        var model = tfl.sequential({
            layers: [
                tfl.layers.embedding({
                    inputDim: 10,
                    outputDim: 8,
                    inputShape: [null, 5],
                    name: lyrName01
                }),
                tfl.layers.dense({ units: 3, name: lyrName02 }),
            ]
        });
        model.summary();
        expect(consoleLogHistory).toEqual([
            '_________________________________________________________________',
            'Layer (type)                 Output shape              Param #   ',
            '=================================================================',
            lyrName01 + " (Embedding)     [null,null,5,8]           80        ",
            '_________________________________________________________________',
            lyrName02 + " (Dense)         [null,null,5,3]           27        ",
            '=================================================================',
            'Total params: 107', 'Trainable params: 107', 'Non-trainable params: 0',
            '_________________________________________________________________'
        ]);
    });
    it('Sequential model: nested', function () {
        var mdlName01 = getRandomLayerOrModelName();
        var innerModel = tfl.sequential({
            layers: [tfl.layers.dense({ units: 3, inputShape: [10] })],
            name: mdlName01
        });
        var outerModel = tfl.sequential();
        outerModel.add(innerModel);
        var lyrName02 = getRandomLayerOrModelName();
        outerModel.add(tfl.layers.dense({ units: 1, name: lyrName02 }));
        outerModel.summary();
        expect(consoleLogHistory).toEqual([
            '_________________________________________________________________',
            'Layer (type)                 Output shape              Param #   ',
            '=================================================================',
            mdlName01 + " (Sequential)    [null,3]                  33        ",
            '_________________________________________________________________',
            lyrName02 + " (Dense)         [null,1]                  4         ",
            '=================================================================',
            'Total params: 37',
            'Trainable params: 37',
            'Non-trainable params: 0',
            '_________________________________________________________________',
        ]);
    });
    it('Functional model', function () {
        var lyrName01 = getRandomLayerOrModelName();
        var input1 = tfl.input({ shape: [3], name: lyrName01 });
        var lyrName02 = getRandomLayerOrModelName();
        var input2 = tfl.input({ shape: [4], name: lyrName02 });
        var lyrName03 = getRandomLayerOrModelName();
        var input3 = tfl.input({ shape: [5], name: lyrName03 });
        var lyrName04 = getRandomLayerOrModelName();
        var concat1 = tfl.layers.concatenate({ name: lyrName04 }).apply([input1, input2]);
        var lyrName05 = getRandomLayerOrModelName();
        var output = tfl.layers.concatenate({ name: lyrName05 }).apply([concat1, input3]);
        var model = tfl.model({ inputs: [input1, input2, input3], outputs: output });
        var lineLength = 70;
        var positions = [0.42, 0.64, 0.75, 1];
        model.summary(lineLength, positions);
        expect(consoleLogHistory).toEqual([
            '______________________________________________________________________',
            'Layer (type)                 Output shape   Param # Receives inputs   ',
            '======================================================================',
            lyrName01 + " (InputLayer)    [null,3]       0                         ",
            '______________________________________________________________________',
            lyrName02 + " (InputLayer)    [null,4]       0                         ",
            '______________________________________________________________________',
            lyrName04 + " (Concatenate)   [null,7]       0       " + lyrName01 + "[0][0]",
            "                                                    " + lyrName02 + "[0][0]",
            '______________________________________________________________________',
            lyrName03 + " (InputLayer)    [null,5]       0                         ",
            '______________________________________________________________________',
            lyrName05 + " (Concatenate)   [null,12]      0       " + lyrName04 + "[0][0]",
            "                                                    " + lyrName03 + "[0][0]",
            '======================================================================',
            'Total params: 0', 'Trainable params: 0', 'Non-trainable params: 0',
            '______________________________________________________________________'
        ]);
    });
    it('Model with multiple outputs', function () {
        var lyrName01 = getRandomLayerOrModelName();
        var input1 = tfl.input({ shape: [3, 4], name: lyrName01 });
        var lyrName02 = getRandomLayerOrModelName();
        var outputs = tfl.layers.simpleRNN({ units: 2, returnState: true, name: lyrName02 })
            .apply(input1);
        var model = tfl.model({ inputs: input1, outputs: outputs });
        var lineLength = 70;
        model.summary(lineLength);
        expect(consoleLogHistory).toEqual([
            '______________________________________________________________________',
            'Layer (type)                   Output shape                Param #    ',
            '======================================================================',
            lyrName01 + " (InputLayer)      [null,3,4]                  0          ",
            '______________________________________________________________________',
            lyrName02 + " (SimpleRNN)       [[null,2],[null,2]]         14         ",
            '======================================================================',
            'Total params: 14', 'Trainable params: 14', 'Non-trainable params: 0',
            '______________________________________________________________________'
        ]);
    });
});
//# sourceMappingURL=model_summary_test.js.map