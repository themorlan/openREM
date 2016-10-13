/**
 * Function to create a 2D skin dose map object
 * @param skinDoseMapCanvasName
 * @param colourScaleName
 */
function skinDoseMapObject(skinDoseMapCanvasName, colourScaleName) {

    this.useNewColourScale = useNewColourScale;
    /**
     * Internal function to create a new colour scale
     * @param new_scale
     */
    function useNewColourScale(new_scale) {
        var _this = this;
        _this.colourScale = chroma.scale(new_scale);
    }


    this.setPixel = setPixel;
    /**
     * Internal function to set the rgba value of a pixel
     * @param imageData
     * @param x
     * @param y
     * @param r
     * @param g
     * @param b
     * @param a
     */
    function setPixel(imageData, x, y, r, g, b, a) {
        var index = (x + y * imageData.width) * 4;
        imageData.data[index    ] = r;
        imageData.data[index + 1] = g;
        imageData.data[index + 2] = b;
        imageData.data[index + 3] = a;
    }


    this.draw = draw;
    /**
     * Internal function to draw the skin dose map
     */
    function draw() {
        var _this = this;
        var x, y, dose, scaledDose;
        for (x = 0; x < _this.skinDoseMapWidth; x++) {
            for (y = 0; y < _this.skinDoseMapHeight; y++) {
                dose = _this.skinDoseMap[(y) * _this.skinDoseMapWidth + x];
                scaledDose = dose - (_this.windowLevel - (_this.windowWidth / 2.0));
                if (scaledDose < 0) scaledDose = 0;
                if (scaledDose > _this.windowWidth) scaledDose = _this.windowWidth;
                _this.skinDoseMapContext.fillStyle = _this.colourScale(scaledDose / _this.windowWidth).hex();
                _this.skinDoseMapContext.fillRect(x*_this.mag, y*_this.mag, _this.mag, _this.mag);
            }
        }
    }


    this.drawOverlay = drawOverlay;
    /**
     * Internal function to draw the overlay on the skin dose map
     */
    function drawOverlay() {
        var _this = this;
        _this.skinDoseMapContext.textAlign = 'center';
        _this.skinDoseMapContext.font = '12pt arial';

        _this.skinDoseMapContext.fillStyle = 'rgba(0, 80, 0, 0.85)';
        _this.skinDoseMapContext.fillText('Superior', _this.skinDoseMapCanvas.width/2, 15);
        _this.skinDoseMapContext.fillText('Inferior', _this.skinDoseMapCanvas.width/2, _this.skinDoseMapCanvas.height-10);

        _this.skinDoseMapContext.save();

        _this.skinDoseMapContext.rotate(0.5*Math.PI);
        _this.skinDoseMapContext.fillStyle = 'rgba(255, 0, 0, 0.85)';
        _this.skinDoseMapContext.fillText('Anterior', _this.skinDoseMapCanvas.height/2, -_this.frontLeftBoundary/2);
        _this.skinDoseMapContext.fillText('Posterior', _this.skinDoseMapCanvas.height/2, -_this.leftBackBoundary - (_this.backRightBoundary-_this.leftBackBoundary)/2);
        _this.skinDoseMapContext.fillText('Left', _this.skinDoseMapCanvas.height/2, -_this.frontLeftBoundary - (_this.leftBackBoundary-_this.frontLeftBoundary)/2);
        _this.skinDoseMapContext.fillText('Right', _this.skinDoseMapCanvas.height/2, -_this.rightFrontBoundary + (_this.rightFrontBoundary-_this.backRightBoundary)/2);

        _this.skinDoseMapContext.restore();

        _this.skinDoseMapContext.lineWidth = 1;
        _this.skinDoseMapContext.setLineDash([5, 15]);
        _this.skinDoseMapContext.strokeStyle = 'rgba(255, 0, 0, 0.25)';

        _this.skinDoseMapContext.beginPath();
        _this.skinDoseMapContext.moveTo(_this.frontLeftBoundary, 0);
        _this.skinDoseMapContext.lineTo(_this.frontLeftBoundary, _this.skinDoseMapCanvas.height-1);
        _this.skinDoseMapContext.stroke();

        _this.skinDoseMapContext.beginPath();
        _this.skinDoseMapContext.moveTo(_this.leftBackBoundary, 0);
        _this.skinDoseMapContext.lineTo(_this.leftBackBoundary, _this.skinDoseMapCanvas.height-1);
        _this.skinDoseMapContext.stroke();

        _this.skinDoseMapContext.beginPath();
        _this.skinDoseMapContext.moveTo(_this.backRightBoundary, 0);
        _this.skinDoseMapContext.lineTo(_this.backRightBoundary, _this.skinDoseMapCanvas.height-1);
        _this.skinDoseMapContext.stroke();
    }


    this.resizeSkinDoseMap = resizeSkinDoseMap;
    /**
     * Internal function to resize the skin dose map
     */
    function resizeSkinDoseMap() {
        var _this = this;
        _this.skinDoseMapCanvas.width = _this.skinDoseMapWidth * _this.mag;
        _this.skinDoseMapCanvas.height = _this.skinDoseMapHeight * _this.mag;
    }


    this.reset = reset;
    /**
     * Internal function to reset the skin dose map
     */
    function reset() {
        var _this = this;
        _this.updateWindowWidth(_this.maxDose - _this.minDose);
        _this.updateWindowLevel(_this.minDose + (_this.windowWidth / 2.0));
    }


    this.toggleOverlay = toggleOverlay;
    /**
     * Internal function to toggle the display of the overlay
     */
    function toggleOverlay() {
        var _this = this;
        _this.showOverlay = _this.showOverlay ? _this.showOverlay = false : _this.showOverlay = true;

        if (_this.showOverlay) {
            _this.drawOverlay();
        } else {
            _this.draw();
        }
    }


    this.updateWindowLevel = updateWindowLevel;
    /**
     * Internal function to update the window level
     * @param newWindowLevel
     */
    function updateWindowLevel(newWindowLevel) {
        var _this = this;
        if (newWindowLevel < 0) newWindowLevel = 0;
        _this.windowLevel = parseFloat(newWindowLevel);

        _this.minDisplayedDose = _this.windowLevel - (_this.windowWidth / 2.0);
        _this.maxDisplayedDose = _this.windowLevel + (_this.windowWidth / 2.0);
    }


    this.updateWindowWidth = updateWindowWidth;
    /**
     * Internal function to update the window width
     * @param newWindowWidth
     */
    function updateWindowWidth(newWindowWidth) {
        var _this = this;
        _this.windowWidth = newWindowWidth;

        _this.minDisplayedDose = _this.windowLevel - (_this.windowWidth / 2.0);
        _this.maxDisplayedDose = _this.windowLevel + (_this.windowWidth / 2.0);
    }


    this.updateMinDisplayedDose = updateMinDisplayedDose;
    /**
     * Internal function to update the minimum displayed dose
     * @param minDisplayedDose
     */
    function updateMinDisplayedDose(minDisplayedDose) {
        var _this = this;
        minDisplayedDose = parseFloat(minDisplayedDose);
        
        if (minDisplayedDose <= _this.minDose) {
            minDisplayedDose = _this.minDose;
        }
        else if (minDisplayedDose >= _this.maxDose) {
            minDisplayedDose = _this.maxDose;
        }

        _this.minDisplayedDose = minDisplayedDose;

        // Prevent the minDisplatedDose exceeding the maxDisplayedDose
        if (minDisplayedDose >= _this.maxDisplayedDose) {
            _this.maxDisplayedDose = minDisplayedDose;
        }

        _this.windowWidth = _this.maxDisplayedDose - _this.minDisplayedDose;
        _this.windowLevel = _this.minDisplayedDose + (_this.windowWidth / 2.0);
    }


    this.updateMaxDisplayedDose = updateMaxDisplayedDose;
    /**
     * Internal function to update the maximum displayed dose
     * @param maxDisplayedDose
     */
    function updateMaxDisplayedDose(maxDisplayedDose) {
        var _this = this;
        maxDisplayedDose = parseFloat(maxDisplayedDose);

        if (maxDisplayedDose <= _this.minDose) {
            maxDisplayedDose = _this.minDose;
        }
        else if (maxDisplayedDose >= _this.maxDose) {
            maxDisplayedDose = _this.maxDose;
        }

        _this.maxDisplayedDose = maxDisplayedDose;

        // Prevent the maxDisplatedDose being smaller than the minDisplayedDose
        if (maxDisplayedDose <= _this.minDisplayedDose) {
            _this.minDisplayedDose = maxDisplayedDose;
        }

        _this.windowWidth = _this.maxDisplayedDose - _this.minDisplayedDose;
        _this.windowLevel = _this.minDisplayedDose + (_this.windowWidth / 2.0);
    }


    this.updateMinDisplayedDoseManual = updateMinDisplayedDoseManual;
    /**
     * Internal function to update the minimum displayed dose
     * @param minDisplayedDose
     */
    function updateMinDisplayedDoseManual(minDisplayedDose) {
        var _this = this;
        minDisplayedDose = parseFloat(minDisplayedDose);

        _this.minDisplayedDose = minDisplayedDose;

        // Prevent the minDisplatedDose exceeding the maxDisplayedDose
        if (minDisplayedDose >= _this.maxDisplayedDose) {
            _this.maxDisplayedDose = minDisplayedDose;
        }

        _this.windowWidth = _this.maxDisplayedDose - _this.minDisplayedDose;
        _this.windowLevel = _this.minDisplayedDose + (_this.windowWidth / 2.0);
    }


    this.updateMaxDisplayedDoseManual = updateMaxDisplayedDoseManual;
    /**
     * Internal function to update the maximum displayed dose
     * @param maxDisplayedDose
     */
    function updateMaxDisplayedDoseManual(maxDisplayedDose) {
        var _this = this;
        maxDisplayedDose = parseFloat(maxDisplayedDose);

        _this.maxDisplayedDose = maxDisplayedDose;

        // Prevent the maxDisplatedDose being smaller than the minDisplayedDose
        if (maxDisplayedDose <= _this.minDisplayedDose) {
            _this.minDisplayedDose = maxDisplayedDose;
        }

        _this.windowWidth = _this.maxDisplayedDose - _this.minDisplayedDose;
        _this.windowLevel = _this.minDisplayedDose + (_this.windowWidth / 2.0);
    }


    this.initialise = initialise;
    /**
     * Internal function to initialise the skin dose map
     * @param skinMapData
     * @param skinMapWidth
     * @param skinMapHeight
     * @param phantomFlatWidth
     * @param phantomCurvedEdgeWidth
     */
    function initialise(skinMapData, skinMapWidth, skinMapHeight, phantomFlatWidth, phantomCurvedEdgeWidth) {
        var _this = this;
        _this.skinDoseMap = skinMapData;
        _this.skinDoseMapWidth = skinMapWidth;
        _this.skinDoseMapHeight = skinMapHeight;
        _this.minDose = Math.min.apply(null, _this.skinDoseMap);
        _this.maxDose = Math.max.apply(null, _this.skinDoseMap);
        _this.windowWidth = _this.maxDose - _this.minDose;
        _this.windowLevel = _this.minDose + (_this.windowWidth / 2.0);
        _this.minDisplayedDose = _this.minDose;
        _this.maxDisplayedDose = _this.maxDose;

        _this.phantomFlatWidth = phantomFlatWidth;
        _this.phantomCurvedEdgeWidth = phantomCurvedEdgeWidth;

        _this.resizeSkinDoseMap();
        _this.updateBoundaries();
    }


    this.updateBoundaries = updateBoundaries;
    /**
     * Internal function to update the boundaries of the phantom sections
     */
    function updateBoundaries () {
        var _this = this;
        _this.frontLeftBoundary = _this.phantomFlatWidth * _this.mag;
        _this.leftBackBoundary = _this.frontLeftBoundary + (_this.phantomCurvedEdgeWidth * _this.mag);
        _this.backRightBoundary = _this.leftBackBoundary + (_this.phantomFlatWidth * _this.mag);
        _this.rightFrontBoundary = _this.backRightBoundary + (_this.phantomCurvedEdgeWidth * _this.mag);
    }


    this.skinDoseMapCanvasName = skinDoseMapCanvasName;
    this.skinDoseMapCanvas = document.getElementById(this.skinDoseMapCanvasName);
    this.skinDoseMapContext = this.skinDoseMapCanvas.getContext('2d');

    this.colourScaleName = colourScaleName;
    this.colourScale = chroma.scale(colourScaleName);

    this.mag = 6;

    this.skinDoseMap = [];
    this.skinDoseMapWidth = 10;
    this.skinDoseMapHeight = 10;

    this.phantomFlatWidth = 14;
    this.phantomCurvedEdgeWidth = 31;

    this.showOverlay = false;

    this.frontLeftBoundary = this.phantomFlatWidth * this.mag;
    this.leftBackBoundary = this.frontLeftBoundary + (this.phantomCurvedEdgeWidth * this.mag);
    this.backRightBoundary = this.leftBackBoundary + (this.phantomFlatWidth * this.mag);
    this.rightFrontBoundary = this.backRightBoundary + (this.phantomCurvedEdgeWidth * this.mag);

    this.minDose = 0.0;
    this.maxDose = 10.0;

    this.windowWidth = this.maxDose - this.minDose;
    this.windowLevel = this.minDose + (this.windowWidth / 2.0);
    this.minDisplayedDose = this.minDose;
    this.maxDisplayedDose = this.maxDose;
}