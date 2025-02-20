/*global chroma, THREE, animate, render*/
/*eslint no-undef: "error"*/
/*eslint security/detect-object-injection: "off" */

/**
 * Function to create a 3d skin dose map object
 * @param skinDoseMap3dCanvasName
 * @param colourScaleName
 */
function skinDoseMap3dObject(skinDoseMap3dCanvasName, colourScaleName) {

    this.canvasName = skinDoseMap3dCanvasName;
    this.canvas = document.getElementById(this.canvasName);

    this.colourScale = chroma.scale(colourScaleName);

    this.windowWidth = 10.0;
    this.windowLevel = 5.0;

    this.skinDoseMap = [];

    this.phantomHeight = 10;
    this.phantomFlatWidth = 10;
    this.phantomCurvedEdgeWidth = 10;
    this.phantomCurvedRadius = 10;
    this.phantomHeadHeight = 10;

    this.skinDoseMapWidth = 10;
    this.skinDoseMapHeight = 10;

    var frontData = [];
    var backData = [];
    var leftData = [];
    var rightData = [];
    var headData = [];

    this.camera = 0;
    this.scene = 0;
    this.mesh = 0;

    var dataTextureFront = 0;
    var dataTextureBack = 0;
    var dataTextureLeft = 0;
    var dataTextureRight = 0;
    var dataTextureHead = 0;

    var materialFront = 0;
    var materialBack = 0;
    var materialLeft = 0;
    var materialRight = 0;
    var materialHead = 0;


    /**
     * Internal function to create a new colour scale
     * @param newScale
     */
    this.useNewColourScale = function (newScale) {
        var _this = this;
        _this.colourScale = chroma.scale(newScale);
    };


    /**
     * Internal function to return a colour for a skin dose map patch given
     * the i, j location in the skin dose map array
     */
    this.getNewColour = function(i, j) {
        var _this = this;
        var currentDose = _this.skinDoseMap[j * (_this.phantomHeight +_this.phantomHeadHeight) + i];
        var scaledDose = currentDose - (_this.windowLevel - (_this.windowWidth / 2.0));
        if (scaledDose < 0) {scaledDose = 0;}
        if (scaledDose > _this.windowWidth) {scaledDose = _this.windowWidth;}
        return _this.colourScale(scaledDose / _this.windowWidth).rgb();
    };


    /**
     * Internal function to draw the 3d skin dose map
     */
     this.draw = function () {
        var _this = this;
        var newColour, i, j, k, l, m, n, o;
        var torsoStart = _this.phantomHeadHeight;
        var torsoEnd = _this.phantomHeadHeight + _this.phantomHeight;
        var phantomHeadCircumference = 2 * Math.PI * _this.phantomHeadRadius;

        // Draw the torso values
        k = l = m = n = 0;
        for (i = torsoStart; i < torsoEnd; i++) {
            for (j = 0; j < _this.phantomFlatWidth; j++) {
                newColour = this.getNewColour(i, j);
                dataTextureFront.image.data[k] = newColour[0];
                dataTextureFront.image.data[k+1] = newColour[1];
                dataTextureFront.image.data[k+2] = newColour[2];
                dataTextureFront.image.data[k+3] = 0;
                k += 4;
            }

            for (j = _this.phantomFlatWidth; j < _this.phantomFlatWidth+_this.phantomCurvedEdgeWidth; j++) {
                newColour = this.getNewColour(i, j);
                dataTextureLeft.image.data[l] = newColour[0];
                dataTextureLeft.image.data[l+1] = newColour[1];
                dataTextureLeft.image.data[l+2] = newColour[2];
                dataTextureLeft.image.data[l+3] = 0;
                l += 4;
            }

            for (j = _this.phantomFlatWidth+_this.phantomCurvedEdgeWidth; j < _this.phantomFlatWidth*2+_this.phantomCurvedEdgeWidth; j++) {
                newColour = this.getNewColour(i, j);
                dataTextureBack.image.data[m] = newColour[0];
                dataTextureBack.image.data[m+1] = newColour[1];
                dataTextureBack.image.data[m+1] = newColour[1];
                dataTextureBack.image.data[m+2] = newColour[2];
                dataTextureBack.image.data[m+3] = 0;
                m += 4;
            }

            for (j = _this.phantomFlatWidth*2+_this.phantomCurvedEdgeWidth; j < _this.phantomFlatWidth*2+_this.phantomCurvedEdgeWidth*2; j++) {
                newColour = this.getNewColour(i, j);
                dataTextureRight.image.data[n] = newColour[0];
                dataTextureRight.image.data[n+1] = newColour[1];
                dataTextureRight.image.data[n+2] = newColour[2];
                dataTextureRight.image.data[n+3] = 0;
                n += 4;
            }
        }

        // Draw the head values
        o = 0;
        for (i = 0; i < torsoStart; i++) {
            for (j = 0; j < phantomHeadCircumference; j++) {
                newColour = this.getNewColour(i, j);
                dataTextureHead.image.data[o] = newColour[0];
                dataTextureHead.image.data[o+1] = newColour[1];
                dataTextureHead.image.data[o+2] = newColour[2];
                dataTextureHead.image.data[o+3] = 0;
                o += 4;
            }
        }

        dataTextureFront.needsUpdate = true;
        dataTextureBack.needsUpdate  = true;
        dataTextureLeft.needsUpdate  = true;
        dataTextureRight.needsUpdate = true;
        dataTextureHead.needsUpdate = true;
    };


    /**
     * Internal function to merge three.js meshes
     * @param meshes
     * @returns {THREE.Geometry}
     */
    this.mergeMeshes = function (meshes) {
        var combined = new THREE.Geometry();

        var lastFace = 0, j;

        for (var i = 0; i < meshes.length; i++) {
            meshes[i].updateMatrix();
            combined.merge(meshes[i].geometry, meshes[i].matrix);
            for(j = lastFace; j < combined.faces.length; j++) {
                combined.faces[j].materialIndex = i;
            }
            lastFace = combined.faces.length;
        }

        return combined;
    };


    /**
     * Internal function to initialse the 3d skin dose map
     * @param skinDoseMap
     * @param phantomFlatWidth
     * @param phantomCurvedEdgeWidth
     * @param phantomHeight
     * @param phantomCurvedRadius
     */
    this.initialise = function (skinDoseMap, phantomFlatWidth, phantomCurvedEdgeWidth, phantomHeight, phantomCurvedRadius,
    phantomHeadHeight,phantomHeadRadius) {
        var _this = this;

        _this.skinDoseMap = new Array(skinDoseMap.length);
        _this.skinDoseMapWidth = 2*phantomFlatWidth + 2*phantomCurvedEdgeWidth;
        _this.skinDoseMapHeight = phantomHeight + phantomHeadHeight;
        _this.phantomFlatWidth = phantomFlatWidth;
        _this.phantomCurvedEdgeWidth = phantomCurvedEdgeWidth;
        _this.phantomHeight = phantomHeight;
        _this.phantomHeadHeight = phantomHeadHeight;
        _this.phantomHeadRadius = phantomHeadRadius;
        _this.phantomCurvedRadius = phantomCurvedRadius;

        var x, y, offset;
        for (x = 0; x < _this.skinDoseMapWidth; x++) {
            for (y = 0; y < _this.skinDoseMapHeight; y++) {
                offset = _this.skinDoseMapHeight * x + y;
                _this.skinDoseMap[offset] = skinDoseMap[_this.skinDoseMapWidth * (y) + x];
            }
        }

        frontData = new Uint8Array(_this.phantomFlatWidth*_this.phantomHeight*4);
        backData  = new Uint8Array(_this.phantomFlatWidth*_this.phantomHeight*4);
        leftData  = new Uint8Array(_this.phantomCurvedEdgeWidth*_this.phantomHeight*4);
        rightData = new Uint8Array(_this.phantomCurvedEdgeWidth*_this.phantomHeight*4);
        headData = new Uint8Array(2 * Math.PI * _this.phantomHeadRadius*_this.phantomHeadHeight*4);

        dataTextureFront = new THREE.DataTexture( frontData, _this.phantomFlatWidth, _this.phantomHeight,  THREE.RGBAFormat );
        dataTextureBack  = new THREE.DataTexture( backData,  _this.phantomFlatWidth, _this.phantomHeight,  THREE.RGBAFormat );
        dataTextureLeft  = new THREE.DataTexture( leftData,  _this.phantomCurvedEdgeWidth, _this.phantomHeight, THREE.RGBAFormat );
        dataTextureRight = new THREE.DataTexture( rightData, _this.phantomCurvedEdgeWidth, _this.phantomHeight, THREE.RGBAFormat );
        dataTextureHead = new THREE.DataTexture(headData, 2 * Math.PI * _this.phantomHeadRadius, _this.phantomHeadHeight, THREE.RGBAFormat);

        materialFront = new THREE.MeshLambertMaterial( { map: dataTextureFront } );
        materialBack  = new THREE.MeshLambertMaterial( { map: dataTextureBack  } );
        materialLeft  = new THREE.MeshLambertMaterial( { map: dataTextureLeft  } );
        materialRight = new THREE.MeshLambertMaterial( { map: dataTextureRight } );
        materialHead  = new THREE.MeshLambertMaterial( { map: dataTextureHead } );
        materialFront.map.minFilter = THREE.NearestFilter;
        materialBack.map.minFilter = THREE.NearestFilter;
        materialLeft.map.minFilter = THREE.NearestFilter;
        materialRight.map.minFilter = THREE.NearestFilter;
        materialHead.map.minFilter = THREE.NearestFilter;
        materialFront.map.magFilter = THREE.NearestFilter;
        materialBack.map.magFilter = THREE.NearestFilter;
        materialLeft.map.magFilter = THREE.NearestFilter;
        materialRight.map.magFilter = THREE.NearestFilter;
        materialHead.map.magFilter = THREE.NearestFilter;

        var aspectRatio = _this.canvas.width / _this.canvas.height;

        _this.scene = new THREE.Scene();

        // Set up the camera
        _this.camera = new THREE.PerspectiveCamera(50, aspectRatio, 1, 10000);
        _this.camera.position.x = _this.phantomHeight + _this.phantomHeadHeight;
        _this.camera.position.y = 0;
        _this.camera.position.z = 100;
        _this.camera.lookAt(_this.scene.position );
        _this.scene.add(_this.camera);

        var geometry;
        _this.meshes = [];

        THREE.ImageUtils.crossOrigin = "anonymous";

        var endMaterial = new THREE.MeshLambertMaterial( { color: 0x7092be } );
        var materials = [materialBack, materialLeft, materialFront, materialRight, endMaterial, endMaterial, endMaterial,
            endMaterial, endMaterial, endMaterial, materialHead, endMaterial];
        var meshFaceMaterial = new THREE.MeshFaceMaterial(materials);

        geometry = new THREE.PlaneGeometry(_this.phantomFlatWidth, _this.phantomHeight);
        var mesh = new THREE.Mesh(geometry);
        mesh.rotation.y = Math.PI;
        mesh.position.z = -_this.phantomCurvedRadius;
        _this.meshes.push(mesh);

        geometry = new THREE.CylinderGeometry(_this.phantomCurvedRadius, _this.phantomCurvedRadius, _this.phantomHeight, 32, 1, true, 0, Math.PI );
        mesh = new THREE.Mesh(geometry);
        mesh.position.x = _this.phantomFlatWidth / 2;
        _this.meshes.push(mesh);

        geometry = new THREE.PlaneGeometry(_this.phantomFlatWidth, _this.phantomHeight);
        mesh = new THREE.Mesh(geometry);
        mesh.position.z = _this.phantomCurvedRadius;
        _this.meshes.push(mesh);

        geometry = new THREE.CylinderGeometry  (_this.phantomCurvedRadius, _this.phantomCurvedRadius, _this.phantomHeight, 32, 1, true, Math.PI, Math.PI );
        mesh = new THREE.Mesh(geometry);
        mesh.position.x = -_this.phantomFlatWidth / 2;
        _this.meshes.push(mesh);

        geometry = new THREE.PlaneGeometry(_this.phantomFlatWidth, _this.phantomCurvedRadius*2);
        mesh = new THREE.Mesh(geometry);
        mesh.rotation.x = Math.PI / 2;
        mesh.position.y = -_this.phantomHeight / 2;
        _this.meshes.push(mesh);

        geometry = new THREE.PlaneGeometry(_this.phantomFlatWidth, _this.phantomCurvedRadius*2);
        mesh = new THREE.Mesh(geometry);
        mesh.rotation.x = -Math.PI / 2;
        mesh.position.y = _this.phantomHeight/ 2;
        _this.meshes.push(mesh);

        geometry = new THREE.CircleGeometry(_this.phantomCurvedRadius, 32, Math.PI/2, Math.PI);
        mesh = new THREE.Mesh(geometry);
        mesh.rotation.x = -Math.PI / 2;
        mesh.position.y = _this.phantomHeight / 2;
        mesh.position.x = -_this.phantomFlatWidth / 2;
        _this.meshes.push(mesh);

        geometry = new THREE.CircleGeometry(_this.phantomCurvedRadius, 32, -Math.PI/2, Math.PI);
        mesh = new THREE.Mesh(geometry);
        mesh.rotation.x = -Math.PI / 2;
        mesh.position.y = _this.phantomHeight / 2;
        mesh.position.x = _this.phantomFlatWidth / 2;
        _this.meshes.push(mesh);

        geometry = new THREE.CircleGeometry(_this.phantomCurvedRadius, 32, Math.PI/2, Math.PI);
        mesh = new THREE.Mesh(geometry);
        mesh.rotation.x = -Math.PI / 2;
        mesh.rotation.y = Math.PI;
        mesh.position.y = -_this.phantomHeight / 2;
        mesh.position.x = _this.phantomFlatWidth / 2;
        _this.meshes.push(mesh);

        geometry = new THREE.CircleGeometry(_this.phantomCurvedRadius, 32, -Math.PI/2, Math.PI);
        mesh = new THREE.Mesh(geometry);
        mesh.rotation.x = -Math.PI / 2;
        mesh.rotation.y = Math.PI;
        mesh.position.y = -_this.phantomHeight / 2;
        mesh.position.x = -_this.phantomFlatWidth / 2;
        _this.meshes.push(mesh);

        geometry = new THREE.CylinderGeometry(_this.phantomHeadRadius,
            _this.phantomHeadRadius, _this.phantomHeadHeight, 57, 1, true, 1.5*Math.PI, 2*Math.PI );
        mesh = new THREE.Mesh(geometry);
        mesh.position.x = Math.round(_this.phantomFlatWidth / 2) - ( _this.phantomFlatWidth / 2 ) ;
        mesh.position.y = ( _this.phantomHeight + _this.phantomHeadHeight) / 2;
        _this.meshes.push(mesh);

        geometry = new THREE.CircleGeometry(_this.phantomHeadRadius, 32, 0, 2 * Math.PI);
        mesh = new THREE.Mesh(geometry);
        mesh.position.y = ( _this.phantomHeight/2 + _this.phantomHeadHeight);
        mesh.position.x = Math.round(_this.phantomFlatWidth / 2) - ( _this.phantomFlatWidth / 2 ) ;
        mesh.rotation.x = Math.PI / 2;
        mesh.rotation.y = Math.PI;
        _this.meshes.push(mesh);

        //merge all the geometries
        geometry = _this.mergeMeshes(_this.meshes);
        _this.mesh = new THREE.Mesh(geometry, meshFaceMaterial);
        _this.scene.add(_this.mesh);

        // A light source is needed for the MehLambertMaterial
        var directionalLight = new THREE.DirectionalLight( 0xffffff, 1.0 );
        directionalLight.position.set( 0, 0, 1 );
        _this.scene.add( directionalLight );
    };


    /**
     * Internal function to reset the 3d skin dose map
     */
    this.reset = function () {
        var _this = this;
        _this.mesh.position.set( 0, 0, 0 );
        _this.mesh.rotation.set( 0, 0, 0 );
        _this.mesh.scale.set( 1, 1, 1 );
        _this.mesh.updateMatrix();

        _this.camera.position.x = _this.phantomHeight + _this.phantomHeadHeight;
        _this.camera.position.y = 0;
        _this.camera.position.z = 100;
        _this.camera.lookAt(_this.scene.position);
    };


    /**
     * Internal function to animate the 3d skin dose map
     */
    this.animate = function () {
        requestAnimationFrame(animate);
        render();
    };
}
