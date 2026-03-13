{
    "header": {
        "releaseVersion": "2026.1.0+develop",
        "fileVersion": "2.0",
        "nodesVersions": {
            "CameraInit": "12.0",
            "CopyFiles": "1.3",
            "ImageSegmentationSam3": "1.0",
            "Sam3dObjects": "1.1"
        },
        "template": true
    },
    "graph": {
        "CameraInit_1": {
            "nodeType": "CameraInit",
            "position": [
                0,
                0
            ],
            "inputs": {}
        },
        "ImageSegmentationSam3_1": {
            "nodeType": "ImageSegmentationSam3",
            "position": [
                200,
                0
            ],
            "inputs": {
                "input": "{CameraInit_1.output}",
                "prompt": "main object",
                "keepFilename": true
            }
        },
        "Sam3dObjects_1": {
            "nodeType": "Sam3dObjects",
            "position": [
                400,
                0
            ],
            "inputs": {
                "input": "{ImageSegmentationSam3_1.input}",
                "maskFolder": "{ImageSegmentationSam3_1.output}",
                "maskExtension": "exr"
            }
        },
        "CopyFiles_1": {
            "nodeType": "CopyFiles",
            "position": [
                600,
                0
            ],
            "inputs": {
                "inputFiles": [
                    "{Sam3dObjects_1.output}"
                ]
            }
        }
    }
}