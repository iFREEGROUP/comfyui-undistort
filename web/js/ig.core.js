import { app } from '../../../scripts/app.js'
import { api } from '../../../scripts/api.js'

function chainCallback(object, property, callback) {
    if (object == undefined) {
        //This should not happen.
        console.error("Tried to add callback to non-existant object")
        return;
    }
    if (property in object) {
        const callback_orig = object[property]
        object[property] = function () {
            const r = callback_orig.apply(this, arguments);
            callback.apply(this, arguments);
            return r
        };
    } else {
        object[property] = callback;
    }
}


async function uploadFile(file) {
    //TODO: Add uploaded file to cache with Cache.put()?
    try {
        // Wrap file in formdata so it includes filename
        const body = new FormData();
        const i = file.webkitRelativePath.lastIndexOf('/');
        const subfolder = file.webkitRelativePath.slice(0,i+1)
        const new_file = new File([file], file.name, {
            type: file.type,
            lastModified: file.lastModified,
        });
        body.append("image", new_file);
        if (i > 0) {
            body.append("subfolder", subfolder);
        }
        const resp = await api.fetchApi("/upload/image", {
            method: "POST",
            body,
        });

        if (resp.status === 200) {
            return resp.status
        } else {
            alert(resp.status + " - " + resp.statusText);
        }
    } catch (error) {
        alert(error);
    }
}


function addUploadWidget(nodeType, nodeData, widgetName, type="video") {
    chainCallback(nodeType.prototype, "onNodeCreated", function() {
        const pathWidget = this.widgets.find((w) => w.name === widgetName);
        const fileInput = document.createElement("input");
        chainCallback(this, "onRemoved", () => {
            fileInput?.remove();
        });
        if (type == "folder") {
            Object.assign(fileInput, {
                type: "file",
                style: "display: none",
                webkitdirectory: true,
                onchange: async () => {
                    const directory = fileInput.files[0].webkitRelativePath;
                    const i = directory.lastIndexOf('/');
                    if (i <= 0) {
                        throw "No directory found";
                    }
                    const path = directory.slice(0,directory.lastIndexOf('/'))
                    if (pathWidget.options.values.includes(path)) {
                        alert("A folder of the same name already exists");
                        return;
                    }
                    let successes = 0;
                    for(const file of fileInput.files) {
                        if (await uploadFile(file) == 200) {
                            successes++;
                        } else {
                            //Upload failed, but some prior uploads may have succeeded
                            //Stop future uploads to prevent cascading failures
                            //and only add to list if an upload has succeeded
                            if (successes > 0) {
                                break
                            } else {
                                return;
                            }
                        }
                    }
                    pathWidget.options.values.push(path);
                    pathWidget.value = path;
                    if (pathWidget.callback) {
                        pathWidget.callback(path)
                    }
                },
            });
        } else {
            throw "Unknown upload type"
        }
        document.body.append(fileInput);
        let uploadWidget = this.addWidget("button", "choose " + type + " to upload", "image", () => {
            //clear the active click event
            app.canvas.node_widget = null

            fileInput.click();
        });
        uploadWidget.options.serialize = false;
    });
}

function useKVState(nodeType) {
    chainCallback(nodeType.prototype, "onNodeCreated", function () {
        chainCallback(this, "onConfigure", function(info) {
            if (!this.widgets) {
                //Node has no widgets, there is nothing to restore
                return
            }
            if (typeof(info.widgets_values) != "object") {
                //widgets_values is in some unknown inactionable format
                return
            }
            let widgetDict = info.widgets_values
            if (info.widgets_values.length) {
                //widgets_values is in the old list format
                if (this.type in convDict) {
                    //widget does not have a conversion format provided
                    let convList = convDict[this.type];
                    if(info.widgets_values.length >= convList.length) {
                        //has all required fields
                        widgetDict = {}
                        for (let i = 0; i < convList.length; i++) {
                            if(!convList[i]) {
                                //Element should not be processed (upload button on load image sequence)
                                continue
                            }
                            widgetDict[convList[i]] = info.widgets_values[i];
                        }
                    } else {
                        //widgets_values is missing elements marked as required
                        //let it fall through to failure state
                    }
                }
            }
            if (widgetDict.length == undefined) {
                for (let w of this.widgets) {
                    if (w.name in widgetDict) {
                        w.value = widgetDict[w.name];
                    } else {
                        
                        //attempt to restore default value
                        let inputs = LiteGraph.getNodeType(this.type).nodeData.input;
                        let initialValue = null;
                        if (inputs?.required?.hasOwnProperty(w.name)) {
                            if (inputs.required[w.name][1]?.hasOwnProperty("default")) {
                                initialValue = inputs.required[w.name][1].default;
                            } else if (inputs.required[w.name][0].length) {
                                initialValue = inputs.required[w.name][0][0];
                            }
                        } else if (inputs?.optional?.hasOwnProperty(w.name)) {
                            if (inputs.optional[w.name][1]?.hasOwnProperty("default")) {
                                initialValue = inputs.optional[w.name][1].default;
                            } else if (inputs.optional[w.name][0].length) {
                                initialValue = inputs.optional[w.name][0][0];
                            }
                        }
                        if (initialValue) {
                            w.value = initialValue;
                        }
                    }
                }
            } else {
                //Saved data was not a map made by this method
                //and a conversion dict for it does not exist
                //It's likely an array and that has been blindly applied
                if (info?.widgets_values?.length != this.widgets.length) {
                    //Widget could not have restored properly
                    //Note if multiple node loads fail, only the latest error dialog displays
                    app.ui.dialog.show("Failed to restore node: " + this.title + "\nPlease remove and re-add it.")
                    this.bgcolor = "#C00"
                }
            }
        });
        chainCallback(this, "onSerialize", function(info) {
            info.widgets_values = {};
            if (!this.widgets) {
                //object has no widgets, there is nothing to store
                return;
            }
            for (let w of this.widgets) {
                info.widgets_values[w.name] = w.value;
            }
        });
    })
}

function addLoadImagesCommon(nodeType, nodeData) {
    // addVideoPreview(nodeType);
    // addPreviewOptions(nodeType);
    chainCallback(nodeType.prototype, "onNodeCreated", function() {
        const pathWidget = this.widgets.find((w) => w.name === "directory");
        
        //do first load
        requestAnimationFrame(() => {
            for (let w of [pathWidget]) {
                w.callback(w.value, null, this);
            }
        });
    });
}


app.registerExtension({
    name: "IG.Core",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if(nodeData?.name?.startsWith("IG_")) {
            useKVState(nodeType);
            chainCallback(nodeType.prototype, "onNodeCreated", function () {
                let new_widgets = []
                if (this.widgets) {
                    for (let w of this.widgets) {
                        let input = this.constructor.nodeData.input
                        let config = input?.required[w.name] ?? input.optional[w.name]
                        if (!config) {
                            continue
                        }
                        if (w?.type == "text" && config[1].vhs_path_extensions) {
                            new_widgets.push(app.widgets.VHSPATH({}, w.name, ["VHSPATH", config[1]]));
                        } else {
                            new_widgets.push(w)
                        }
                    }
                    this.widgets = new_widgets;
                }
            });
        }
        if (nodeData?.name == "IG_LoadCheckerboardImageForCalibrateCamera") {
            addUploadWidget(nodeType, nodeData, "directory", "folder");
            chainCallback(nodeType.prototype, "onNodeCreated", function() {
                const pathWidget = this.widgets.find((w) => w.name === "directory");
                chainCallback(pathWidget, "callback", (value) => {
                    if (!value) {
                        return;
                    }
                });
            });
            addLoadImagesCommon(nodeType, nodeData);
        } 
    },
    async getCustomWidgets() {
        return {
            VHSPATH(node, inputName, inputData) {
                let w = {
                    name : inputName,
                    type : "VHS.PATH",
                    value : "",
                    draw : function(ctx, node, widget_width, y, H) {
                        //Adapted from litegraph.core.js:drawNodeWidgets
                        var show_text = app.canvas.ds.scale > 0.5;
                        var margin = 15;
                        var text_color = LiteGraph.WIDGET_TEXT_COLOR;
                        var secondary_text_color = LiteGraph.WIDGET_SECONDARY_TEXT_COLOR;
                        ctx.textAlign = "left";
                        ctx.strokeStyle = LiteGraph.WIDGET_OUTLINE_COLOR;
                        ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR;
                        ctx.beginPath();
                        if (show_text)
                            ctx.roundRect(margin, y, widget_width - margin * 2, H, [H * 0.5]);
                        else
                            ctx.rect( margin, y, widget_width - margin * 2, H );
                        ctx.fill();
                        if (show_text) {
                            if(!this.disabled)
                                ctx.stroke();
                            ctx.save();
                            ctx.beginPath();
                            ctx.rect(margin, y, widget_width - margin * 2, H);
                            ctx.clip();

                            //ctx.stroke();
                            ctx.fillStyle = secondary_text_color;
                            const label = this.label || this.name;
                            if (label != null) {
                                ctx.fillText(label, margin * 2, y + H * 0.7);
                            }
                            ctx.fillStyle = text_color;
                            ctx.textAlign = "right";
                            let disp_text = this.format_path(String(this.value))
                            ctx.fillText(disp_text, widget_width - margin * 2, y + H * 0.7); //30 chars max
                            ctx.restore();
                        }
                    },
                    mouse : searchBox,
                    options : {},
                    format_path : function(path) {
                        //Formats the full path to be under 30 characters
                        if (path.length <= 30) {
                            return path;
                        }
                        let filename = path_stem(path)[1]
                        if (filename.length > 28) {
                            //may all fit, but can't squeeze more info
                            return filename.substr(0,30);
                        }
                        //TODO: find solution for windows, path[1] == ':'?
                        let isAbs = path[0] == '/';
                        let partial = path.substr(path.length - (isAbs ? 28:29))
                        let cutoff = partial.indexOf('/');
                        if (cutoff < 0) {
                            //Can occur, but there isn't a nicer way to format
                            return path.substr(path.length-30);
                        }
                        return (isAbs ? '/…':'…') + partial.substr(cutoff);

                    }
                };
                if (inputData.length > 1) {
                    if (inputData[1].vhs_path_extensions) {
                        w.options.extensions = inputData[1].vhs_path_extensions;
                    }
                    if (inputData[1].default) {
                        w.value = inputData[1].default;
                    }
                }

                if (!node.widgets) {
                    node.widgets = [];
                }
                node.widgets.push(w);
                return w;
            }
        }
    },
    async loadedGraphNode(node) {
        //Check and migrate inputs named batch_manager from old workflows
        if (node.type?.startsWith("VHS_") && node.inputs) {
            const batchInput = node.inputs.find((i) => i.name == "batch_manager")
            if (batchInput) {
                batchInput.name = "meta_batch"
            }
        }
    }
});
