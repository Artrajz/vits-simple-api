/*
* 获取已加载模型信息
* */
$(document).ready(function () {
        get_models_info();
        get_model_to_load();
        $("#save-current-model").click(function () {
            save_current_model();
        });
    }
);

function get_models_info() {
    $.ajax({
        url: '/admin/get_models_info',
        type: 'GET',
        dataType: 'json',
        contentType: 'application/json',
        success: function (response) {
            var model_list = response;
            show_model(model_list);
        }
    });
}


var modelTypes = {
    'VITS': $('#VITS'),
    'HUBERT-VITS': $('#HUBERT-VITS'),
    'W2V2-VITS': $('#W2V2-VITS'),
    'BERT-VITS2': $('#BERT-VITS2'),
    'GPT-SOVITS': $('#GPT-SOVITS')
};


function show_model(model_list) {
    for (var modelType in modelTypes) {
        modelTypes[modelType].empty();
        modelTypes[modelType].prev('label').remove();
    }

    for (var model_type in model_list) {
        if (model_list[model_type].length > 0) {
            var model_datas = model_list[model_type];
            var label = $('<label></label>').text(model_type);
            modelTypes[model_type].before(label);

            $.each(model_datas, function (key, model_data) {
                renderModelCard(model_data, model_type);
            });
        }
    }
}

function renderModelCard(model_data, model_type) {
    var id = String(model_data["model_id"]);
    var vits_path = String(model_data["vits_path"]);
    var n_speakers = String(model_data["n_speakers"]);
    var t2s_path = String(model_data["t2s_path"]);

    var card = $('<div></div>').addClass("card model-card " + model_type).attr({
        "data-model-type": model_type,
        "data-model-id": id
    });

    var wrap = $('<div></div>').addClass("wrap");

    // $('<div></div>').text("id: " + id).appendTo(wrap);
    if (model_type == "GPT-SOVITS") {
        $('<div></div>').text(vits_path).addClass("model-card-info1").appendTo(wrap);
        $('<div></div>').text(t2s_path).addClass("model-card-info3").appendTo(wrap);
        $('<div></div>').text("x").addClass("unload-model").appendTo(wrap);
    } else {
        $('<div></div>').text(vits_path).addClass("model-card-info1").appendTo(wrap);
        $('<div></div>').text("n_speakers: " + n_speakers).addClass("model-card-info2").appendTo(wrap);
        $('<div></div>').text("x").addClass("unload-model").appendTo(wrap);
    }


    card.append(wrap);

    modelTypes[model_type].append(card);
}


/*
* 获取项目模型目录下的模型与配置文件路径
* */
var modelLoadContent = $('.model-load-content');
var isRequestInProgress = false;

function get_model_to_load() {
    $.get('/admin/get_path', function (response) {
        var model_datas = response;
        renderModelLoadCards(model_datas);
    });
}

function renderModelLoadCards(data) {
    modelLoadContent.empty();

    $.each(data, function (index, model) {
        var card = $('<div></div>').addClass('model-load-item flex');
        var model_id = model.model_id;
        var model_type = model.model_type;
        var vits_path = model.vits_path;
        var config_path = model.config_path;
        var t2s_path = model.t2s_path;

        var folder = null;
        var vits_name = null;
        var config_name = null;
        var t2s_name = null;

        $('<div></div>').text(model_id.toString()).addClass("unload-model-id").appendTo(card);
        $('<div></div>').text(model_type).addClass("unload-model-type").appendTo(card);

        if (vits_path != null && config_path != null) {
            folder = vits_path.split("/")[0];
            vits_name = vits_path.split("/")[1];
            config_name = config_path.split("/")[1];

            $('<div></div>').text(folder).addClass("unload-model-folder").appendTo(card);
            $('<div></div>').text(vits_name).addClass("unload-vits-path model1").appendTo(card);
            $('<div></div>').text(config_name).addClass("unload-vits-config model2").appendTo(card);
        } else {
            folder = vits_path.split("/")[0];
            vits_name = vits_path.split("/")[1];
            t2s_name = t2s_path.split("/")[1];

            $('<div></div>').text(folder).addClass("unload-model-folder").appendTo(card);
            $('<div></div>').text(vits_name).addClass("unload-vits-path model1").appendTo(card);
            $('<div></div>').text(t2s_name).addClass("unload-t2s-config model2").appendTo(card);
        }

        card.on('click', function () {
            if (!isRequestInProgress) {
                isRequestInProgress = true;
                let tts_model = {};
                if (model_type == "GPT-SOVITS") {
                    tts_model = {
                        model_type: model_type,
                        t2s_path: t2s_path,
                        vits_path: vits_path
                    }
                } else {
                    tts_model = {
                        model_type: model_type,
                        vits_path: vits_path,
                        config_path: config_path
                    }
                }

                loadModel(card, tts_model);
            }

        });


        modelLoadContent.append(card);
    });

}


/*
* 加载模型
* */
function loadModel(card, tts_model) {
    var csrftoken = $('meta[name="csrf-token"]').attr('content');

    $.ajax({
        url: '/admin/load_model',
        type: 'POST',
        contentType: 'application/json',
        headers: {
            'X-CSRFToken': csrftoken
        },
        data: JSON.stringify(tts_model),
        success: function (response) {
            card.fadeOut();
            get_models_info();
            isRequestInProgress = false;
        },
        error: function (response) {
            alert("Model loading failed!");
            isRequestInProgress = false;
        }
    });

}


/*
* 卸载模型
* */
$('#model-data').on('click', '.unload-model', function (event) {
    var cardElement = $(event.target).closest('.card');
    var modelType = cardElement.attr('data-model-type');
    var modelId = cardElement.attr('data-model-id');

    var params = {
        "model_type": modelType,
        "model_id": modelId
    };

    $.ajax({
        url: '/admin/unload_model',
        type: 'POST',
        dataType: 'json',
        contentType: 'application/json',
        data: JSON.stringify(params),
        headers: {
            'X-CSRFToken': $('meta[name="csrf-token"]').attr('content')
        },
        success: function (response) {
            get_models_info();
        },
        error: function (response) {
            alert("Unload model failed!");
        }
    });
});

function save_current_model() {
    $.ajax({
        url: '/admin/save_current_model',
        type: 'POST',
        headers: {
            'X-CSRFToken': $('meta[name="csrf-token"]').attr('content')
        },
        success: function (response) {
            alert("当前已加载的模型已保存，下次启动将自动加载。");
        },
        error: function (response) {
            alert("保存失败！");
        }
    });
}
