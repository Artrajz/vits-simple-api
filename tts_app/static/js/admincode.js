/*
* 获取已加载模型信息
* */
$(document).ready(
    get_models_info()
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


var modelData = $('#model-data');

function show_model(model_list) {
    for (var model_type in model_list) {
        if (model_list[model_type].length > 0) {
            var model_datas = model_list[model_type];
            var label = $('<label></label>').text(model_type);

            $.each(model_datas, function (key, model_data) {
                renderModelCard(model_data, model_type);
            });
        }
    }
}

function renderModelCard(model_data, model_type) {
    var id = String(model_data["model_id"]);
    var model_path = String(model_data["model_path"]);
    var n_speakers = String(model_data["n_speakers"]);

    var card = $('<div></div>').addClass("card " + model_type).attr({
        "data-model-type": model_type,
        "data-model-id": id
    });

    var wrap = $('<div></div>').addClass("wrap");

    $('<div></div>').text("id: " + id).appendTo(wrap);
    $('<div></div>').text("path: " + model_path).appendTo(wrap);
    $('<div></div>').text("n_speakers: " + n_speakers).appendTo(wrap);
    $('<div></div>').text("x").addClass("unload-model").appendTo(wrap);

    card.append(wrap);

    modelData.append(card);
}


/*
* 获取项目模型目录下的模型与配置文件路径
* */
var loadModelBtn = $('#load-model-btn');
var modelLoadContent = $('.model-load-content');

loadModelBtn.on('change', function () {
    if (this.checked) {
        $.get('/admin/get_path', function (response) {
            var model_datas = response;
            renderModelLoadCards(model_datas);
        });
    } else {
        modelLoadContent.empty();
    }
});

function renderModelLoadCards(data) {
    modelLoadContent.empty();

    $.each(data, function (index, model) {
        var card = $('<div></div>').addClass('model-load-card');
        var model_id = model.model_id;
        var model_path = model.model_path;
        var config_path = model.config_path;
        card.text(model_id + "|" + model_path + "|" + config_path);
        card.on('click', function () {
            loadModel(model_path, config_path);
        });

        modelLoadContent.append(card);
    });

    modelLoadContent.css('display', 'block');
}


/*
* 加载模型
* */
function loadModel(modelPath, configPath) {
    var csrftoken = $('meta[name="csrf-token"]').attr('content');
    
    $.ajax({
        url: '/admin/load_model',
        type: 'POST',
        contentType: 'application/json',
        headers: {
            'X-CSRFToken': csrftoken
        },
        data: JSON.stringify({
            model_path: modelPath,
            config_path: configPath
        }),
        success: function (response) {
            modelData.empty();
            get_models_info();
        },
        error: function (response) {
            alert("Unload model failed!");
        }
    });

    // 关闭模型加载框
    $('#load-model-btn').prop('checked', false);
    $('.model-load-content').hide();
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
            modelData.empty();
            get_models_info();
        },
        error: function (response) {
            alert("Unload model failed!");
        }
    });
});


