$(document).ready(function () {
        $.ajax({
            url: '/admin/get_config',
            type: 'POST',
            dataType: 'json',
            contentType: 'application/json',
            headers: {
                'X-CSRFToken': $('meta[name="csrf-token"]').attr('content')
            },
            success: function (response) {
                show_config(response);
            },
            error: function (response) {

            }
        });
        $(".config-save").click(function () {
            set_config();
        });
        $("#addModelRow").click(function () {
            var newRow = $('<div class="input-group mb-3 item"></div>');
            var inputGroupText1 = $('<span class="input-group-text"></span>').text("模型路径");
            var inputGroupText2 = $('<span class="input-group-text"></span>').text("配置路径");
            var input1 = $('<input type="text" class="form-control model-path">').attr("placeholder", "模型路径");
            var input2 = $('<input type="text" class="form-control config-path">').attr("placeholder", "配置路径");

            inputGroupText1.appendTo(newRow);
            input1.appendTo(newRow);
            inputGroupText2.appendTo(newRow);
            input2.appendTo(newRow);
            newRow.appendTo(".model-list");
        });
    }
);

function show_config(configData) {
    $.each(configData.default_parameter, function (key, value) {
        $('#' + key).val(value);
    })

    $.each(configData.model_config, function (key, value) {
        if (key != 'model_list') {
            $('#' + key).val(value);
        } else {
            $.each(value, function (index, element) {
                var newRow = $('<div class="input-group mb-3 item"></div>');
                var inputGroupText1 = $('<span class="input-group-text"></span>').text("模型路径");
                var inputGroupText2 = $('<span class="input-group-text"></span>').text("配置路径");
                var input1 = $('<input type="text" class="form-control model-path">').attr("placeholder", "模型路径").val(element[0]);
                var input2 = $('<input type="text" class="form-control config-path">').attr("placeholder", "配置路径").val(element[1]);
                inputGroupText1.appendTo(newRow);
                input1.appendTo(newRow);
                inputGroupText2.appendTo(newRow);
                input2.appendTo(newRow);
                newRow.appendTo(".model-list");
            })

        }
    })
    // console.log(configData.LANGUAGE_IDENTIFICATION_LIBRARY)
    $('#lang-lib').val(configData.LANGUAGE_IDENTIFICATION_LIBRARY);
    $('#espeak-ng').val(configData.ESPEAK_LIBRARY);
    $('#detect').val(configData.LANGUAGE_AUTOMATIC_DETECT);

    $('#logging-level').val(configData.LOGGING_LEVEL);
    $('#log-backupcount').val(configData.LOGS_BACKUPCOUNT);
    $('#logs-path').val(configData.LOGS_PATH);

    $('#api-key').val(configData.API_KEY);
    $('#api-key-enable').prop('checked', configData.API_KEY_ENABLED);
    $.each(configData.users.admin, function (key, value) {
        if (key == "admin") {
            $('#username').val(value.username);
            $('#password').val(value.password);
            $('#password').prop('type', 'password');
        }

    })


}

function set_config() {
    var configData = {
        API_KEY: $('#api-key').val(),
        API_KEY_ENABLED: $('#api-key-enable').prop('checked'),
        default_parameter: {
            format: $('#format').val(),
            id: $('#id').val(),
            lang: $('#lang').val(),
            length: $('#length').val(),
            segment_size: $('#segment_size').val(),
            noise: $('#noise').val(),
            noisew: $('#noisew').val(),
            sdp_ratio: $('#sdp_ratio').val(),
            length_zh: $('#length_zh').val(),
            length_ja: $('#length_ja').val(),
            length_en: $('#length_en').val()
        },
        model_config: {
            dimensional_emotion_npy: $('#dimensional_emotion_npy').val(),
            hubert_soft_model: $('#hubert_soft_model').val(),
        },
        LANGUAGE_IDENTIFICATION_LIBRARY: $('#lang-lib').val(),
        ESPEAK_LIBRARY: $('#espeak-ng').val(),
        LOGGING_LEVEL: $('#logging-level').val(),
        LOGS_BACKUPCOUNT: $('#log-backupcount').val(),
        LOGS_PATH: $('#logs-path').val(),
        LANGUAGE_AUTOMATIC_DETECT: $('#detect').val(),
        users: {admin: {admin: {id: 1, username: $('#username').val(), password: $('#password').val()}}}

    }

    var modelListData = [];
    $(".model-list .item").each(function () {
        var modelPath = $(this).find(".model-path").val();
        var configPath = $(this).find(".config-path").val();
        if (modelPath || configPath) {
            modelListData.push([modelPath, configPath]);
        }
    })
    configData.model_config.model_list = modelListData;


    $.ajax({
        type: "POST",
        url: "/admin/set_config",
        data: JSON.stringify(configData), // 将配置数据转换为JSON字符串
        contentType: "application/json",
        headers: {
            'X-CSRFToken': $('meta[name="csrf-token"]').attr('content')
        },
        success: function (response) {
            alert("配置已保存");
            location.reload();
        },
        error: function (error) {
            alert("保存配置时出错");
        }
    });

    return configData
}
