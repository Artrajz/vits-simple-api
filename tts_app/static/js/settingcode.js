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
    }
);

function show_config(configData) {
    $.each(configData.vits_config, function (key, value) {
        var formattedKey = key.replace(/_/g, '-');
        //为了避免id冲突，组合key作为id
        var itemId = 'vits-config-' + formattedKey;
        $('#vits-config').append(`
        <div class="input-group mb-3 item">
            <span class="input-group-text">${key}</span>
            <input type="text" class="form-control" id="${itemId}" value="${value}">
        </div>
        `);
    })

    $.each(configData.w2v2_vits_config, function (key, value) {
        var formattedKey = key.replace(/_/g, '-');
        var itemId = 'w2v2-vits-config-' + formattedKey;
        $('#w2v2-vits-config').append(`
        <div class="input-group mb-3 item">
            <span class="input-group-text">${key}</span>
            <input type="text" class="form-control" id="${itemId}" value="${value}">
        </div>
        `);
    })

    $.each(configData.hubert_vits_config, function (key, value) {
        var formattedKey = key.replace(/_/g, '-');
        var itemId = 'hubert-vits-config-' + formattedKey;
        $('#hubert-vits-config').append(`
        <div class="input-group mb-3 item">
            <span class="input-group-text">${key}</span>
            <input type="text" class="form-control" id="${itemId}" value="${value}">
        </div>
        `);
    })

    $.each(configData.bert_vits2_config, function (key, value) {
        var formattedKey = key.replace(/_/g, '-');
        var itemId = 'bert-vits2-config-' + formattedKey;
        $('#bert-vits2-config').append(`
        <div class="input-group mb-3 item">
            <span class="input-group-text">${key}</span>
            <input type="text" class="form-control" id="${itemId}" value="${value}">
        </div>
        `);
    })

    $.each(configData.log_config, function (key, value) {
        var formattedKey = key.replace(/_/g, '-');

        if (key != 'logging_level') {
            $('#log-config').append(`
            <div class="input-group mb-3 item">
                <span class="input-group-text">${key}</span>
                <input type="text" class="form-control" id="${formattedKey}" value="${value}">
            </div>
            `);
        }

    });

    $.each(configData.model_config, function (key, value) {
        var formattedKey = key.replace(/_/g, '-');

        $('#model-config').append(`
        <div class="input-group mb-3 item">
            <span class="input-group-text">${key}</span>
            <input type="text" class="form-control" id="${formattedKey}" value="${value}">
        </div>
        `);
    });

    $.each(configData.language_identification, function (key, value) {
        var formattedKey = key.replace(/_/g, '-');
        $('#language-identification ' + '#' + formattedKey).val(value)
    });

    $.each(configData.http_service, function (key, value) {
        var formattedKey = key.replace(/_/g, '-');
        if (formattedKey == 'api-key-enable' || formattedKey == 'debug') {
            $('#' + formattedKey).prop('checked', value);
        } else {
            $('#' + formattedKey).val(value);
        }
    });

    $.each(configData.system, function (key, value) {
        var formattedKey = key.replace(/_/g, '-');
        if (formattedKey == 'api-key-enable' || formattedKey == 'cache-audio') {
            $('#' + formattedKey).prop('checked', value);
        } else {
            $('#' + formattedKey).val(value);
        }
    });

    $.each(configData.admin, function (key, value) {
        $('#' + key).val(value);
    });


}

function set_config() {
    var configData = {}

    $('.configuration .form-label').each(function () {
        var labelId = $(this).next().attr('id');
        var nestedDict = {};


        $('#' + labelId).find('.item').each(function () {
            var itemId = $(this).find('input, select').attr('id').replace(/-/g, '_');

            if (labelId == 'vits-config' || labelId == 'w2v2-vits-config' || labelId == 'hubert-vits-config' || labelId == 'bert-vits2-config') {
                //还原组合key
                itemId = itemId.replace(labelId.replace(/-/g, '_') + "_", "");
            }
            var itemValue = $(this).find('input, select').val();
            nestedDict[itemId] = itemValue;
        });

        configData[labelId.replace(/-/g, '_')] = nestedDict;
    });

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
