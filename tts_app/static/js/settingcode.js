var configData = {}

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
                configData = response;
                show_config(response);
            },
            error: function (response) {

            }
        });
        $(".config-save").click(function () {
            set_config();
        });

        // Add new API key
        $('#add-api-key').click(function () {
            var newKey = {
                key: generateRandomKey(),
                enabled: true
            };
            configData.system.api_keys.push(newKey);
            addApiKeyHtml(newKey, configData.system.api_keys.length - 1);
        });

        // Remove API key
        $('#api-keys').on('click', '.btn-remove-key', function () {
            var index = $(this).closest('.input-group').data('index');
            configData.system.api_keys.splice(index, 1);

            $(this).closest('.input-group').remove();

            $('#api-keys .input-group').each(function (i) {
                $(this).attr('data-index', i);
                $(this).find('.input-group-text').text(`API Key ${i + 1}`);
            });
        });

        // Toggle API Key Enabled status
        $('#api-key-enabled').change(function () {
            configData.system.api_key_enabled = $(this).prop('checked');
        });
    }
);

function renderApiKeys() {
    var container = $('#api-keys');
    container.empty(); // Clear previous keys

    configData.system.api_keys.forEach(function (apiKey, index) {
        addApiKeyHtml(apiKey, index);
    });
}

function addApiKeyHtml(apiKey, index) {
    var container = $('#api-keys');

    var apiKeyHtml = `
        <div class="input-group mb-3" data-index="${index}">
            <span class="input-group-text">API Key ${index + 1}</span>
            <input type="text" class="form-control" value="${apiKey.key}" readonly>
            <button class="btn btn-danger btn-remove-key">Remove</button>
            <div class="form-check form-switch ms-2">
                <input class="form-check-input" type="checkbox" ${apiKey.enabled ? 'checked' : ''}>
                <label class="form-check-label">Enabled</label>
            </div>
        </div>
    `;
    container.append(apiKeyHtml);
}

function generateRandomKey() {
    var characters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    var keyLength = 24;
    var result = '';
    for (var i = 0; i < keyLength; i++) {
        var randomIndex = Math.floor(Math.random() * characters.length);
        result += characters[randomIndex];
    }
    return result;
}

function show_config() {
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
        var inputValue = (value === null) ? '' : value;
        $('#bert-vits2-config').append(`
        <div class="input-group mb-3 item">
            <span class="input-group-text">${key}</span>
            <input type="text" class="form-control" id="${itemId}" value="${inputValue}">
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

    $.each(configData.tts_model_config, function (key, value) {
        var formattedKey = key.replace(/_/g, '-');

        if (formattedKey !== "tts-models") {
            $('#tts-model-config').append(`
        <div class="input-group mb-3 item">
            <span class="input-group-text">${key}</span>
            <input type="text" class="form-control" id="${formattedKey}" value="${value}">
        </div>
        `);
        }
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

    // api keys
    renderApiKeys(configData);

    $.each(configData.system, function (key, value) {
        var formattedKey = key.replace(/_/g, '-');
        if (formattedKey == 'api-key-enabled' || formattedKey == 'cache-audio') {
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
    configData = {}

    $('.configuration .form-label').each(function () {
        var labelId = $(this).next().attr('id');
        var nestedDict = {};


        $('#' + labelId).find('.item').each(function () {
            var itemId = $(this).find('input, select').attr('id').replace(/-/g, '_');

            // 还原组合 key
            itemId = itemId.replace(labelId.replace(/-/g, '_') + "_", "");

            var itemValue;
            if ($(this).find('input').is(':checkbox')) {
                // 如果是复选框，获取复选框的状态
                itemValue = $(this).find('input').prop('checked');
            } else {
                // 如果不是复选框，获取输入框或选择框的值
                itemValue = $(this).find('input, select').val();

                // 将空字符串转换为 null
                if (itemValue === "") {
                    itemValue = null;
                }

                // 特殊处理 language_automatic_detect 字段
                if (itemId === "language_automatic_detect") {
                    itemValue = itemValue ? itemValue.split(" ") : [];
                }
            }
            nestedDict[itemId] = itemValue;
        });

        // 处理 system 配置中的 api_keys 信息
        if (labelId === "system") {
            nestedDict['api_keys'] = [];
            $('#api-keys .input-group').each(function () {
                var apiKey = $(this).find('input[type="text"]').val();
                var apiKeyEnabled = $(this).find('.form-check-input').is(':checked');
                nestedDict['api_keys'].push({
                    key: apiKey,
                    enabled: apiKeyEnabled
                });
            });
        }

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
            // location.reload();
        },
        error: function (error) {
            alert("保存配置时出错，请查看日志！");
        }
    });

    return configData
}
