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
    }
);


function show_config(configData, parentKey = 1) {
    var configuration = $(".configuration");

    $.each(configData, function (key, value) {
        if (key == "model_list") {
            var wrap = $('<div></div>').addClass("config-item" + parentKey + " model-list");
            $('<label></label>').text(key).addClass("config-key").appendTo(wrap);
            for (var i = 0; i < value.length; i++) {
                console.log(value[i][0])
                let item = $('<div></div>').addClass("model-data model-item flex")
                let model_id = $('<label></label>').text(i).addClass("model-label")
                model_id.appendTo(item);
                let model = $('<input>').val(value[i][0]).addClass("model")
                model.appendTo(item);
                let config = $('<input>').val(value[i][1]).addClass("config")
                config.appendTo(item);
                item.appendTo(wrap);
            }
            wrap.appendTo(configuration);
            return true;
        }
        var wrap = $('<div></div>').addClass("config-item" + parentKey);
        $('<label></label>').text(key).addClass("config-key").appendTo(wrap);

        if (typeof value === 'object') {
            wrap.appendTo(configuration);
            show_config(value, parentKey + 1);
        } else {
            $('<input>').val(value).addClass("config-value").appendTo(wrap);
            wrap.appendTo(configuration);
        }


    });
}


