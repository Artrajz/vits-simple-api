/*
* 获取已加载模型信息
* */
window.addEventListener('DOMContentLoaded', function () {
    var xhr = new XMLHttpRequest();

    xhr.open('GET', '/admin/get_models_info', true);
    xhr.setRequestHeader('Content-Type', 'application/json');

    xhr.onreadystatechange = function () {
        if (xhr.readyState === 4 && xhr.status === 200) {
            var response = JSON.parse(xhr.responseText);
            var model_list = response;
            show_model(model_list);
        }
    };

    xhr.send();
});


var modelData = document.getElementById('model-data');

function show_model(model_list) {

    for (var model_type in model_list) {
        if (model_list[model_type].length > 0) {
            let label = document.createElement('label');
            label.innerText = model_type;

            var model_datas = model_list[model_type];

            for (var key in model_datas) {
                let model_data = model_datas[key];
                renderModelCard(model_data, model_type);
            }
        }

    }
}

function renderModelCard(model_data, model_type) {
    let id = String(model_data["model_id"]);
    let model_path = String(model_data["model_path"]);
    let n_speakers = String(model_data["n_speakers"]);

    let card = document.createElement('div');
    card.classList.add("card");
    card.classList.add(model_type);
    card.setAttribute("data-model-type", model_type);
    card.setAttribute("data-model-id", id);

    let wrap = document.createElement('div');
    wrap.classList.add("wrap");

    let id_obj = document.createElement('div');
    id_obj.innerText = "id: " + id;
    wrap.appendChild(id_obj);
    let model_path_obj = document.createElement('div');
    model_path_obj.innerText = "path: " + model_path;
    wrap.appendChild(model_path_obj);
    let n_speakers_obj = document.createElement('div');
    n_speakers_obj.innerText = "n_speakers: " + n_speakers;
    wrap.appendChild(n_speakers_obj);
    let unload_model = document.createElement('div');
    unload_model.innerText = "x";
    unload_model.classList.add("unload-model");
    wrap.appendChild(unload_model);

    card.appendChild(wrap)

    modelData.appendChild(card);

}

/*
* 获取项目模型目录下的模型与配置文件路径
* */
var loadModelBtn = document.getElementById('load-model-btn');
var modelLoadContent = document.querySelector('.model-load-content');

loadModelBtn.addEventListener('change', function () {
    if (this.checked) {
        // 发送请求获取数据
        var xhr = new XMLHttpRequest();
        xhr.open('GET', '/admin/get_path');
        xhr.onload = function () {
            if (xhr.status === 200) {
                var model_datas = JSON.parse(xhr.responseText);
                renderModelLoadCards(model_datas);

            }
        };
        xhr.send();
    } else {
        modelLoadContent.innerHTML = '';
    }
});

function renderModelLoadCards(data) {
    modelLoadContent.innerHTML = '';

    data.forEach(function (model) {
        var card = document.createElement('div');
        card.classList.add('model-load-card');
        let model_id = model["model_id"];
        let model_path = model.model_path;
        let config_path = model.config_path
        card.textContent = model_id + "|" + model_path + "|" + config_path;
        card.addEventListener('click', function () {
            loadModel(model_path, config_path);
        });

        modelLoadContent.appendChild(card);
    });

    modelLoadContent.style.display = 'block';
}

/*
* 加载模型
* */
function loadModel(modelPath, configPath) {
    var csrftoken = document.querySelector('meta[name="csrf-token"]').getAttribute('content');
    var xhr = new XMLHttpRequest();
    xhr.open('POST', '/admin/load_model');
    xhr.setRequestHeader('Content-Type', 'application/json');
    xhr.setRequestHeader('X-CSRFToken', csrftoken);
    xhr.onload = function () {
        if (xhr.status === 200) {
            location.reload();
        }
    };

    xhr.send(JSON.stringify({
        model_path: modelPath,
        config_path: configPath
    }));

    // 关闭模型加载框
    loadModelBtn.checked = false;
    modelLoadContent.style.display = 'none';
}


/*
* 卸载模型
* */
document.getElementById('model-data').addEventListener('click', function (event) {
    if (event.target.classList.contains('unload-model')) {

        var cardElement = event.target.closest('.card');


        var modelType = cardElement.getAttribute('data-model-type');
        var modelId = cardElement.getAttribute('data-model-id');
        
        const params = {
            "model_type": modelType,
            "model_id": modelId
        };

        var xhr = new XMLHttpRequest();
        var csrftoken = document.querySelector('meta[name="csrf-token"]').getAttribute('content');
        xhr.open('POST', '/admin/unload_model');
        xhr.setRequestHeader('Content-Type', 'application/json');
        xhr.setRequestHeader('X-CSRFToken', csrftoken);
        xhr.onload = function () {
            if (xhr.status === 200) {
                location.reload();
            }
        };

        xhr.send(JSON.stringify(params));

    }
});
