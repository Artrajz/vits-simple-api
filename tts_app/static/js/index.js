var baseUrl = getProtocol() + getUrl();
var currentModelPage = 1;
var vitsSpeakersCount = 0;
var w2v2SpeakersCount = 0;
var bertVits2SpeakersCount = 0;
var selectedFile = null;

function speakersInit() {
    $.ajax({
        url: '/voice/speakers',
        type: 'GET',
        dataType: 'json',
        success: function (data) {
            vitsSpeakersCount = data['VITS'].length;
            w2v2SpeakersCount = data['W2V2-VITS'].length;
            bertVits2SpeakersCount = data['BERT-VITS2'].length;
            showModelContentBasedOnStatus();
        },
        error: function (xhr, status, error) {
            console.error('Request failed with status', status, 'and error', error);
        }
    });
}

$(function () {
    $('[data-toggle="tooltip"]').tooltip()
})

function getProtocol() {
    return 'https:' == location.protocol ? "https://" : "http://";
}

function getUrl() {
    let url = window.location.host;
    return url;
}


function setBaseUrl() {
    let text = document.getElementById("input_text" + currentModelPage).value;
    let id = document.getElementById("input_id" + currentModelPage).value;

    let vits_link = document.getElementById("vits_link");
    let speakers_link = document.getElementById("speakers_link");

    let vits_url = baseUrl + "/voice/vits?text=" + text + "&id=" + id;
    let speakers_url = baseUrl + "/voice/speakers";

    vits_link.href = vits_url;
    vits_link.textContent = vits_url;

    speakers_link.href = speakers_url;
    speakers_link.textContent = speakers_url;
}

function getLink() {
    let text = document.getElementById("input_text" + currentModelPage).value;
    let id = document.getElementById("input_id" + currentModelPage).value;
    let format = document.getElementById("input_format" + currentModelPage).value;
    let lang = document.getElementById("input_lang" + currentModelPage).value;
    let length = document.getElementById("input_length" + currentModelPage).value;
    let noise = document.getElementById("input_noise" + currentModelPage).value;
    let noisew = document.getElementById("input_noisew" + currentModelPage).value;
    let segment_size = document.getElementById("input_segment_size" + currentModelPage).value;

    let url = baseUrl
    let streaming = null;
    let sdp_ratio = null;
    let emotion = null;
    if (currentModelPage == 1) {
        streaming = document.getElementById('streaming1');
        url += "/voice/vits?text=" + text + "&id=" + id;
    } else if (currentModelPage == 2) {
        emotion = document.getElementById('emotion').value;
        url += "/voice/w2v2-vits?text=" + text + "&id=" + id + "&emotion=" + emotion;
    } else if (currentModelPage == 3) {
        sdp_ratio = document.getElementById("input_sdp_ratio").value;
        streaming = document.getElementById('streaming3');
        emotion = document.getElementById('input_emotion3').value;
        url += "/voice/bert-vits2?text=" + text + "&id=" + id;

    } else {
        console.error("Invalid model page: ", currentModelPage);
        return null;
    }
    if (format != "") {
        url += "&format=" + format;
    }
    if (lang != "") {
        url += "&lang=" + lang;
    }
    if (length != "") {
        url += "&length=" + length;
    }
    if (noise != "") {
        url += "&noise=" + noise;
    }
    if (noisew != "") {
        url += "&noisew=" + noisew;
    }
    if (segment_size != "") {
        url += "&segment_size=" + segment_size;
    }

    if (currentModelPage == 1) {
        if (streaming.checked)
            url += '&streaming=true';
    } else if (currentModelPage == 3) {
        if (streaming.checked)
            url += '&streaming=true';
        if (sdp_ratio != "")
            url += "&sdp_ratio=" + sdp_ratio;
        length_zh = document.getElementById("input_length_zh3").value;
        if (length_zh != "")
            url += "&length_zh=" + length_zh;
        length_ja = document.getElementById("input_length_ja3").value;
        if (length_ja != "")
            url += "&length_ja=" + length_ja;
        length_en = document.getElementById("input_length_en3").value;
        if (length_en != "")
            url += "&length_en=" + length_en;
        if (emotion !== null && emotion !== "")
            url += "&emotion=" + emotion;
    }

    return url;
}

function updateLink() {
    let url = getLink();
    let link = document.getElementById("vits_link");
    link.href = url;
    link.textContent = url;
}

function setAudioSourceByGet() {
    if (currentModelPage == 1 && vitsSpeakersCount <= 0) {
        alert("未加载VITS模型");
        return;
    }
    if (currentModelPage == 2 && w2v2SpeakersCount <= 0) {
        alert("未加载W2V2-VITS模型");
        return;
    }
    if (currentModelPage == 3 && bertVits2SpeakersCount <= 0) {
        alert("未加载Bert-VITS2模型");
        return;
    }
    let url = getLink();

    // Add a timestamp parameter to prevent browser caching
    let timestamp = new Date().getTime();
    url += '&t=' + timestamp;

    let audioPlayer = document.getElementById("audioPlayer" + currentModelPage);
    audioPlayer.src = url;
    audioPlayer.play();
}

function setAudioSourceByPost() {
    if (currentModelPage == 1 && vitsSpeakersCount <= 0) {
        alert("未加载VITS模型");
        return;
    }
    if (currentModelPage == 2 && w2v2SpeakersCount <= 0) {
        alert("未加载W2V2-VITS模型");
        return;
    }
    if (currentModelPage == 3 && bertVits2SpeakersCount <= 0) {
        alert("未加载Bert-VITS2模型");
        return;
    }

    let text = $("#input_text" + currentModelPage).val();
    let id = $("#input_id" + currentModelPage).val();
    let format = $("#input_format" + currentModelPage).val();
    let lang = $("#input_lang" + currentModelPage).val();
    let length = $("#input_length" + currentModelPage).val();
    let noise = $("#input_noise" + currentModelPage).val();
    let noisew = $("#input_noisew" + currentModelPage).val();
    let segment_size = $("#input_segment_size" + currentModelPage).val();

    let formData = new FormData();
    formData.append('text', text);
    formData.append('id', id);
    formData.append('format', format);
    formData.append('lang', lang);
    formData.append('length', length);
    formData.append('noise', noise);
    formData.append('noisew', noisew);
    formData.append('segment_size', segment_size);

    let url = "";
    let streaming = null;
    let sdp_ratio = null;
    let length_zh = 0;
    let length_ja = 0;
    let length_en = 0;
    let emotion = null;

    if (currentModelPage == 1) {
        url = baseUrl + "/voice/vits";
        streaming = $("#streaming1")[0];
    } else if (currentModelPage == 2) {
        emotion = $("#emotion").val();
        url = baseUrl + "/voice/w2v2-vits";
    } else if (currentModelPage == 3) {
        sdp_ratio = $("#input_sdp_ratio").val();
        url = baseUrl + "/voice/bert-vits2";
        streaming = $("#streaming3")[0];
        length_zh = $("#input_length_zh3").val();
        length_ja = $("#input_length_ja3").val();
        length_en = $("#input_length_en3").val();
        emotion = $("#input_emotion3").val();
    }

    // 添加其他配置参数到 FormData
    if ((currentModelPage == 1 || currentModelPage == 3) && streaming.checked) {
        formData.append('streaming', true);
    }
    if (currentModelPage == 3 && sdp_ratio != "") {
        formData.append('sdp_ratio', sdp_ratio);
    }
    if (currentModelPage == 3 && length_zh != "") {
        formData.append('length_zh', length_zh);
    }
    if (currentModelPage == 3 && length_ja != "") {
        formData.append('length_ja', length_ja);
    }
    if (currentModelPage == 3 && length_en != "") {
        formData.append('length_en', length_en);
    }
    if ((currentModelPage == 2 || currentModelPage == 3) && emotion != null && emotion != "") {
        formData.append('emotion', emotion);
    }
    if (currentModelPage == 3 && selectedFile) {
        formData.append('reference_audio', selectedFile);
    }

    let downloadButton = document.getElementById("downloadButton" + currentModelPage);

    // 发送请求
    $.ajax({
        url: url,
        method: 'POST',
        data: formData,
        processData: false,
        contentType: false,
        responseType: 'blob',
        xhrFields: {
            responseType: 'blob'
        },
        success: function (response, status, xhr) {
            let blob = new Blob([response], {type: 'audio/wav'});
            let audioPlayer = document.getElementById("audioPlayer" + currentModelPage);
            let audioFileName = getFileNameFromResponseHeader(xhr);
            audioPlayer.setAttribute('data-file-name', audioFileName);
            audioPlayer.src = URL.createObjectURL(blob);
            audioPlayer.load();
            audioPlayer.play();

            downloadButton.disabled = false;
        },
        error: function (error) {
            console.error('Error:', error);
            alert("无法获取音频数据");
            downloadButton.disabled = true;
        }
    });
}


function getFileNameFromResponseHeader(xhr) {
    var contentDispositionHeader = xhr.getResponseHeader('Content-Disposition');
    var matches = contentDispositionHeader.match(/filename=(.+)$/);
    return matches ? matches[1] : 'audio.wav';  // 如果无法从响应头获取文件名，则使用默认值
}

function downloadAudio() {
    let audioPlayer = document.getElementById("audioPlayer" + currentModelPage);
    let audioFileName = audioPlayer.getAttribute('data-file-name') || 'audio.wav';

    let downloadLink = document.createElement('a');
    downloadLink.href = audioPlayer.src;

    downloadLink.download = audioFileName;

    document.body.appendChild(downloadLink);
    downloadLink.click();
    document.body.removeChild(downloadLink);
}


function showContent(index) {
    const panes = document.querySelectorAll(".content-pane");
    const buttons = document.querySelectorAll(".tab-button");
    currentModelPage = index + 1;

    for (let i = 0; i < panes.length; i++) {
        if (i === index) {
            panes[i].classList.add("active");
            buttons[i].classList.add("active");

        } else {
            panes[i].classList.remove("active");
            buttons[i].classList.remove("active");
        }
    }
    updateLink();
}

function showModelContentBasedOnStatus() {
    if (vitsSpeakersCount > 0) {
        showContent(0);
    } else if (w2v2SpeakersCount > 0) {
        showContent(1);
    } else if (bertVits2SpeakersCount > 0) {
        showContent(2);
    } else {
        showContent(0);
    }
}

function setDefaultParameter() {
    $.ajax({
        url: "/voice/default_parameter",
        method: 'POST',
        responseType: 'json',
        success: function (response) {
            default_parameter = response;
            for (var key in default_parameter) {
                if (default_parameter.hasOwnProperty(key)) {
                    $(".input_" + key).attr("placeholder", default_parameter[key]);
                }
            }
        },
        error: function (error) {
        }
    });
}

$(document).ready(function () {
    speakersInit();
    setDefaultParameter();
    setBaseUrl();

    $('.reference_audio').fileinput({
        language: 'zh',
        dropZoneEnabled: true,
        dropZoneTitle: "上传参考音频",
        uploadAsync: false,
        maxFileCount: 1,
        showPreview: false,
        showUpload: false
    }).on("fileloaded", function (event, file, previewId, index, reader) {
        selectedFile = file;
    }).on("filecleared", function (event) {
        selectedFile = null;
    });
});