// 从 localStorage 中获取之前保存的 player_type 值，如果没有保存的值，默认为 0
// 全局范围声明 player_type
let player_type = localStorage.getItem('player_type') || 0;

document.addEventListener('DOMContentLoaded', function () {
    var audioPlayer1 = document.getElementById('audioPlayer1');
    var audioPlayer2 = document.getElementById('audioPlayer2');
    var audioPlayer3 = document.getElementById('audioPlayer3');

    // 通过类名获取所有包含频谱 Canvas 的元素
    var spectrogramCanvases = document.querySelectorAll('.spectrogramCanvas');
    var toggleSpectrumLabel = document.getElementById('toggleSpectrumLabel');


    // 初始状态，显示播放器或频谱，根据 localStorage 中保存的值
    if (player_type == 0) {
        audioPlayer1.style.display = 'block';
        audioPlayer2.style.display = 'block';
        audioPlayer3.style.display = 'block';

        // 遍历所有频谱 Canvas 隐藏它们
        spectrogramCanvases.forEach(function (canvas) {
            canvas.style.display = 'none';
        });
    } else {
        audioPlayer1.style.display = 'none';
        audioPlayer2.style.display = 'none';
        audioPlayer3.style.display = 'none';

        // 遍历所有频谱 Canvas 显示它们
        spectrogramCanvases.forEach(function (canvas) {
            canvas.style.display = 'block';
        });
    }

    // 切换按钮点击事件
    toggleSpectrumLabel.addEventListener('click', function () {
        // 暂停或停止播放器
        audioPlayer1.pause(); // 或 audioPlayer.stop()，具体取决于你的实现
        audioPlayer2.pause(); // 或 audioPlayer.stop()，具体取决于你的实现
        audioPlayer3.pause(); // 或 audioPlayer.stop()，具体取决于你的实现

        // 根据 player_type 的值切换播放器类型
        if (player_type == 0) {
            // 切换到频谱播放器
            audioPlayer1.style.display = 'none';
            audioPlayer2.style.display = 'none';
            audioPlayer3.style.display = 'none';

            // 遍历所有频谱 Canvas 显示它们
            spectrogramCanvases.forEach(function (canvas) {
                canvas.style.display = 'block';
            });

            player_type = 1;
        } else {
            // 切换到原始播放器
            audioPlayer1.style.display = 'block';
            audioPlayer2.style.display = 'block';
            audioPlayer3.style.display = 'block';

            // 遍历所有频谱 Canvas 隐藏它们
            spectrogramCanvases.forEach(function (canvas) {
                canvas.style.display = 'none';
            });

            player_type = 0;
        }

        // 保存 player_type 的值到 localStorage
        localStorage.setItem('player_type', player_type);
    });
});
