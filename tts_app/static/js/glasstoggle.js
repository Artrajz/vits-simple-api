document.addEventListener("DOMContentLoaded", function() {
    // 在页面加载时设置开关状态
    setInitialSwitchState();
});

function toggleGlassEffect() {
    var glassToggle = document.getElementById('glassToggle');
    var body = document.body;

    if (glassToggle.checked) {
        // 启用毛玻璃效果
        body.style.backdropFilter = 'blur(10px)';
        body.style.opacity = '0.8';  // 调整透明度
    } else {
        // 禁用毛玻璃效果
        body.style.backdropFilter = 'none';
        body.style.opacity = '1';  // 恢复透明度
    }

    // 保存开关状态到本地存储
    saveSwitchState(glassToggle.checked);
}

function setInitialSwitchState() {
    var glassToggle = document.getElementById('glassToggle');
    var savedState = getSwitchStateFromStorage();

    if (savedState !== null) {
        // 设置开关状态为上次保存的状态
        glassToggle.checked = savedState;
    } else {
        // 如果本地存储中没有保存的状态，默认关闭开关
        glassToggle.checked = false;
    }

    // 更新毛玻璃效果
    toggleGlassEffect();
}

function saveSwitchState(state) {
    // 使用本地存储保存开关状态
    localStorage.setItem('glassToggleState', state);
}

function getSwitchStateFromStorage() {
    // 从本地存储获取开关状态
    return localStorage.getItem('glassToggleState') === 'true';
}