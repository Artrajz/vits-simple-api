// 从localStorage加载上次保存的主题色
const savedThemeColor = localStorage.getItem('themeColor');
if (savedThemeColor) {
  document.documentElement.style.setProperty('--主题色', savedThemeColor);
  document.getElementById('themeColor').value = savedThemeColor;
}

// 监听主题色输入框的变化
document.getElementById('themeColor').addEventListener('input', function () {
  const newThemeColor = this.value;
  document.documentElement.style.setProperty('--主题色', newThemeColor);
  
  // 保存主题色到localStorage
  localStorage.setItem('themeColor', newThemeColor);
});