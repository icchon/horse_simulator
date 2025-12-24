const ctx = document.getElementById('resultChart').getContext('2d');
const loading = document.getElementById('loading');
let resultChart = null;

document.addEventListener('DOMContentLoaded', function() {
  // ナビゲーションバーのトグル処理
  const navbarToggle = document.querySelector('.navbar-toggle');
  const mainContent = document.querySelector(".main-content");
  const navLinks = document.querySelectorAll(".nav-link");

  navbarToggle.addEventListener('click', function() {
    sidebar.classList.toggle('active');
    mainContent.classList.toggle("shifted");
  });

  const currentPath = window.location.pathname;
  navLinks.forEach(function(link) {
    if (link.getAttribute("href") === currentPath) {
      link.classList.add("active");
    }
  });
});

// グラフ描画関数
function drawChart(labels, chartData) {
  // グラフ描画の処理
}

// ページ読み込み時の処理
window.addEventListener('DOMContentLoaded', function() {
  const labels = JSON.parse(document.getElementById('labels').textContent);
  const chartData = JSON.parse(document.getElementById('chartData').textContent);
  
  if (labels.length > 0 && chartData.length > 0) {
    // グラフを描画
    drawChart(labels, chartData);
    loading.style.display = "none";
    document.querySelector(".horse-animation").classList.remove("running");
  }

  // ポップオーバーを有効化
  var popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
  var popoverList = popoverTriggerList.map(function (popoverTriggerEl) {
    return new bootstrap.Popover(popoverTriggerEl, {
      trigger: 'focus'
    });
  });
});

// フォーム送信時の処理
document.querySelector('form').addEventListener('submit', function(event) {
  // フォーム送信時の処理
});