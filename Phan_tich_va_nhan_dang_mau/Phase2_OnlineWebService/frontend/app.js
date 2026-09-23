(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const video = $('video');
  const canvas = $('canvas');
  const context = canvas.getContext('2d');
  const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
  const signs = [...alphabet, 'Space', 'Delete'];
  const labels = { Space: 'Khoảng trắng', Delete: 'Xóa chữ' };
  const mediaPipeBase = 'https://cdn.jsdelivr.net/npm/@mediapipe/hands/';

  let hands = null;
  let stream = null;
  let active = false;
  let ready = false;
  let frame = 0;
  let generation = 0;
  let requestPending = false;
  let lastRequest = 0;
  let lastCommitted = 0;
  let history = [];
  let prediction = '';
  let textValue = '';
  let controlOpenFrames = 0;
  let controlClosedFrames = 0;
  let controlArmed = false;

  function message(value) {
    $('action-message').textContent = value;
  }

  function renderText() {
    $('typed-text').textContent = textValue;
    $('typed-placeholder').hidden = Boolean(textValue);
    $('copy-text').disabled = !textValue;
    $('speak-text').disabled = !textValue;
    $('delete-char').disabled = !textValue;
    $('reset-text').disabled = !textValue;
  }

  function setPrediction(value) {
    prediction = value;
    $('prediction').textContent = value || '—';
    $('prediction-help').textContent = value ? (labels[value] || 'Mở rồi nắm tay phải để chốt chữ') : 'Đưa tay tạo ký hiệu vào khung hình';
  }

  function resetPrediction() {
    generation++;
    history = [];
    setPrediction('');
  }

  function commitPrediction() {
    if (!prediction) return;
    if (prediction === 'Delete') textValue = textValue.slice(0, -1);
    else if (prediction === 'Space') textValue += ' ';
    else if (alphabet.includes(prediction)) textValue += prediction;
    else return;
    renderText();
    message(prediction === 'Delete' ? 'Đã xóa một ký tự.' : `Đã chốt ${labels[prediction] || prediction}.`);
  }

  async function checkHealth() {
    $('connection-banner').classList.remove('ready', 'error');
    $('connection-text').textContent = 'Đang kiểm tra mô hình nhận diện...';
    $('retry-connection').hidden = true;
    $('start-camera').disabled = true;
    try {
      const response = await fetch('/api/health', { cache: 'no-store' });
      if (!response.ok) throw new Error('Mô hình chưa sẵn sàng');
      if (typeof window.Hands !== 'function' || typeof window.drawConnectors !== 'function') {
        throw new Error('Không tải được thư viện theo dõi bàn tay');
      }
      ready = true;
      $('connection-banner').classList.add('ready');
      $('connection-text').textContent = 'Mô hình đã sẵn sàng. Bạn có thể bật camera.';
      $('start-camera').disabled = active;
    } catch (error) {
      ready = false;
      $('connection-banner').classList.add('error');
      $('connection-text').textContent = `${error.message}. Kiểm tra kết nối rồi thử lại.`;
      $('retry-connection').hidden = false;
      if (active) stopCamera();
    }
  }

  function extendedFingers(points) {
    let count = 0;
    for (const [tip, pip] of [[8, 6], [12, 10], [16, 14], [20, 18]]) {
      const distance = (p) => Math.hypot(p.x - points[0].x, p.y - points[0].y);
      if (distance(points[tip]) > distance(points[pip]) * 1.08) count++;
    }
    return count;
  }

  function updateControl(points) {
    if (!points) {
      controlOpenFrames = 0;
      controlClosedFrames = 0;
      controlArmed = false;
      $('right-status').textContent = 'Tay chốt chữ: chưa thấy';
      return;
    }
    const extended = extendedFingers(points);
    $('right-status').textContent = extended >= 3 ? 'Tay chốt chữ: đang mở' : extended <= 1 ? 'Tay chốt chữ: đang nắm' : 'Tay chốt chữ: đang chuyển động';
    if (extended >= 3) {
      controlOpenFrames++;
      controlClosedFrames = 0;
      if (controlOpenFrames >= 2) controlArmed = true;
    } else if (extended <= 1) {
      controlClosedFrames++;
      controlOpenFrames = 0;
      if (controlArmed && controlClosedFrames >= 2 && Date.now() - lastCommitted > 650) {
        commitPrediction();
        lastCommitted = Date.now();
        controlArmed = false;
      }
    } else {
      controlOpenFrames = 0;
      controlClosedFrames = 0;
    }
  }

  async function recognize(points) {
    if (requestPending || Date.now() - lastRequest < 180) return;
    requestPending = true;
    lastRequest = Date.now();
    const requestGeneration = generation;
    try {
      // The video is mirrored in CSS. Flip x to match the original model input.
      const landmarks = points.map((point) => ({ x: 1 - point.x, y: point.y, z: point.z }));
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ landmarks })
      });
      if (!response.ok) {
        if (response.status === 503) checkHealth();
        throw new Error(`API ${response.status}`);
      }
      const result = await response.json();
      if (!active || requestGeneration !== generation) return;
      const rawValue = String(result.prediction || '').trim();
      const value = /^(del|delete)$/i.test(rawValue) ? 'Delete' : /^space$/i.test(rawValue) ? 'Space' : rawValue.toUpperCase();
      if (!signs.includes(value)) {
        resetPrediction();
        return;
      }
      history.push(value);
      if (history.length > 4) history.shift();
      const counts = new Map(history.map((entry) => [entry, history.filter((item) => item === entry).length]));
      const [candidate, count] = [...counts].sort((a, b) => b[1] - a[1])[0];
      if (count >= 2) setPrediction(candidate);
    } catch (error) {
      if (active) message('Tạm mất kết nối nhận diện. Hãy kiểm tra mạng hoặc thử lại.');
    } finally {
      requestPending = false;
    }
  }

  function onResults(results) {
    if (!active) return;
    context.clearRect(0, 0, canvas.width, canvas.height);
    let signHand = null;
    let controlHand = null;
    (results.multiHandLandmarks || []).forEach((points, index) => {
      const label = results.multiHandedness?.[index]?.label;
      // MediaPipe labels correspond to a mirrored input. With the unmirrored
      // camera image, its "Right" label is the user's physical left hand.
      if (label === 'Right') signHand = points;
      if (label === 'Left') controlHand = points;
      window.drawConnectors(context, points, window.HAND_CONNECTIONS, { color: label === 'Right' ? '#91a3ff' : '#54d9ae', lineWidth: 3 });
      window.drawLandmarks(context, points, { color: '#ffffff', fillColor: '#5065bf', radius: 2 });
    });
    $('left-status').textContent = signHand ? 'Tay tạo ký hiệu: đã thấy' : 'Tay tạo ký hiệu: chưa thấy';
    if (signHand) recognize(signHand);
    else resetPrediction();
    updateControl(controlHand);
  }

  async function cameraLoop() {
    if (!active || !hands) return;
    try {
      if (video.readyState >= 2) await hands.send({ image: video });
    } catch (error) {
      message('Không thể theo dõi bàn tay. Hãy tắt rồi bật lại camera.');
      stopCamera();
      return;
    }
    if (active) frame = requestAnimationFrame(cameraLoop);
  }

  async function startCamera() {
    if (!ready || active) return;
    $('start-camera').disabled = true;
    try {
      if (!navigator.mediaDevices?.getUserMedia) throw new Error('Trình duyệt cần HTTPS hoặc localhost để dùng camera');
      stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' } });
      video.srcObject = stream;
      await video.play();
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      hands = new window.Hands({ locateFile: (file) => `${mediaPipeBase}${file}` });
      hands.setOptions({ maxNumHands: 2, modelComplexity: 1, minDetectionConfidence: 0.65, minTrackingConfidence: 0.55 });
      hands.onResults(onResults);
      active = true;
      $('camera-placeholder').hidden = true;
      $('live-chip').hidden = false;
      $('stop-camera').hidden = false;
      $('camera-indicator').textContent = 'Đang bật';
      $('camera-indicator').classList.add('live');
      message('Camera đang bật. Đưa tay trái tạo ký hiệu vào khung hình.');
      cameraLoop();
    } catch (error) {
      stopCamera();
      message(`Không mở được camera: ${error.name === 'NotAllowedError' ? 'Bạn cần cấp quyền camera trong trình duyệt.' : error.message}`);
    }
  }

  function stopCamera() {
    active = false;
    generation++;
    cancelAnimationFrame(frame);
    stream?.getTracks().forEach((track) => track.stop());
    stream = null;
    video.srcObject = null;
    const oldHands = hands;
    hands = null;
    if (oldHands) Promise.resolve(oldHands.close()).catch(() => {});
    context.clearRect(0, 0, canvas.width, canvas.height);
    $('camera-placeholder').hidden = false;
    $('live-chip').hidden = true;
    $('stop-camera').hidden = true;
    $('start-camera').disabled = !ready;
    $('camera-indicator').textContent = 'Chưa bật';
    $('camera-indicator').classList.remove('live');
    $('left-status').textContent = 'Tay tạo ký hiệu: chưa thấy';
    $('right-status').textContent = 'Tay chốt chữ: chưa thấy';
    controlArmed = false;
    history = [];
    setPrediction('');
  }

  function openImage(sign) {
    $('expanded-image').src = `/assets/${sign}.jpg`;
    $('expanded-image').alt = `Minh họa ký hiệu ${labels[sign] || sign}`;
    $('image-caption').textContent = labels[sign] || sign;
    $('image-dialog').showModal();
  }

  function signTile(sign) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'sign-tile';
    button.setAttribute('aria-label', `Xem ký hiệu ${labels[sign] || sign}`);
    const img = document.createElement('img');
    img.src = `/assets/${sign}.jpg`;
    img.alt = `Ký hiệu ${labels[sign] || sign}`;
    img.loading = 'lazy';
    const label = document.createElement('strong');
    label.textContent = labels[sign] || sign;
    button.append(img, label);
    button.addEventListener('click', () => openImage(sign));
    return button;
  }

  $('learn-form').addEventListener('submit', (event) => {
    event.preventDefault();
    const value = $('learn-input').value.normalize('NFD').replace(/[\u0300-\u036f]/g, '').replace(/đ/gi, 'D').toUpperCase();
    const gallery = $('tutorial-gallery');
    gallery.replaceChildren();
    let found = false;
    for (const char of value) {
      if (alphabet.includes(char)) {
        gallery.append(signTile(char));
        found = true;
      } else if (char === ' ' && found) {
        gallery.append(signTile('Space'));
      }
    }
    if (!found) {
      const empty = document.createElement('p');
      empty.className = 'empty-gallery';
      empty.textContent = 'Nhập chữ cái để xem cách đánh vần.';
      gallery.append(empty);
    }
  });

  $('dictionary-grid').append(...signs.map(signTile));
  $('open-dictionary').addEventListener('click', () => $('dictionary-dialog').showModal());
  $('close-dictionary').addEventListener('click', () => $('dictionary-dialog').close());
  $('close-image').addEventListener('click', () => $('image-dialog').close());
  for (const id of ['dictionary-dialog', 'image-dialog']) {
    $(id).addEventListener('click', (event) => { if (event.target === $(id)) $(id).close(); });
  }
  $('start-camera').addEventListener('click', startCamera);
  $('stop-camera').addEventListener('click', stopCamera);
  $('retry-connection').addEventListener('click', checkHealth);
  $('copy-text').addEventListener('click', async () => {
    try { await navigator.clipboard.writeText(textValue); message('Đã sao chép văn bản.'); }
    catch { message('Không thể sao chép tự động trong trình duyệt này.'); }
  });
  $('speak-text').addEventListener('click', () => {
    if (!('speechSynthesis' in window)) { message('Trình duyệt chưa hỗ trợ phát âm.'); return; }
    speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(textValue);
    utterance.lang = 'en-US';
    speechSynthesis.speak(utterance);
  });
  $('delete-char').addEventListener('click', () => { textValue = textValue.slice(0, -1); renderText(); message('Đã xóa một ký tự.'); });
  $('reset-text').addEventListener('click', () => { textValue = ''; renderText(); message('Đã làm mới văn bản.'); });
  window.addEventListener('pagehide', stopCamera);
  renderText();
  checkHealth();
})();
