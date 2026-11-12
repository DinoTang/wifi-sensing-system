import {demoMessages} from './demo.js';
import {paintChart, paintSkeleton} from './plots.js';
import {CsiStream, LANDMARK_COUNT, MAX_GAP_MS} from './stream.js';

const element = id => document.getElementById(id);
const stream = new CsiStream();
const format = new Intl.NumberFormat('vi-VN', {maximumFractionDigits: 2});
const dateTime = new Intl.DateTimeFormat('vi-VN', {
  year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit',
  second: '2-digit', fractionalSecondDigits: 3, hour12: false,
});
const PHASES = {
  idle: ['Chưa bắt đầu', 'Bấm Bắt đầu phiên để nhận CSI.', 'neutral'],
  starting: ['Đang tải mô hình', 'Đợi backend chuẩn bị phiên.', 'warn'],
  warming_up: ['Đang tích lũy dữ liệu', 'Đợi đủ mẫu CSI cho một cửa sổ.', 'warn'],
  ready: ['Đang nhận dữ liệu', 'Đang cập nhật theo timestamp CSI.', 'ok'],
  insufficient_data: ['Thiếu dữ liệu CSI', 'Đợi cửa sổ CSI hợp lệ tiếp theo.', 'warn'],
  incompatible: ['Sai cấu hình CSI', 'Nguồn CSI không khớp cấu hình mô hình.', 'error'],
  disconnected: ['Mất nguồn CSI', 'Backend đang đợi dữ liệu từ bộ thu.', 'error'],
  model_error: ['Lỗi mô hình', 'Backend chưa thể trả kết quả suy luận.', 'error'],
  source_error: ['Lỗi bản ghi CSI', 'Kiểm tra tệp dữ liệu và log của backend.', 'error'],
  ended: ['Đã phát lại xong', 'Bắt đầu phiên mới để xem lại.', 'neutral'],
  stopped: ['Đã dừng phiên', 'Bắt đầu phiên mới để tiếp tục.', 'neutral'],
};
const REASONS = {
  CSI_GAP: 'Có khoảng trống giữa các bản ghi CSI.',
  CSI_CADENCE: 'Tần số lấy mẫu nằm ngoài khoảng chấp nhận.',
  CSI_TIMESTAMP_ORDER: 'Timestamp của bộ thu bị trùng hoặc đi lùi.',
  RESULT_TIMEOUT: 'Backend chưa trả kết quả mới.',
  WAITING_FOR_CAPTURE_FILES: 'Backend đang đợi tệp của bộ thu.',
  MODEL_LOAD_FAILED: 'Backend không nạp được mô hình.',
  MODEL_INFERENCE_FAILED: 'Backend gặp lỗi khi chạy suy luận.',
};
const SOURCE_LABELS = {live: 'Trực tiếp', replay: 'Phát lại', mock: 'Giả lập'};
const STARTABLE = new Set(['idle', 'ended', 'stopped', 'source_error']);
const MAX_RECONNECT_MS = 5000;

let mode = new URLSearchParams(location.search).get('source') === 'demo' ? 'mock' : 'backend';
let view = 'signal';
let transport = 'connecting';
let socket = null;
let reconnect = null;
let reconnectDelay = 500;
let stopDemo = null;
let pendingAction = null;
let animationFrame = 0;
let pointer = null;
let lastChart = null;
let lastPose = null;
let lastView = view;
let drawTime = null;

function setText(id, value) {
  const target = element(id);
  if (target.textContent !== value) target.textContent = value;
}

function scheduleRender() {
  if (!animationFrame) animationFrame = requestAnimationFrame(render);
}

function receive(message) {
  if (stream.accept(message, performance.now())) scheduleRender();
}

function connect() {
  transport = 'connecting';
  stream.reset();
  scheduleRender();
  const url = new URL('/ws', location.href);
  url.protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const current = new WebSocket(url);
  socket = current;
  current.addEventListener('open', () => {
    if (current !== socket) return;
    transport = 'open';
    reconnectDelay = 500;
    scheduleRender();
  });
  current.addEventListener('message', event => {
    if (current !== socket || typeof event.data !== 'string') return;
    let message;
    try {
      message = JSON.parse(event.data);
    } catch {
      return;
    }
    if (message?.source === 'live' || message?.source === 'replay') receive(message);
  });
  current.addEventListener('close', () => {
    if (current !== socket) return;
    socket = null;
    stream.reset();
    transport = 'retrying';
    scheduleRender();
    reconnect = setTimeout(connect, reconnectDelay);
    reconnectDelay = Math.min(reconnectDelay * 2, MAX_RECONNECT_MS);
  });
}

function changeMode() {
  clearTimeout(reconnect);
  const previousSocket = socket;
  socket = null;
  previousSocket?.close();
  stopDemo?.();
  stopDemo = null;
  pendingAction?.abort();
  pendingAction = null;
  element('request-error').hidden = true;
  stream.reset();
  if (mode === 'mock') {
    transport = 'demo';
    stopDemo = demoMessages(receive, element('demo-state').value);
  } else {
    reconnectDelay = 500;
    connect();
  }
  scheduleRender();
}

async function controlSession(action) {
  if (pendingAction || mode !== 'backend' || transport !== 'open') return;
  const controller = new AbortController();
  pendingAction = controller;
  const timeout = setTimeout(() => controller.abort(), 5000);
  element('request-error').hidden = true;
  scheduleRender();
  try {
    const response = await fetch(`/api/session/${action}`, {method: 'POST', signal: controller.signal});
    if (!response.ok) {
      throw new Error(response.status === 409 ? 'Một phiên khác đang chạy.' : `Backend trả lỗi HTTP ${response.status}.`);
    }
  } catch (error) {
    if (pendingAction === controller) {
      const message = controller.signal.aborted || error instanceof TypeError
        ? 'Backend chưa phản hồi lệnh điều khiển.' : error.message;
      setText('request-error', message);
      element('request-error').hidden = false;
    }
  } finally {
    clearTimeout(timeout);
    if (pendingAction === controller) {
      pendingAction = null;
      scheduleRender();
    }
  }
}

function route() {
  if (location.hash === '#presence') view = 'presence';
  if (!location.hash || location.hash === '#signal') view = 'signal';
  pointer = null;
  scheduleRender();
}

function phaseFor(snapshot) {
  if (transport === 'retrying') {
    return ['Mất kết nối', 'Đang thử kết nối lại với backend.', 'error'];
  }
  if (!stream.status) return ['Đang kết nối', 'Đang đợi trạng thái từ backend.', 'warn'];
  if (snapshot.stale) return ['Dữ liệu đã cũ', 'Đang đợi cửa sổ CSI mới.', 'warn'];
  if (stream.status.status === 'ready' && !snapshot.chart) {
    return ['Chờ cửa sổ CSI', 'Backend đang chuẩn bị mẫu tiếp theo.', 'warn'];
  }
  const [title, description, tone] = PHASES[stream.status.status];
  return [title, REASONS[stream.status.reason] ?? description, tone];
}

function updateHeader(phase) {
  const state = stream.status;
  const source = mode === 'mock' ? 'mock' : state?.source;
  const transportLabel = {
    connecting: ['Đang kết nối', 'warn'], open: ['Đã kết nối', 'ok'],
    retrying: ['Mất kết nối', 'error'], demo: ['Chế độ thử giao diện', 'neutral'],
  };
  setText('transport', transportLabel[transport][0]);
  element('transport').dataset.tone = transportLabel[transport][1];
  setText('source-tag', SOURCE_LABELS[source] ?? 'Chờ nguồn');
  element('source-tag').dataset.source = source ?? 'unknown';
  setText('session-id', mode === 'mock' ? 'Giả lập' : state?.session_id?.slice(0, 8) ?? '—');
  element('session-id').title = mode === 'mock' ? '' : state?.session_id ?? '';
  const capture = state?.capture;
  setText('capture-info', capture?.rate && capture.features
    ? `${capture.rate.split(',').at(-1)} · ${capture.features} giá trị` : '—');

  element('demo-controls').hidden = mode !== 'mock';
  element('session-actions').hidden = mode === 'mock';
  const startable = STARTABLE.has(state?.status)
    || (state?.status === 'model_error' && state.reason === 'MODEL_LOAD_FAILED');
  const controlsEnabled = transport === 'open' && !pendingAction && Boolean(state);
  element('start').disabled = !controlsEnabled || !startable;
  element('stop').disabled = !controlsEnabled || startable;
  setText('stream-status', phase[0]);
  element('stream-status').dataset.tone = phase[2];
}

function updateView() {
  const presence = view === 'presence';
  element('signal-panel').hidden = presence;
  element('presence-panel').hidden = !presence;
  element('nav-signal').toggleAttribute('aria-current', !presence);
  element('nav-presence').toggleAttribute('aria-current', presence);
  element(presence ? 'nav-presence' : 'nav-signal').setAttribute('aria-current', 'page');
  setText('page-title', presence ? 'Phát hiện hiện diện' : 'Tín hiệu CSI');
  setText('page-description', presence
    ? 'Trạng thái trong phòng và tư thế 2D từ mô hình CSI.'
    : 'Biên độ Channel State Information theo thời gian.');
}

function showTimestamp(id, timestamp) {
  setText(id, timestamp === null ? 'Chưa có timestamp' : dateTime.format(timestamp));
  element(id).title = timestamp === null ? '' : `${timestamp} ms`;
}

function sampleRate(points) {
  const intervals = [];
  for (let index = 1; index < points.length; index += 1) {
    const interval = points[index].timestamp_ms - points[index - 1].timestamp_ms;
    if (interval > 0 && interval <= MAX_GAP_MS) intervals.push(interval);
  }
  if (!intervals.length) return '—';
  intervals.sort((a, b) => a - b);
  const mid = Math.floor(intervals.length / 2);
  const median = intervals.length % 2 ? intervals[mid] : (intervals[mid - 1] + intervals[mid]) / 2;
  return format.format(1000 / median);
}

function renderSignal(chart, phase) {
  const range = Number(element('range').value);
  const points = chart
    ? stream.points.filter(point => point.timestamp_ms >= chart.timestamp - range * 1000)
    : [];
  setText('chart-description', mode === 'mock' ? 'Biên độ I/Q mẫu' : 'Biên độ I/Q thô trước chuẩn hóa');
  setText('amplitude', points.length ? format.format(points.at(-1).value) : '—');
  setText('sample-rate', chart ? sampleRate(chart.points) : '—');
  setText('point-count', String(points.length));
  showTimestamp('chart-time', chart?.timestamp ?? null);
  element('chart-placeholder').hidden = Boolean(points.length);
  setText('chart-empty-title', phase[0]);
  setText('chart-empty-note', phase[1]);

  const canvas = element('chart');
  const hover = paintChart(canvas, points, chart?.timestamp ?? 0, range, pointer);
  canvas.setAttribute('aria-label', points.length
    ? `Biểu đồ biên độ CSI, ${points.length} mẫu trong ${range} giây`
    : 'Biểu đồ CSI chưa có mẫu');
  const tooltip = element('chart-tooltip');
  tooltip.hidden = !hover;
  if (hover) {
    tooltip.textContent = `${format.format(hover.point.value)} đv. thô · ${dateTime.format(hover.point.timestamp_ms)}`;
    const left = Math.max(5, Math.min(hover.x + 10, canvas.clientWidth - tooltip.offsetWidth - 6));
    tooltip.style.left = `${left}px`;
  }
}

function renderPresence(chart, pose, phase) {
  const label = pose ? (pose.present ? 'Có người' : 'Phòng trống') : 'Chưa có kết quả';
  const note = pose
    ? (mode === 'mock' ? 'Trạng thái mẫu' : 'Dự đoán của mô hình từ CSI')
    : (chart ? 'Chờ kết quả cùng timestamp với biểu đồ.' : phase[1]);
  const joints = pose?.joints.size ?? 0;
  const probability = pose?.probability ?? null;
  setText('presence-state', label);
  setText('presence-note', note);
  element('verdict').dataset.tone = pose ? (pose.present ? 'ok' : 'neutral') : phase[2];
  setText('joint-count', pose ? `${joints}/${LANDMARK_COUNT}` : '—');
  setText('model-version', pose?.model ?? stream.status?.model_version ?? '—');
  setText('backend-latency', pose?.processingMs != null ? `${format.format(pose.processingMs)} ms` : '—');
  element('probability-group').hidden = probability === null;
  element('probability').value = probability ?? 0;
  setText('probability-value', probability === null ? '—' : `${format.format(probability * 100)}%`);
  setText('pose-description', mode === 'mock' ? 'Tọa độ mẫu · BlazePose' : 'Tọa độ dự đoán từ CSI · BlazePose');
  setText('frame-size', pose?.frame ? `${pose.frame.width} × ${pose.frame.height}` : '—');
  showTimestamp('pose-time', pose?.timestamp ?? null);

  element('pose-placeholder').hidden = joints > 0;
  setText('pose-empty-title', pose?.present && !joints ? 'Thiếu tọa độ khớp' : label);
  setText('pose-empty-note', pose && !pose.present ? 'Không có bộ xương ở cửa sổ này.' : note);
  const canvas = element('skeleton');
  paintSkeleton(canvas, pose);
  canvas.setAttribute('aria-label', joints ? `Bộ xương 2D, ${joints} khớp` : 'Bộ xương 2D chưa có khớp');
}

function render() {
  animationFrame = 0;
  const snapshot = stream.snapshot(performance.now());
  const {chart, pose} = snapshot;
  const phase = phaseFor(snapshot);
  updateHeader(phase);
  updateView();
  if (view === 'signal') renderSignal(chart, phase);
  else renderPresence(chart, pose, phase);

  const result = view === 'signal' ? chart : pose;
  const previous = view === 'signal' ? lastChart : lastPose;
  if (view !== lastView) {
    drawTime = null;
  } else if (result !== previous) {
    drawTime = result ? performance.now() - result.receivedAt : null;
  }
  const arrivals = stream.chartArrivals;
  const duration = arrivals.at(-1) - arrivals[0];
  const refresh = chart && duration > 0 ? `${format.format(1000 * (arrivals.length - 1) / duration)} Hz cập nhật` : null;
  setText('display-timing', [
    refresh, drawTime !== null ? `${format.format(drawTime)} ms từ nhận đến vẽ` : null,
  ].filter(Boolean).join(' · ') || '—');
  lastChart = chart;
  lastPose = pose;
  lastView = view;
}

element('source-mode').value = mode;
element('source-mode').addEventListener('change', event => {
  mode = event.target.value;
  const url = new URL(location.href);
  if (mode === 'mock') url.searchParams.set('source', 'demo');
  else url.searchParams.delete('source');
  history.replaceState(null, '', url);
  changeMode();
});
element('demo-state').addEventListener('change', changeMode);
element('range').addEventListener('change', scheduleRender);
element('start').addEventListener('click', () => controlSession('start'));
element('stop').addEventListener('click', () => controlSession('stop'));
element('chart').addEventListener('pointermove', event => {
  pointer = event.clientX - element('chart').getBoundingClientRect().left;
  scheduleRender();
});
element('chart').addEventListener('pointerleave', () => {
  pointer = null;
  scheduleRender();
});
window.addEventListener('hashchange', route);
document.addEventListener('visibilitychange', scheduleRender);
const resizeObserver = new ResizeObserver(scheduleRender);
resizeObserver.observe(element('chart'));
resizeObserver.observe(element('skeleton'));
setInterval(() => {
  const {chart, pose} = stream.snapshot(performance.now());
  if (chart !== lastChart || pose !== lastPose) scheduleRender();
}, 200);
route();
changeMode();
