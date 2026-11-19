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
  source_error: ['Lỗi nguồn CSI', 'Kiểm tra tệp dữ liệu và log của backend.', 'error'],
  ended: ['Đã phát lại xong', 'Bắt đầu phiên mới để xem lại.', 'neutral'],
  stopped: ['Đã dừng phiên', 'Bắt đầu phiên mới để tiếp tục.', 'neutral'],
};
const REASONS = {
  CSI_GAP: 'Có khoảng trống giữa các bản ghi CSI.',
  CSI_CADENCE: 'Tần số lấy mẫu nằm ngoài khoảng chấp nhận.',
  CSI_TIMESTAMP_ORDER: 'Timestamp của bộ thu bị trùng hoặc đi lùi.',
  RESULT_TIMEOUT: 'Backend chưa trả kết quả mới.',
  WAITING_FOR_CAPTURE_FILES: 'Backend đang đợi tệp của bộ thu.',
  CAPTURE_PLATFORM: 'Thu tự động cần chạy backend trên Ubuntu có bộ thu Intel.',
  CAPTURE_TOOLS: 'Thiếu công cụ thu Intel IAX. Xem thông báo trong terminal backend.',
  CAPTURE_WIFI: 'Kết nối card Intel với Wi-Fi đã cấu hình rồi bắt đầu lại.',
  CAPTURE_WIFI_CHANNEL: 'Kiểm tra kênh Wi-Fi đã cấu hình và độ rộng 20 MHz.',
  CAPTURE_RECORDER_BUSY: 'Bộ thu iaxcsi đang chạy trong một tiến trình khác.',
  CAPTURE_ROUTER: 'Không ping được router. Kiểm tra kết nối và địa chỉ --router.',
  CAPTURE_ACTIVATION: 'Driver Intel chưa bật CSI. Xem setup.log của phiên.',
  CAPTURE_SUDO: 'Quyền sudo cần được xác thực. Chạy lại lệnh mở backend trong terminal Ubuntu.',
  CAPTURE_RECORDER_EXITED: 'Bộ thu Intel thoát sớm. Xem recorder.log của phiên.',
  CAPTURE_NO_DATA: 'Chưa ghi được khung CSI. Kiểm tra recorder.log và ping.log của phiên.',
  CAPTURE_NO_RESULT: 'Phiên chưa có kết quả. Kiểm tra nhịp gói VHT20 và log suy luận.',
  EMPTY_REFERENCE_INSUFFICIENT: 'Chưa đủ dữ liệu nền. Để phòng trống rồi bấm Hiệu chuẩn phòng trống lại.',
  MODEL_LOAD_FAILED: 'Backend không nạp được mô hình.',
  MODEL_INFERENCE_FAILED: 'Backend gặp lỗi khi chạy suy luận.',
};
const SOURCE_LABELS = {live: 'Trực tiếp', replay: 'Phát lại', mock: 'Giả lập'};
const POSE_DESCRIPTIONS = {
  live: 'Tọa độ ước lượng từ CSI của phiên đang thu',
  replay: 'Tọa độ ước lượng từ bản ghi CSI',
  mock: 'Tọa độ mẫu · BlazePose',
};
const STARTABLE = new Set(['idle', 'ended', 'stopped', 'source_error']);
const MAX_RECONNECT_MS = 5000;
const durationLabel = seconds => seconds % 60 === 0 ? format.format(seconds / 60) + ' phút' : format.format(seconds) + ' giây';

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
let lastSignal = null;
let lastPose = null;
let lastView = view;
let drawTime = null;
let durationOptions = '';

function setText(id, value) {
  const target = element(id);
  if (target.textContent !== value) target.textContent = value;
}

function captureDurations(capture) {
  return capture?.automatic && Array.isArray(capture.duration_options_s) ? capture.duration_options_s : [];
}

function scheduleRender() {
  if (!animationFrame) animationFrame = requestAnimationFrame(render);
}

function receive(message) {
  const previousSession = stream.status?.session_id;
  if (!stream.accept(message, performance.now())) return;
  if (message.type === 'status' && message.session_id && previousSession !== message.session_id) {
    const seconds = String(message.capture?.duration_s);
    const range = element('range');
    if ([...range.options].some(option => option.value === seconds)) range.value = seconds;
  }
  scheduleRender();
}

function connect() {
  transport = 'connecting';
  stream.reset();
  scheduleRender();
  const url = new URL('/ws', location.href);
  url.protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
  url.searchParams.set('include_signal', 'true');
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
  const timeout = setTimeout(() => controller.abort(), action === 'stop' ? 8000 : 5000);
  element('request-error').hidden = true;
  scheduleRender();
  try {
    const url = new URL(`/api/session/${action}`, location.href);
    if (action === 'stop' && stream.status?.session_id) url.searchParams.set('session_id', stream.status.session_id);
    const duration = Number(element('capture-duration').value);
    if (action === 'start' && captureDurations(stream.status?.capture).includes(duration)) {
      url.searchParams.set('duration_s', duration);
    }
    const response = await fetch(url, {method: 'POST', signal: controller.signal});
    if (!response.ok) {
      throw new Error(response.status === 409 ? 'Trạng thái phiên đã thay đổi.' : `Backend trả lỗi HTTP ${response.status}.`);
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
  const signalView = view === 'signal';
  if (signalView ? snapshot.signalStale : snapshot.stale) {
    return ['Dữ liệu đã cũ', 'Đang đợi mẫu CSI mới.', 'warn'];
  }
  if (stream.status.status === 'ready' && !(signalView ? snapshot.signal : snapshot.chart)) {
    return ['Chờ cửa sổ CSI', 'Backend đang chuẩn bị mẫu tiếp theo.', 'warn'];
  }
  if (stream.status.reason === 'PREPARING_CAPTURE') {
    return ['Đang chuẩn bị bộ thu', 'Đang kiểm tra Wi-Fi, router và driver Intel.', 'warn'];
  }
  if (stream.status.empty_room?.calibrating) {
    return ['Đang hiệu chuẩn phòng trống', 'Giữ vùng thu trống đến hết 30 giây.', 'warn'];
  }
  if (stream.status.reason === 'EMPTY_REFERENCE_SAVED') {
    return ['Đã lưu nền phòng trống', 'Bấm Bắt đầu phiên để nhận diện.', 'neutral'];
  }
  if (stream.status.reason === 'CAPTURE_COMPLETE') {
    const duration = stream.status.capture?.duration_s;
    return [duration ? 'Đã thu xong ' + durationLabel(duration) : 'Đã thu xong', 'Bấm Bắt đầu phiên để thu lại.', 'neutral'];
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
  setText('source-tag', state?.capture?.automatic ? 'Thu tự động' : SOURCE_LABELS[source] ?? 'Chờ nguồn');
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
  const durations = captureDurations(capture);
  const durationSelect = element('capture-duration');
  const canSelectDuration = durations.length > 0;
  const optionsKey = [capture?.default_duration_s, ...durations].join(',');
  durationSelect.disabled = !canSelectDuration || !controlsEnabled || !startable;
  if (canSelectDuration && durationOptions !== optionsKey) {
    durationOptions = optionsKey;
    durationSelect.replaceChildren(...durations.map(seconds => new Option(durationLabel(seconds), seconds)));
    durationSelect.value = capture.default_duration_s;
  }
  if (canSelectDuration && state.session_id && !startable) durationSelect.value = capture.duration_s;
  let durationNote = '';
  if (source === 'replay') durationNote = 'Phát lại theo thời lượng của bản ghi.';
  if (source === 'live' && !canSelectDuration) {
    durationNote = capture?.automatic ? 'Cập nhật backend để chọn thời lượng thu.'
      : 'Mở backend với --source live --auto-capture để chọn thời lượng.';
  }
  setText('capture-duration-note', durationNote);
  element('capture-duration-note').hidden = !durationNote;
  element('start').disabled = !controlsEnabled || !startable;
  element('stop').disabled = !controlsEnabled || startable;
  element('calibrate-empty').disabled = !controlsEnabled || !startable || !capture?.automatic;
  element('calibrate-empty').hidden = source !== 'live';
  const reference = state?.empty_room;
  let referenceNote = '';
  if (source === 'live' && capture?.automatic) {
    referenceNote = reference?.available ? 'Đã lưu nền phòng trống cho cấu hình thu này.'
      : 'Để phòng trống và bấm Hiệu chuẩn phòng trống trong 30 giây trước khi kiểm tra hiện diện.';
    if (reference?.error) referenceNote = 'Nền đã lưu không dùng được. Hiệu chuẩn lại khi phòng trống.';
    if (reference?.calibrating) referenceNote = `Đang lấy nền. Giữ phòng trống · ${reference.collected_windows} cửa sổ hợp lệ.`;
  }
  setText('empty-room-note', referenceNote);
  element('empty-room-note').hidden = !referenceNote;
  setText('stream-status', phase[0]);
  element('stream-status').dataset.tone = phase[2];
}

function updateCaptureClock(now) {
  const status = stream.status;
  const capture = status?.capture;
  const clock = element('capture-clock');
  clock.hidden = !capture?.started_ms || STARTABLE.has(status.status);
  if (clock.hidden) return;
  const serverTime = status.server_sent_ms + now - stream.statusReceivedAt;
  const seconds = Math.ceil(Math.max(0, capture.started_ms + capture.duration_s * 1000 - serverTime) / 1000);
  const mm = String(Math.floor(seconds / 60)).padStart(2, '0');
  const ss = String(seconds % 60).padStart(2, '0');
  setText('capture-clock', seconds ? 'Còn ' + mm + ':' + ss : 'Đang kết thúc');
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
  const end = stream.points.at(-1)?.timestamp_ms;
  const points = end !== undefined
    ? stream.points.filter(point => point.timestamp_ms >= end - range * 1000)
    : [];
  setText('chart-description', mode === 'mock' ? 'Biên độ I/Q mẫu'
    : !chart && points.length ? 'Biên độ I/Q đã ghi trong phiên' : 'Biên độ I/Q thô trước chuẩn hóa');
  setText('amplitude', points.length ? format.format(points.at(-1).value) : '—');
  setText('sample-rate', chart ? sampleRate(chart.points) : sampleRate(points));
  setText('point-count', String(points.length));
  showTimestamp('chart-time', end ?? null);
  element('chart-placeholder').hidden = Boolean(points.length);
  setText('chart-empty-title', phase[0]);
  setText('chart-empty-note', phase[1]);

  const canvas = element('chart');
  const hover = paintChart(canvas, points, end ?? 0, range, pointer);
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
  const calibrating = stream.status?.empty_room?.calibrating;
  const needsReference = pose?.presenceReason === 'EMPTY_REFERENCE_REQUIRED';
  const pending = pose?.pending === true;
  const label = calibrating ? 'Đang lấy nền phòng trống'
    : needsReference ? 'Cần hiệu chuẩn phòng trống'
    : pending ? 'Đang xác nhận' : pose ? (pose.present ? 'Có người' : 'Phòng trống') : 'Chưa có kết quả';
  const motion = pose?.motion;
  const note = calibrating ? 'Giữ vùng thu trống đến khi thông báo đã lưu nền.'
    : needsReference ? 'Dừng phiên, để phòng trống rồi bấm Hiệu chuẩn phòng trống trong 30 giây.'
    : pose?.emptyRoom?.matched ? 'CSI khớp với nền phòng trống đã hiệu chuẩn.'
    : pending ? 'Đợi thêm cửa sổ CSI liên tiếp để xác nhận có người.' : pose
    ? (mode === 'mock' ? 'Trạng thái mẫu' : 'Dự đoán của mô hình từ CSI')
    : (chart ? 'Chờ kết quả cùng timestamp với biểu đồ.' : phase[1]);
  const joints = pose?.joints.size ?? 0;
  const probability = pose?.probability ?? null;
  setText('presence-state', label);
  setText('presence-note', note);
  element('verdict').dataset.tone = pending ? 'warn' : pose ? (pose.present ? 'ok' : 'neutral') : phase[2];
  setText('joint-count', pose && !pending ? `${joints}/${LANDMARK_COUNT}` : '—');
  setText('model-version', pose?.model ?? stream.status?.model_version ?? '—');
  setText('backend-latency', pose?.processingMs != null ? `${format.format(pose.processingMs)} ms` : '—');
  setText('motion-level', motion ? `${motion.level === 'high' ? 'Cao' : 'Thấp'} · ${format.format(motion.score)}` : '—');
  element('probability-group').hidden = probability === null;
  element('probability').value = probability ?? 0;
  setText('probability-value', probability === null ? '—' : `${format.format(probability * 100)}%`);
  setText('pose-description', POSE_DESCRIPTIONS[stream.status?.source] ?? 'Đang đợi dữ liệu CSI');
  setText('frame-size', pose?.frame ? `${pose.frame.width} × ${pose.frame.height}` : '—');
  showTimestamp('pose-time', pose?.timestamp ?? null);

  element('pose-placeholder').hidden = joints > 0;
  if (pose?.reason === 'CSI_OUTSIDE_POSE_CALIBRATION') {
    setText('pose-empty-title', 'CSI ngoài dải mô hình tư thế');
    const calibration = pose.calibration;
    setText('pose-empty-note', calibration
      ? `Độ lệch chuẩn hóa ${format.format(calibration.distance)}. Giới hạn ${format.format(calibration.limit)}.`
      : 'Tín hiệu CSI khác dữ liệu dùng để học tư thế 2D.');
  } else {
    setText('pose-empty-title', pose?.present && !joints ? 'Thiếu tọa độ khớp' : label);
    setText('pose-empty-note', pose && !pending && !pose.present ? 'Không có bộ xương ở cửa sổ này.' : note);
  }
  const canvas = element('skeleton');
  paintSkeleton(canvas, pose);
  canvas.setAttribute('aria-label', joints ? `Bộ xương 2D, ${joints} khớp` : 'Bộ xương 2D chưa có khớp');
}

function render() {
  animationFrame = 0;
  const snapshot = stream.snapshot(performance.now());
  const {chart, pose, signal} = snapshot;
  const phase = phaseFor(snapshot);
  updateHeader(phase);
  updateCaptureClock(performance.now());
  updateView();
  if (view === 'signal') renderSignal(signal, phase);
  else renderPresence(chart, pose, phase);

  const result = view === 'signal' ? signal : pose;
  const previous = view === 'signal' ? lastSignal : lastPose;
  if (view !== lastView) {
    drawTime = null;
  } else if (result !== previous) {
    drawTime = result ? performance.now() - result.receivedAt : null;
  }
  const arrivals = stream.chartArrivals;
  const duration = arrivals.at(-1) - arrivals[0];
  const refresh = signal && duration > 0 ? `${format.format(1000 * (arrivals.length - 1) / duration)} Hz cập nhật CSI` : null;
  setText('display-timing', [
    refresh, drawTime !== null ? `${format.format(drawTime)} ms từ nhận đến vẽ` : null,
  ].filter(Boolean).join(' · ') || '—');
  lastChart = chart;
  lastSignal = signal;
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
element('calibrate-empty').addEventListener('click', () => controlSession('calibrate-empty'));
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
  const now = performance.now();
  updateCaptureClock(now);
  const {chart, pose, signal} = stream.snapshot(now);
  if (chart !== lastChart || pose !== lastPose || signal !== lastSignal) scheduleRender();
}, 200);
route();
changeMode();
