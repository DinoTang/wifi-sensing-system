import {segmentSeries} from './stream.js';

const COLORS = {
  grid: '#25343f', axis: '#a4b3bf', teal: '#75e5be', amber: '#f5c078', center: '#dceef0',
};

const LEFT_BONES = [
  [0, 1], [1, 2], [2, 3], [3, 7], [11, 13], [13, 15], [15, 17], [15, 19], [15, 21],
  [17, 19], [23, 25], [25, 27], [27, 29], [27, 31], [29, 31],
];
const RIGHT_BONES = [
  [0, 4], [4, 5], [5, 6], [6, 8], [12, 14], [14, 16], [16, 18], [16, 20], [16, 22],
  [18, 20], [24, 26], [26, 28], [28, 30], [28, 32], [30, 32],
];
const BODY_BONES = [[9, 10], [11, 12], [11, 23], [12, 24], [23, 24]];
const LEFT_JOINTS = new Set([1, 2, 3, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]);

function prepareCanvas(canvas) {
  const {width, height} = canvas.getBoundingClientRect();
  if (width < 1 || height < 1) return null;

  const scale = Math.min(window.devicePixelRatio || 1, 2);
  const pixelWidth = Math.round(width * scale);
  const pixelHeight = Math.round(height * scale);
  if (canvas.width !== pixelWidth) canvas.width = pixelWidth;
  if (canvas.height !== pixelHeight) canvas.height = pixelHeight;
  const context = canvas.getContext('2d');
  context.setTransform(scale, 0, 0, scale, 0, 0);
  context.clearRect(0, 0, width, height);
  context.font = '11px "IBM Plex Mono", Consolas, monospace';
  context.lineWidth = 1;
  context.lineCap = 'round';
  context.lineJoin = 'round';
  return {context, width, height};
}

function line(context, x1, y1, x2, y2) {
  context.beginPath();
  context.moveTo(x1, y1);
  context.lineTo(x2, y2);
  context.stroke();
}

function dot(context, x, y, radius, color) {
  context.fillStyle = color;
  context.beginPath();
  context.arc(x, y, radius, 0, 2 * Math.PI);
  context.fill();
}

function yScale(points) {
  if (!points.length) return {min: 0, max: 1, step: .25};
  const values = points.map(point => point.value);
  const low = Math.min(...values);
  const high = Math.max(...values);
  const padding = Math.max(1, (high - low) * .14);
  const proposed = (high - low + 2 * padding) / 4;
  const unit = 10 ** Math.floor(Math.log10(proposed));
  const step = ([1, 2, 5, 10].find(factor => factor * unit >= proposed) ?? 10) * unit;
  const min = Math.max(0, Math.floor((low - padding) / step) * step);
  const max = Math.ceil((high + padding) / step) * step;
  return {min, max, step};
}

export function paintChart(canvas, points, end, range, pointer) {
  const stage = prepareCanvas(canvas);
  if (!stage) return null;
  const {context, width, height} = stage;
  const left = width < 420 ? 43 : 54;
  const right = width - 17;
  const top = 18;
  const bottom = height - 30;
  const {min, max, step} = yScale(points);
  const x = time => right - ((end - time) / (range * 1000)) * (right - left);
  const y = value => bottom - (value - min) / (max - min) * (bottom - top);

  context.strokeStyle = COLORS.grid;
  context.fillStyle = COLORS.axis;
  context.textAlign = 'right';
  context.textBaseline = 'middle';
  const tickCount = Math.round((max - min) / step);
  for (let index = 0; index <= tickCount; index += 1) {
    const value = min + step * index;
    const level = y(value);
    line(context, left, level, right, level);
    context.fillText(points.length ? value.toLocaleString('vi-VN', {maximumFractionDigits: 2}) : '·', left - 8, level);
  }
  for (let index = 0; index <= 5; index += 1) {
    const seconds = -range + range * index / 5;
    const position = left + (right - left) * index / 5;
    line(context, position, top, position, bottom);
    context.textAlign = index === 0 ? 'left' : index === 5 ? 'right' : 'center';
    context.fillText(`${seconds} s`, position, bottom + 15);
  }
  if (!points.length) return null;

  const area = context.createLinearGradient(0, top, 0, bottom);
  area.addColorStop(0, 'rgba(117, 229, 190, .21)');
  area.addColorStop(1, 'rgba(117, 229, 190, .015)');
  context.save();
  context.beginPath();
  context.rect(left - 5, top - 5, right - left + 10, bottom - top + 10);
  context.clip();
  for (const segment of segmentSeries(points)) {
    if (segment.length === 1) {
      dot(context, x(segment[0].timestamp_ms), y(segment[0].value), 2, COLORS.teal);
      continue;
    }
    const first = segment[0];
    const last = segment.at(-1);
    context.beginPath();
    context.moveTo(x(first.timestamp_ms), y(first.value));
    for (const point of segment.slice(1)) context.lineTo(x(point.timestamp_ms), y(point.value));
    context.strokeStyle = COLORS.teal;
    context.lineWidth = 1.8;
    context.stroke();
    context.lineTo(x(last.timestamp_ms), bottom);
    context.lineTo(x(first.timestamp_ms), bottom);
    context.closePath();
    context.fillStyle = area;
    context.fill();
  }
  const last = points.at(-1);
  dot(context, x(last.timestamp_ms), y(last.value), 5, 'rgba(117, 229, 190, .24)');
  dot(context, x(last.timestamp_ms), y(last.value), 2.7, COLORS.teal);

  let hover = null;
  if (pointer !== null && pointer >= left && pointer <= right) {
    let nearest = null;
    let distance = 17;
    for (const point of points) {
      const difference = Math.abs(x(point.timestamp_ms) - pointer);
      if (difference < distance) {
        nearest = point;
        distance = difference;
      }
    }
    if (nearest) {
      const px = x(nearest.timestamp_ms);
      context.setLineDash([3, 4]);
      context.strokeStyle = COLORS.axis;
      context.lineWidth = 1;
      line(context, px, top, px, bottom);
      context.setLineDash([]);
      dot(context, px, y(nearest.value), 3, COLORS.center);
      hover = {point: nearest, x: px};
    }
  }
  context.restore();
  return hover;
}

export function paintSkeleton(canvas, pose) {
  const stage = prepareCanvas(canvas);
  if (!stage) return;
  const {context, width, height} = stage;
  const frame = pose?.frame ?? {width: 1920, height: 1080};
  const scale = Math.min((width - 28) / frame.width, (height - 34) / frame.height);
  const frameWidth = frame.width * scale;
  const frameHeight = frame.height * scale;
  const left = (width - frameWidth) / 2;
  const top = (height - frameHeight) / 2;
  const x = value => left + value * frameWidth;
  const y = value => top + value * frameHeight;

  context.fillStyle = '#121e25';
  context.fillRect(left, top, frameWidth, frameHeight);
  context.strokeStyle = COLORS.grid;
  for (let index = 1; index < 8; index += 1) {
    const fraction = index / 8;
    line(context, x(fraction), top, x(fraction), y(1));
    line(context, left, y(fraction), x(1), y(fraction));
  }
  context.strokeStyle = '#354a57';
  context.strokeRect(left, top, frameWidth, frameHeight);
  if (!pose?.joints.size) return;

  context.fillStyle = COLORS.axis;
  context.textAlign = 'left';
  context.textBaseline = 'bottom';
  context.fillText('0,0', left, top - 3);
  context.textBaseline = 'top';
  context.textAlign = 'right';
  context.fillText(`${frame.width} × ${frame.height}`, x(1), y(1) + 3);

  context.lineWidth = 2.1;
  for (const [edges, color] of [
    [BODY_BONES, COLORS.center], [LEFT_BONES, COLORS.teal], [RIGHT_BONES, COLORS.amber],
  ]) {
    context.strokeStyle = color;
    for (const [first, second] of edges) {
      const a = pose.joints.get(first);
      const b = pose.joints.get(second);
      if (a && b) line(context, x(a.x), y(a.y), x(b.x), y(b.y));
    }
  }
  for (const [id, point] of pose.joints) {
    const color = id === 0 ? COLORS.center : LEFT_JOINTS.has(id) ? COLORS.teal : COLORS.amber;
    const radius = id < 11 ? 2 : 3.2;
    dot(context, x(point.x), y(point.y), radius + 1.3, '#111920');
    dot(context, x(point.x), y(point.y), radius, color);
  }
}
