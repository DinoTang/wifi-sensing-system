export const STALE_MS = 2_500;
export const MAX_GAP_MS = 300;
export const MAX_HISTORY_MS = 20_000;
export const MAX_POINTS = 300;
export const LANDMARK_COUNT = 33;
export const QUANTITY = 'mean_raw_csi_amplitude';

const SOURCES = new Set(['live', 'replay', 'mock']);
const STATES = new Set([
  'idle', 'starting', 'warming_up', 'ready', 'insufficient_data', 'incompatible',
  'disconnected', 'model_error', 'source_error', 'ended', 'stopped',
]);
const MAX_PENDING_POSES = 8;

const isTimestamp = value => Number.isSafeInteger(value) && value >= 0 && value <= 8.64e15;
const inUnitInterval = value => Number.isFinite(value) && value >= 0 && value <= 1;

function chartPoints(message) {
  if (message.quantity !== QUANTITY || !Array.isArray(message.points)
    || !message.points.length || message.points.length > MAX_POINTS) return null;

  let previous = -1;
  for (const point of message.points) {
    if (!point || !isTimestamp(point.timestamp_ms) || point.timestamp_ms <= previous
      || !Number.isFinite(point.value) || point.value < 0) return null;
    previous = point.timestamp_ms;
  }
  return previous === message.timestamp_ms ? message.points : null;
}

function readPose(message) {
  if (message.status !== 'ready' || typeof message.person_present !== 'boolean') return null;

  const coordinates = message.coordinate_system;
  const frame = coordinates?.origin === 'top_left'
    && Number.isFinite(coordinates.reference_width) && coordinates.reference_width > 0
    && Number.isFinite(coordinates.reference_height) && coordinates.reference_height > 0
    ? {width: coordinates.reference_width, height: coordinates.reference_height}
    : null;
  const joints = new Map();
  if (message.person_present && frame && Array.isArray(message.keypoints)) {
    for (const joint of message.keypoints.slice(0, LANDMARK_COUNT)) {
      if (!joint || !Number.isInteger(joint.id) || joint.id < 0
        || joint.id >= LANDMARK_COUNT || joints.has(joint.id)) continue;
      const x = joint.x_norm ?? (Number.isFinite(joint.x_pixel) ? joint.x_pixel / frame.width : NaN);
      const y = joint.y_norm ?? (Number.isFinite(joint.y_pixel) ? joint.y_pixel / frame.height : NaN);
      if (inUnitInterval(x) && inUnitInterval(y)) joints.set(joint.id, {x, y});
    }
  }
  const motion = message.motion;
  return {
    present: message.person_present,
    probability: inUnitInterval(message.presence_probability) ? message.presence_probability : null,
    frame,
    joints,
    motion: Number.isFinite(motion?.score) && motion.score >= 0 && ['low', 'high'].includes(motion.level)
      ? {score: motion.score, level: motion.level} : null,
    processingMs: Number.isFinite(message.timing_ms?.backend) && message.timing_ms.backend >= 0
      ? message.timing_ms.backend : null,
    model: typeof message.model_version === 'string' ? message.model_version : null,
  };
}

export class CsiStream {
  constructor() {
    this.reset();
  }

  reset() {
    this.sequence = -1;
    this.status = null;
    this.segment = 0;
    this.clearWindow();
  }

  clearWindow(keepHistory = false) {
    this.chart = null;
    this.poses = new Map();
    this.lastPoseTimestamp = -1;
    this.chartArrivals = [];
    this.segment += 1;
    if (!keepHistory) this.points = [];
  }

  accept(message, receivedAt) {
    if (!message || message.schema_version !== 1 || !SOURCES.has(message.source)
      || !Number.isSafeInteger(message.sequence) || message.sequence <= this.sequence
      || (message.session_id !== null && typeof message.session_id !== 'string')) return false;

    if (message.type === 'status') {
      if (!STATES.has(message.status)) return false;
      const sameSession = this.status?.session_id === message.session_id
        && this.status.source === message.source;
      if (!sameSession || message.status !== 'ready') {
        const retain = sameSession && message.status === 'insufficient_data'
          && (message.reason === 'CSI_GAP' || message.reason === 'CSI_CADENCE');
        this.clearWindow(retain);
      }
      this.status = message;
    } else {
      if (this.status?.status !== 'ready' || !message.session_id
        || message.session_id !== this.status.session_id || message.source !== this.status.source
        || !isTimestamp(message.timestamp_ms)) return false;

      if (message.type === 'chart') {
        if (!this.acceptChart(message, receivedAt)) return false;
      } else if (message.type === 'pose') {
        if (!this.acceptPose(message, receivedAt)) return false;
      } else {
        return false;
      }
    }
    this.sequence = message.sequence;
    return true;
  }

  acceptChart(message, receivedAt) {
    const timestamp = message.timestamp_ms;
    if (timestamp <= (this.chart?.timestamp ?? -1)) return false;
    const points = chartPoints(message);
    if (!points) return false;

    const previous = this.points.at(-1)?.timestamp_ms ?? -1;
    for (const point of points) {
      if (point.timestamp_ms > previous) this.points.push({...point, segment: this.segment});
    }
    this.points = this.points.filter(point => point.timestamp_ms >= timestamp - MAX_HISTORY_MS)
      .slice(-MAX_POINTS);
    this.chart = {timestamp, receivedAt, points};
    this.chartArrivals = this.chartArrivals.filter(time => receivedAt - time < 5000).slice(-24);
    this.chartArrivals.push(receivedAt);
    for (const time of this.poses.keys()) {
      if (time < timestamp) this.poses.delete(time);
    }
    return true;
  }

  acceptPose(message, receivedAt) {
    const timestamp = message.timestamp_ms;
    if (timestamp <= this.lastPoseTimestamp || timestamp < (this.chart?.timestamp ?? -1)) return false;
    const pose = readPose(message);
    if (!pose) return false;

    this.lastPoseTimestamp = timestamp;
    this.poses.set(timestamp, {...pose, receivedAt, timestamp});
    if (this.poses.size > MAX_PENDING_POSES) this.poses.delete(this.poses.keys().next().value);
    return true;
  }

  snapshot(now) {
    const stale = this.chart !== null && now - this.chart.receivedAt >= STALE_MS;
    const chart = stale ? null : this.chart;
    const match = chart ? this.poses.get(chart.timestamp) : null;
    const pose = match && now - match.receivedAt < STALE_MS ? match : null;
    return {stale, chart, pose};
  }
}

export function segmentSeries(points) {
  const segments = [];
  let previous = null;
  for (const point of points) {
    if (!previous || point.segment !== previous.segment
      || point.timestamp_ms - previous.timestamp_ms > MAX_GAP_MS) segments.push([]);
    segments.at(-1).push(point);
    previous = point;
  }
  return segments;
}
