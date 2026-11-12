import {QUANTITY} from './stream.js';

// Example BlazePose coordinates with the subject's left hand raised.
const JOINTS = [
  [.50, .25], [.515, .232], [.523, .231], [.53, .231],
  [.485, .232], [.477, .231], [.47, .231], [.544, .246], [.456, .246],
  [.515, .277], [.485, .277], [.57, .36], [.43, .36],
  [.642, .31], [.383, .51], [.65, .18], [.409, .64],
  [.667, .143], [.395, .662], [.653, .134], [.405, .675],
  [.631, .151], [.419, .659], [.543, .60], [.457, .60],
  [.55, .74], [.443, .74], [.565, .865], [.426, .865],
  [.576, .888], [.412, .888], [.553, .905], [.435, .905],
];

export function demoMessages(emit, state) {
  let sequence = 0;
  const base = Date.now();
  const envelope = (type, timestamp_ms) => ({
    type, timestamp_ms, sequence: ++sequence, session_id: 'demo',
    schema_version: 1, source: 'mock', model_version: null, server_sent_ms: Date.now(),
  });
  const status = state === 'waiting' ? 'warming_up' : state === 'error' ? 'model_error' : 'ready';
  emit({...envelope('status', null), status, reason: state === 'error' ? 'MODEL_INFERENCE_FAILED' : null});
  if (status !== 'ready') return () => {};

  const present = state === 'present';
  const step = 100;
  let target = base;
  const update = () => {
    const points = Array.from({length: 20}, (_, index) => {
      const timestamp_ms = target - (19 - index) * step;
      const time = (timestamp_ms - base) / 1000;
      const variation = present ? 7 * Math.sin(time * 2.3) + 4 * Math.sin(time * 9) : Math.sin(time * 4);
      return {timestamp_ms, value: 140 + variation};
    });
    emit({...envelope('chart', target), quantity: QUANTITY, points});
    emit({
      ...envelope('pose', target), status: 'ready', person_present: present,
      coordinate_system: {origin: 'top_left', reference_width: 1920, reference_height: 1080},
      keypoints: present ? JOINTS.map(([x_norm, y_norm], id) => ({id, x_norm, y_norm})) : [],
    });
    target += 2 * step;
  };
  update();
  const timer = setInterval(update, 2 * step);
  return () => clearInterval(timer);
}
