import assert from 'node:assert/strict';
import test from 'node:test';
import {demoMessages} from '../dashboard/demo.js';
import {CsiStream, MAX_HISTORY_MS, MAX_POINTS, QUANTITY, STALE_MS, segmentSeries} from '../dashboard/stream.js';

const T = 1_790_000_000_000;
const envelope = (type, sequence, fields = {}) => ({
  type, sequence, schema_version: 1, session_id: 'session-a', source: 'replay', ...fields,
});
const status = (sequence, state, fields) => envelope('status', sequence, {status: state, ...fields});
const chart = (sequence, timestamp, fields) => envelope('chart', sequence, {
  timestamp_ms: timestamp, quantity: QUANTITY,
  points: [{timestamp_ms: timestamp - 100, value: 111}, {timestamp_ms: timestamp, value: 112}],
  ...fields,
});
const pose = (sequence, timestamp, fields) => envelope('pose', sequence, {
  status: 'ready', timestamp_ms: timestamp, person_present: true, presence_probability: .82,
  coordinate_system: {reference_width: 1920, reference_height: 1080, origin: 'top_left'},
  keypoints: [{id: 15, x_norm: .7, y_norm: .2}], ...fields,
});

test('joins only the current session and exact chart timestamp in either arrival order', () => {
  const stream = new CsiStream();
  stream.accept(status(1, 'ready'), 0);
  stream.accept(pose(2, T), 10);
  assert.equal(stream.snapshot(11).pose, null);
  stream.accept(chart(3, T), 15);
  assert.equal(stream.snapshot(16).pose.timestamp, T);

  stream.accept(chart(4, T + 200), 200);
  assert.equal(stream.snapshot(201).pose, null);
  assert.equal(stream.accept(pose(5, T + 200, {session_id: 'previous'}), 202), false);
  assert.equal(stream.accept(pose(6, T), 203), false);
  stream.accept(pose(7, T + 200), 205);
  assert.equal(stream.snapshot(206).pose.timestamp, T + 200);
  assert.equal(stream.accept(chart(7, T + 300), 210), false);
  assert.equal(stream.accept(chart(8, T + 200), 211), false);
});

test('recognizes empty predictions and never fills missing joints', () => {
  const stream = new CsiStream();
  stream.accept(status(1, 'ready'), 0);
  stream.accept(chart(2, T), 5);
  stream.accept(pose(3, T, {
    keypoints: [
      {id: 15, x_norm: .7, y_norm: .2},
      {id: 16, x_pixel: 480, y_pixel: 540},
      {id: 13, x_pixel: null, y_pixel: 80},
      {id: 14, x_norm: -.1, y_norm: .1},
    ],
  }), 6);
  assert.deepEqual([...stream.snapshot(7).pose.joints], [
    [15, {x: .7, y: .2}], [16, {x: .25, y: .5}],
  ]);
  stream.accept(chart(4, T + 200), 200);
  stream.accept(pose(5, T + 200, {person_present: false}), 201);
  assert.equal(stream.snapshot(202).pose.present, false);
  assert.equal(stream.snapshot(202).pose.joints.size, 0);
});

test('clears results on data loss, stale receipt, stop, and reconnect', () => {
  const stream = new CsiStream();
  stream.accept(status(9, 'ready'), 10);
  stream.accept(chart(10, T), 20);
  stream.accept(pose(11, T), 25);
  assert.equal(stream.snapshot(19 + STALE_MS).pose.present, true);
  assert.deepEqual(stream.snapshot(20 + STALE_MS), {stale: true, chart: null, pose: null});
  stream.accept(status(12, 'model_error'), 30);
  assert.deepEqual(stream.snapshot(32), {stale: false, chart: null, pose: null});
  assert.equal(stream.accept(pose(13, T + 200), 40), false);
  stream.accept(status(14, 'stopped'), 50);
  assert.equal(stream.points.length, 0);
  stream.reset();
  stream.accept(status(0, 'ready', {session_id: 'session-b'}), 60);
  assert.equal(stream.accept(chart(15, T), 62), false);
  assert.equal(stream.snapshot(63).pose, null);
});

test('deduplicates overlapping windows and separates both timestamp and backend gaps', () => {
  const stream = new CsiStream();
  stream.accept(status(1, 'ready'), 0);
  stream.accept(chart(2, T), 5);
  stream.accept(chart(3, T + 100), 100);
  assert.equal(stream.points.length, 3);
  stream.accept(status(4, 'insufficient_data', {reason: 'CSI_GAP'}), 200);
  assert.equal(stream.snapshot(200).chart, null);
  stream.accept(status(5, 'ready'), 201);
  stream.accept(chart(6, T + 300), 202);
  stream.accept(chart(7, T + 900), 800);
  assert.deepEqual(segmentSeries(stream.points).map(points => points.length), [3, 2, 2]);
});

test('retains bounded history during continuous input', () => {
  const stream = new CsiStream();
  stream.accept(status(1, 'ready'), 0);
  for (let index = 1; index <= 1000; index += 1) {
    stream.accept(chart(index + 1, T + index * 100), index * 10);
  }
  assert.ok(stream.points.length <= MAX_POINTS);
  assert.ok(stream.points.at(-1).timestamp_ms - stream.points[0].timestamp_ms <= MAX_HISTORY_MS);
  assert.ok(stream.chartArrivals.length <= 25);
});

test('demo visibly identifies its source and uses the subject’s left arm', () => {
  const stream = new CsiStream();
  const stop = demoMessages(message => stream.accept(message, 0), 'present');
  stop();
  const {chart: sample, pose: prediction} = stream.snapshot(1);
  assert.equal(stream.status.source, 'mock');
  assert.equal(sample.timestamp, prediction.timestamp);
  assert.equal(prediction.joints.size, 33);
  assert.ok(prediction.joints.get(15).y < prediction.joints.get(11).y);
  assert.ok(prediction.joints.get(16).y > prediction.joints.get(12).y);
});
