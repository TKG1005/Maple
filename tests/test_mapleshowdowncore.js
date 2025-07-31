#!/usr/bin/env node
const { spawn } = require('child_process');
const readline = require('readline');

// Path to the IPC Battle Server
const SERVER_PATH = 'pokemon-showdown/sim/ipc-battle-server.js';

// Define test cases
const tests = [
  {
    name: 'ping',
    send: { type: 'ping' },
    validate: (msg) => msg.type === 'pong' && msg.success === true
  },
  {
    name: 'unknown',
    send: { type: 'foobar' },
    validate: (msg) => msg.type === 'error' && msg.error_type === 'UNKNOWN_MESSAGE_TYPE'
  }
];

// Spawn the server process
const server = spawn('node', [SERVER_PATH], { stdio: ['pipe', 'pipe', 'inherit'] });

const rl = readline.createInterface({ input: server.stdout });
let currentTest = 0;
let passed = 0;

console.log('Running MapleShowdownCore IPC tests...');

// Send next test or finish
function runNext() {
  if (currentTest >= tests.length) {
    console.log(`\nTests completed: ${passed}/${tests.length} passed.`);
    // Clean up and exit
    rl.close();
    server.kill();
    process.exit(0);
  }
  const test = tests[currentTest];
  console.log(`\n[Test ${currentTest + 1}] ${test.name}`);
  server.stdin.write(JSON.stringify(test.send) + '\n');
}

// Listen for responses
rl.on('line', (line) => {
  let msg;
  try {
    msg = JSON.parse(line);
  } catch (e) {
    console.error('Invalid JSON from server:', line);
    process.exit(1);
  }
  const test = tests[currentTest];
  const ok = test.validate(msg);
  if (ok) {
    console.log(`  ✓ ${test.name} passed`);
    passed++;
  } else {
    console.log(`  ✗ ${test.name} failed`);
    console.log('    Received:', msg);
  }
  currentTest++;
  runNext();
});

// Wait briefly before starting
setTimeout(runNext, 100);