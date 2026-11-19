import './styles.css';

const $ = (selector) => document.querySelector(selector);
const views = ['home', 'playground', 'docs'];
let connected = false;
let serverInfo = null;
let conversation = [];

function showView(name) {
  views.forEach((view) => {
    const section = document.getElementById('view-' + view);
    section.classList.toggle('active', view === name);
  });
  document.querySelectorAll('[data-view]').forEach((button) => {
    if (button.classList.contains('nav-link')) {
      button.classList.toggle('active', button.dataset.view === name);
    }
  });
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

document.querySelectorAll('[data-view]').forEach((button) => {
  button.addEventListener('click', (event) => {
    event.preventDefault();
    showView(button.dataset.view);
  });
});

function setConnectionState(ok, message) {
  const pill = $('#connection-pill');
  const result = $('#connection-result');
  connected = ok;
  pill.classList.toggle('connected', ok);
  pill.innerHTML = '<i></i> ' + (ok ? 'CONNECTED' : 'NOT CONNECTED');
  result.textContent = message;
  result.className = 'connection-result ' + (ok ? 'success' : 'error');
  $('#message-input').disabled = !ok;
  $('#send-button').disabled = !ok;
}

function apiBase() {
  const raw = $('#api-url').value.trim().replace(/\/+$/, '');
  if (!raw) throw new Error('Enter your Memer API base URL first.');
  const parsed = new URL(raw);
  if (!['http:', 'https:'].includes(parsed.protocol)) {
    throw new Error('The API URL must use HTTP or HTTPS.');
  }
  return parsed.toString().replace(/\/+$/, '');
}

$('#connect-button').addEventListener('click', async () => {
  let base;
  try {
    base = apiBase();
  } catch (error) {
    setConnectionState(false, error.message);
    return;
  }
  const button = $('#connect-button');
  button.disabled = true;
  button.textContent = 'Checking…';
  try {
    const response = await fetch(base + '/health', { method: 'GET' });
    if (!response.ok) throw new Error('Server returned HTTP ' + response.status);
    const data = await response.json();
    if (data.status !== 'ok') throw new Error('This endpoint did not identify itself as a healthy Memer Core API.');
    serverInfo = data;
    $('#server-service').textContent = data.service || 'Memer Core';
    $('#server-provider').textContent = data.provider || 'Unknown';
    $('#server-model').textContent = data.model || 'Unknown';
    $('#mock-warning').hidden = data.provider !== 'mock';
    setConnectionState(true, 'Connected to ' + (data.service || 'Memer Core') + '.');
  } catch (error) {
    serverInfo = null;
    $('#server-service').textContent = '—';
    $('#server-provider').textContent = '—';
    $('#server-model').textContent = '—';
    $('#mock-warning').hidden = true;
    const detail = error instanceof TypeError
      ? 'Could not reach the API. Check the address, server status, and MEMER_CORS_ORIGINS.'
      : error.message;
    setConnectionState(false, detail);
  } finally {
    button.disabled = false;
    button.innerHTML = 'Check connection <span>→</span>';
  }
});

function addMessage(role, content) {
  const area = $('#chat-messages');
  const empty = area.querySelector('.empty-state');
  if (empty) empty.remove();
  const message = document.createElement('div');
  message.className = 'message ' + role;
  const label = document.createElement('div');
  label.className = 'message-label';
  label.textContent = role === 'user' ? 'YOU' : 'MEMER API';
  const body = document.createElement('div');
  body.textContent = content;
  message.append(label, body);
  area.append(message);
  area.scrollTop = area.scrollHeight;
}

$('#chat-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  if (!connected || !serverInfo) return;
  const input = $('#message-input');
  const content = input.value.trim();
  if (!content) return;
  let base;
  try {
    base = apiBase();
  } catch (error) {
    setConnectionState(false, error.message);
    return;
  }

  conversation.push({ role: 'user', content });
  addMessage('user', content);
  input.value = '';
  input.disabled = true;
  $('#send-button').disabled = true;
  $('#send-button').textContent = 'Sending…';

  try {
    const response = await fetch(base + '/v1/chat/completions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: conversation })
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      const reason = typeof data.detail === 'string' ? data.detail : 'HTTP ' + response.status;
      throw new Error(reason);
    }
    if (typeof data.content !== 'string') throw new Error('The API response did not contain a text content field.');
    conversation.push({ role: 'assistant', content: data.content });
    addMessage('assistant', data.content);
  } catch (error) {
    conversation.pop();
    addMessage('assistant', 'Request failed: ' + error.message + '. Check your API server and CORS configuration, then try again.');
  } finally {
    input.disabled = false;
    $('#send-button').disabled = false;
    $('#send-button').textContent = 'Send ↗';
    input.focus();
  }
});

$('#message-input').addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault();
    $('#chat-form').requestSubmit();
  }
});

$('#clear-chat').addEventListener('click', () => {
  conversation = [];
  $('#chat-messages').innerHTML = '<div class="empty-state"><div class="empty-mark">M</div><h3>Your workspace is ready.</h3><p>Send a message to begin a new conversation.</p></div>';
});
