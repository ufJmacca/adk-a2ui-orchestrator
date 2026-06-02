const state = {
  surfaces: new Map(),
  textSurfaceIds: new Map(),
  nextTextSurfaceIndex: 1,
  activeSurfaceIdForPayload: null,
};

const COMPONENT_RENDERERS = {
  Column: renderColumn,
  Row: renderRow,
  Text: renderText,
  Button: renderButton,
  TextField: renderTextField,
  Card: renderCard,
  Table: renderTable,
  Accordion: renderAccordion,
  Timeline: renderTimeline,
  Status: renderStatus,
};
const COMPONENT_NAMES = {
  accordion: 'Accordion',
  button: 'Button',
  card: 'Card',
  column: 'Column',
  row: 'Row',
  status: 'Status',
  table: 'Table',
  text: 'Text',
  'text-field': 'TextField',
  textfield: 'TextField',
  text_field: 'TextField',
  textField: 'TextField',
  timeline: 'Timeline',
};
const REQUIRED_PLAN_METADATA_KEYS = ['planId', 'planVersion', 'editedPlanVersion'];

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('request-form').addEventListener('submit', submitRequest);
  refreshStatus();
  refreshArtifacts();
});

async function submitRequest(event) {
  event.preventDefault();
  const input = document.getElementById('request-input').value;
  const response = await submitRequestPayload(input);
  handleTransportResponse(response);
}

async function postUserAction(eventPayload) {
  const response = await submitUserActionPayload(eventPayload);
  handleTransportResponse(response);
}

async function submitRequestPayload(input) {
  const response = await fetch('/api/request', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ input }),
  });
  return readJsonResponse(response);
}

async function submitUserActionPayload(eventPayload) {
  const response = await fetch('/api/user-action', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(eventPayload),
  });
  return readJsonResponse(response);
}

async function readJsonResponse(response) {
  if (!response.ok) {
    throw new Error(`Request failed: ${response.status}`);
  }
  return response.json();
}

async function refreshStatus() {
  const response = await fetch('/api/status', { headers: { accept: 'application/json' } });
  if (!response.ok) {
    return;
  }
  const snapshot = await response.json();
  renderStatusUpdates(snapshot.statusEvents || []);
}

async function refreshArtifacts() {
  const response = await fetch('/api/artifacts', { headers: { accept: 'application/json' } });
  if (!response.ok) {
    return;
  }
  const snapshot = await response.json();
  renderArtifacts(snapshot.artifacts || {});
  renderA2uiParts(snapshot.a2uiParts || []);
}

function handleTransportResponse(response) {
  renderA2uiParts(response.a2uiParts || []);
  renderStatusUpdates(response.statusEvents || []);
  if (response.artifacts) {
    renderArtifacts(response.artifacts);
  }
  refreshStatus();
  refreshArtifacts();
}

function renderStatusUpdates(statusEvents) {
  const list = document.getElementById('status-updates');
  if (statusEvents.length === 0) {
    return;
  }
  const items = statusEvents.map((statusEvent) => {
    const item = document.createElement('li');
    item.textContent = `${statusEvent.status}: ${statusEvent.message}`;
    return item;
  });
  list.replaceChildren(...items);
}

function renderArtifacts(artifacts) {
  const output = document.getElementById('artifact-list');
  output.textContent = JSON.stringify(artifacts, null, 2);
}

function renderA2uiParts(parts) {
  for (const part of parts) {
    if (part.type === 'text' && typeof part.text === 'string') {
      renderTextPart(part);
      continue;
    }
    const payload = part.data || part;
    if (payload.createSurface) {
      createSurface(payload.createSurface);
    }
    if (payload.updateDataModel) {
      updateDataModel(payload.updateDataModel);
    }
    if (payload.updateComponents) {
      updateComponents(payload.updateComponents);
    }
    if (payload.deleteSurface) {
      deleteSurface(payload.deleteSurface.surfaceId);
    }
  }
}

function renderTextPart(part) {
  const text = part.text;
  if (!text.trim()) {
    return;
  }
  const region = document.getElementById('downstream-surfaces');
  const surfaceId = textPartSurfaceId(part);
  const existing = document.getElementById(surfaceId);
  const element = existing || document.createElement('section');
  element.id = surfaceId;
  element.className = 'a2ui-surface a2ui-text-fallback';
  element.dataset.surfaceId = surfaceId;

  const body = document.createElement('p');
  body.className = 'a2ui-text';
  body.textContent = text;

  const diagnostic = part.metadata && part.metadata.developerDiagnostic;
  const diagnosticText = diagnostic ? JSON.stringify(diagnostic, null, 2) : '';
  if (diagnosticText) {
    const details = document.createElement('pre');
    details.className = 'a2ui-diagnostic';
    details.textContent = diagnosticText;
    element.replaceChildren(body, details);
  } else {
    element.replaceChildren(body);
  }

  if (!existing) {
    region.appendChild(element);
  }
}

function textPartSurfaceId(part) {
  const key = JSON.stringify({
    text: part.text,
    metadata: part.metadata || {},
  });
  if (!state.textSurfaceIds.has(key)) {
    state.textSurfaceIds.set(
      key,
      `surface_text_fallback_${state.nextTextSurfaceIndex}`,
    );
    state.nextTextSurfaceIndex += 1;
  }
  return state.textSurfaceIds.get(key);
}

function createSurface(message) {
  const surfaceId = message.surfaceId;
  if (!surfaceId) {
    return;
  }
  const region = surfaceId.startsWith('surface_plan_')
    ? document.getElementById('approval-surfaces')
    : document.getElementById('downstream-surfaces');
  const existing = document.getElementById(surfaceId);
  const element = existing || document.createElement('section');
  element.id = surfaceId;
  element.className = 'a2ui-surface';
  element.dataset.surfaceId = surfaceId;
  if (!existing) {
    region.appendChild(element);
  }
  if (existing && surfaceId.startsWith('surface_plan_')) {
    state.surfaces.set(surfaceId, {
      element,
      components: new Map(),
      dataModel: {},
    });
    return;
  }
  if (!state.surfaces.has(surfaceId)) {
    state.surfaces.set(surfaceId, {
      element,
      components: new Map(),
      dataModel: {},
    });
  }
}

function updateDataModel(message) {
  const surface = surfaceFor(message.surfaceId);
  if (!surface) {
    return;
  }
  surface.dataModel = message.data || {};
}

function updateComponents(message) {
  const surface = surfaceFor(message.surfaceId);
  if (!surface) {
    return;
  }
  surface.components.clear();
  for (const component of message.components || []) {
    if (component.id) {
      surface.components.set(component.id, component);
    }
  }
  renderSurface(message.surfaceId);
}

function deleteSurface(surfaceId) {
  const surface = state.surfaces.get(surfaceId);
  if (surface) {
    surface.element.remove();
  }
  state.surfaces.delete(surfaceId);
}

function renderSurface(surfaceId) {
  const surface = surfaceFor(surfaceId);
  if (!surface) {
    return;
  }
  const root = surface.components.get('root') || Array.from(surface.components.values())[0];
  if (!root) {
    surface.element.replaceChildren();
    return;
  }
  surface.element.replaceChildren(renderComponent(surfaceId, root));
}

function renderComponent(surfaceId, component) {
  const componentName = normalizedComponentName(component);
  const renderer = COMPONENT_RENDERERS[componentName];
  if (!renderer) {
    return renderUnsupportedComponent(component);
  }
  return renderer(surfaceId, component);
}

function normalizedComponentName(component) {
  const rawName = component.component ?? component.type ?? '';
  if (typeof rawName !== 'string') {
    return '';
  }
  return COMPONENT_NAMES[rawName] || COMPONENT_NAMES[rawName.toLowerCase()] || rawName;
}

function renderChildren(surfaceId, children) {
  const surface = surfaceFor(surfaceId);
  if (!surface) {
    return [];
  }
  const childList = Array.isArray(children) ? children : children ? [children] : [];
  return childList
    .map((child) => {
      if (typeof child === 'string') {
        return surface.components.get(child);
      }
      if (child && typeof child === 'object') {
        if (typeof child.componentId === 'string') {
          return surface.components.get(child.componentId);
        }
        return child;
      }
      return null;
    })
    .filter(Boolean)
    .map((component) => renderComponent(surfaceId, component));
}

function renderColumn(surfaceId, component) {
  const element = document.createElement('div');
  element.className = 'a2ui-column';
  element.replaceChildren(...renderChildren(surfaceId, component.children || []));
  return element;
}

function renderRow(surfaceId, component) {
  const element = document.createElement('div');
  element.className = 'a2ui-row';
  element.replaceChildren(...renderChildren(surfaceId, component.children || []));
  return element;
}

function renderText(_surfaceId, component) {
  const variant = component.variant || 'body';
  const element = document.createElement(variant === 'h2' || variant === 'h3' ? 'h3' : 'p');
  element.className = variant === 'h2' ? 'a2ui-text-h2' : variant === 'h3' ? 'a2ui-text-h3' : 'a2ui-text';
  element.textContent = String(component.text || '');
  return element;
}

function renderButton(surfaceId, component) {
  const button = document.createElement('button');
  button.type = 'button';
  button.dataset.variant = component.variant || 'default';
  const label = component.child
    ? renderChildren(surfaceId, [component.child])[0]
    : document.createTextNode(component.label || 'Action');
  button.replaceChildren(label);
  const context = component.action && component.action.event
    ? component.action.event.context
    : null;
  if (context) {
    button.addEventListener('click', () => postUserAction(buildUserAction(context, surfaceId)));
  }
  return button;
}

function renderTextField(surfaceId, component) {
  const wrapper = document.createElement('label');
  wrapper.className = 'a2ui-field';
  const label = document.createElement('span');
  label.textContent = component.label || component.id || 'Field';
  const input = component.variant === 'longText'
    ? document.createElement('textarea')
    : document.createElement('input');
  const path = component.value && component.value.path;
  if (path) {
    input.value = String(valueAtPath(surfaceId, path) || '');
    input.addEventListener('input', () => setValueAtPath(surfaceId, path, input.value));
  }
  wrapper.replaceChildren(label, input);
  return wrapper;
}

function renderCard(surfaceId, component) {
  const element = document.createElement('article');
  element.className = 'a2ui-card';
  const cardChildren = [];
  if (hasRenderableValue(component.title)) {
    const title = document.createElement('h3');
    title.className = 'a2ui-card-title';
    title.textContent = String(component.title);
    cardChildren.push(title);
  }
  if (hasRenderableValue(component.body)) {
    const body = document.createElement('p');
    body.className = 'a2ui-card-body';
    body.textContent = String(component.body);
    cardChildren.push(body);
  }
  if (component.child) {
    cardChildren.push(...renderChildren(surfaceId, [component.child]));
  }
  element.replaceChildren(...cardChildren);
  return element;
}

function hasRenderableValue(value) {
  return value !== undefined && value !== null && String(value) !== '';
}

function renderTable(_surfaceId, component) {
  const table = document.createElement('table');
  table.className = 'a2ui-table';
  const rows = component.rows || [];
  if (Array.isArray(component.columns)) {
    const head = document.createElement('thead');
    const row = document.createElement('tr');
    for (const column of component.columns) {
      const cell = document.createElement('th');
      cell.textContent = String(column.label || column.key || '');
      row.appendChild(cell);
    }
    head.appendChild(row);
    table.appendChild(head);
  }
  const body = document.createElement('tbody');
  for (const rowData of rows) {
    const row = document.createElement('tr');
    const values = Array.isArray(component.columns)
      ? component.columns.map((column) => rowData[column.key])
      : Object.values(rowData);
    for (const value of values) {
      const cell = document.createElement('td');
      cell.textContent = String(value ?? '');
      row.appendChild(cell);
    }
    body.appendChild(row);
  }
  table.appendChild(body);
  return table;
}

function renderAccordion(surfaceId, component) {
  const details = document.createElement('details');
  const summary = document.createElement('summary');
  summary.textContent = component.title || 'Details';
  details.appendChild(summary);
  for (const child of renderChildren(surfaceId, component.children || [])) {
    details.appendChild(child);
  }
  return details;
}

function renderTimeline(_surfaceId, component) {
  const list = document.createElement('ol');
  for (const item of component.items || []) {
    const entry = document.createElement('li');
    entry.textContent = `${item.label || item.title || 'Item'} ${item.detail || ''}`.trim();
    list.appendChild(entry);
  }
  return list;
}

function renderStatus(_surfaceId, component) {
  const element = document.createElement('div');
  element.className = 'a2ui-status';
  element.textContent = component.text || component.message || component.status || 'Status';
  return element;
}

function renderUnsupportedComponent(component) {
  const element = document.createElement('p');
  element.className = 'a2ui-text';
  element.textContent = `Unsupported component: ${component.component || component.type || 'unknown'}`;
  return element;
}

function buildUserAction(context, surfaceId) {
  requireActionSurfaceContextMatchesRenderedSurface(context, surfaceId);
  state.activeSurfaceIdForPayload = surfaceId;
  try {
    requirePlanMetadataForPlanAction(context);
    const userAction = {
      type: context.type,
      surfaceId,
      payload: resolvePayload(context.payload || {}),
    };
    return { userAction };
  } finally {
    state.activeSurfaceIdForPayload = null;
  }
}

function requireActionSurfaceContextMatchesRenderedSurface(context, surfaceId) {
  if (context.surfaceId !== surfaceId) {
    throw new Error('A2UI action surfaceId does not match rendered surface');
  }
}

function requirePlanMetadataForPlanAction(context) {
  if (!context.surfaceId || !context.surfaceId.startsWith('surface_plan_')) {
    return;
  }
  const payload = context.payload || {};
  const missingKey = REQUIRED_PLAN_METADATA_KEYS.find((key) => payload[key] === undefined);
  if (context.type !== 'reject_plan' && missingKey) {
    throw new Error(`Plan action is missing ${missingKey}`);
  }
  if (context.type === 'reject_plan' && payload.planId === undefined) {
    throw new Error('Plan rejection is missing planId');
  }
}

function resolvePayload(value) {
  if (Array.isArray(value)) {
    return value.map((item) => resolvePayload(item));
  }
  if (value && typeof value === 'object') {
    if (typeof value.path === 'string') {
      return valueAtPath(state.activeSurfaceIdForPayload, value.path);
    }
    const resolved = {};
    for (const [key, childValue] of Object.entries(value)) {
      resolved[key] = resolvePayload(childValue);
    }
    return resolved;
  }
  return value;
}

function surfaceFor(surfaceId) {
  if (!surfaceId) {
    return null;
  }
  return state.surfaces.get(surfaceId) || null;
}

function valueAtPath(surfaceId, path) {
  const surface = surfaceFor(surfaceId);
  if (!surface) {
    return undefined;
  }
  const parts = path.split('/').filter(Boolean);
  let current = surface.dataModel;
  for (const part of parts) {
    if (!current || typeof current !== 'object') {
      return undefined;
    }
    current = current[part];
  }
  return current;
}

function setValueAtPath(surfaceId, path, value) {
  const surface = surfaceFor(surfaceId);
  if (!surface) {
    return;
  }
  const parts = path.split('/').filter(Boolean);
  let current = surface.dataModel;
  for (const part of parts.slice(0, -1)) {
    if (!current[part] || typeof current[part] !== 'object') {
      current[part] = {};
    }
    current = current[part];
  }
  current[parts[parts.length - 1]] = value;
}
