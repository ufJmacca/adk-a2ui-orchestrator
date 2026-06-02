const state = {
  surfaces: new Map(),
  textSurfaceIds: new Map(),
  nextTextSurfaceIndex: 1,
  activeSurfaceIdForPayload: null,
  activeDataContextForPayload: undefined,
};

const COMPONENT_RENDERERS = {
  Column: renderColumn,
  Row: renderRow,
  List: renderList,
  Text: renderText,
  Button: renderButton,
  TextField: renderTextField,
  Card: renderCard,
  Table: renderTable,
  Accordion: renderAccordion,
  Timeline: renderTimeline,
  Status: renderStatus,
};
const COMPONENT_RENDERER_KEYS = new Map(
  [
    ...Object.keys(COMPONENT_RENDERERS).map((key) => [key.toLowerCase(), key]),
    ['text-field', 'TextField'],
    ['text_field', 'TextField'],
  ],
);
const REQUIRED_PLAN_METADATA_KEYS = ['planId', 'planVersion', 'editedPlanVersion'];
const BLOCKED_PATH_SEGMENTS = new Set(['__proto__', 'constructor', 'prototype']);

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('request-form').addEventListener('submit', submitRequest);
  refreshStatus();
  refreshArtifacts();
});

async function submitRequest(event) {
  event.preventDefault();
  const input = document.getElementById('request-input').value;
  const response = await submitRequestPayload(input);
  if (response.httpOk !== false) {
    clearA2uiSurfaces();
  }
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
  let payload = {};
  try {
    payload = await response.json();
  } catch (error) {
    payload = {};
  }
  if (response.ok) {
    return payload;
  }
  return normalizeErrorResponse(response, payload);
}

function normalizeErrorResponse(response, payload) {
  const normalized = payload && typeof payload === 'object' && !Array.isArray(payload)
    ? { ...payload }
    : {};
  const errorEvent = statusEventForError(response.status, normalized.error);
  const statusEvents = Array.isArray(normalized.statusEvents)
    ? [...normalized.statusEvents]
    : [];
  if (errorEvent) {
    statusEvents.push(errorEvent);
  }
  return {
    ...normalized,
    status: normalized.status || 'error',
    statusEvents,
    httpOk: false,
    httpStatus: response.status,
  };
}

function statusEventForError(httpStatus, error) {
  if (error && typeof error === 'object' && !Array.isArray(error)) {
    return {
      status: error.code || 'request_failed',
      message: error.message || `Request failed with HTTP ${httpStatus}.`,
    };
  }
  if (typeof error === 'string' && error) {
    return {
      status: 'request_failed',
      message: error,
    };
  }
  return {
    status: 'request_failed',
    message: `Request failed with HTTP ${httpStatus}.`,
  };
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
  if (Object.prototype.hasOwnProperty.call(response, 'a2uiParts')) {
    renderA2uiParts(response.a2uiParts || []);
  }
  renderStatusUpdates(response.statusEvents || []);
  if (response.artifacts) {
    renderArtifacts(response.artifacts);
  }
  if (response.httpOk === false) {
    return;
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
  if (parts.length === 0) {
    return;
  }

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

function clearA2uiSurfaces() {
  for (const surface of state.surfaces.values()) {
    surface.element.remove();
  }
  state.surfaces.clear();
  document.getElementById('approval-surfaces').replaceChildren();
  document.getElementById('downstream-surfaces').replaceChildren();
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
  } else if (isApprovalSurfaceId(surfaceId)) {
    clearApprovalEditData(state.surfaces.get(surfaceId));
  }
}

function updateDataModel(message) {
  const surface = surfaceFor(message.surfaceId);
  if (!surface) {
    return;
  }
  applyDataModelUpdate(surface, message);
  renderSurface(message.surfaceId);
}

function applyDataModelUpdate(surface, message) {
  const hasPath = Object.prototype.hasOwnProperty.call(message, 'path');
  const hasValue = Object.prototype.hasOwnProperty.call(message, 'value');
  if (!hasPath && !hasValue) {
    surface.dataModel = message.data || {};
    return;
  }

  const parts = safePathParts(hasPath ? message.path : '/');
  if (parts === null) {
    return;
  }
  if (parts.length === 0) {
    surface.dataModel = hasValue ? message.value : {};
    return;
  }
  if (hasValue) {
    setValueAtPathParts(surface, parts, message.value);
    return;
  }
  deleteValueAtPathParts(surface, parts);
}

function updateComponents(message) {
  const surface = surfaceFor(message.surfaceId);
  if (!surface) {
    return;
  }
  if (isFullComponentReplacement(message)) {
    surface.components.clear();
  }
  for (const component of message.components || []) {
    if (component.id) {
      surface.components.set(component.id, component);
    }
  }
  renderSurface(message.surfaceId);
}

function isFullComponentReplacement(message) {
  return (
    message.replace === true
    || message.fullReplacement === true
    || message.mode === 'replace'
  );
}

function isApprovalSurfaceId(surfaceId) {
  return typeof surfaceId === 'string' && surfaceId.startsWith('surface_plan_');
}

function clearApprovalEditData(surface) {
  if (
    !surface
    || typeof surface.dataModel !== 'object'
    || Array.isArray(surface.dataModel)
  ) {
    return;
  }
  delete surface.dataModel.approvalEdits;
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

function renderComponent(
  surfaceId,
  component,
  dataContext = undefined,
  renderStack = new Set(),
) {
  const componentId = component && component.id;
  if (typeof componentId === 'string' && componentId) {
    if (renderStack.has(componentId)) {
      return renderRecursiveComponent(componentId);
    }
    renderStack.add(componentId);
  }
  const rendererKey = normalizedComponentRendererKey(component);
  const renderer = rendererKey ? COMPONENT_RENDERERS[rendererKey] : null;
  let rendered;
  if (renderer) {
    rendered = renderer(surfaceId, component, dataContext, renderStack);
  } else {
    rendered = renderUnsupportedComponent(component);
  }
  if (typeof componentId === 'string' && componentId) {
    renderStack.delete(componentId);
  }
  return rendered;
}

function normalizedComponentRendererKey(component) {
  const componentName = Object.prototype.hasOwnProperty.call(component, 'type')
    ? component.type
    : component.component;
  if (typeof componentName !== 'string') {
    return null;
  }
  return COMPONENT_RENDERER_KEYS.get(componentName.toLowerCase()) || null;
}

function renderChildren(
  surfaceId,
  childIds,
  dataContext = undefined,
  renderStack = new Set(),
) {
  const surface = surfaceFor(surfaceId);
  if (!surface) {
    return [];
  }

  if (isChildTemplate(childIds)) {
    const template = surface.components.get(childIds.componentId);
    const items = valueAtPath(surfaceId, childIds.path, dataContext);
    if (!template || !Array.isArray(items)) {
      return [];
    }
    return items.map((item) => (
      renderComponent(surfaceId, template, item, renderStack)
    ));
  }

  if (!Array.isArray(childIds)) {
    return [];
  }

  return childIds
    .map((childId) => (
      typeof childId === 'string'
        ? surface.components.get(childId)
        : childId
    ))
    .filter(Boolean)
    .map((component) => (
      renderComponent(surfaceId, component, dataContext, renderStack)
    ));
}

function isChildTemplate(value) {
  return (
    value
    && typeof value === 'object'
    && !Array.isArray(value)
    && typeof value.componentId === 'string'
    && typeof value.path === 'string'
  );
}

function renderColumn(
  surfaceId,
  component,
  dataContext = undefined,
  renderStack = new Set(),
) {
  const element = document.createElement('div');
  element.className = 'a2ui-column';
  element.replaceChildren(
    ...renderChildren(surfaceId, component.children || [], dataContext, renderStack),
  );
  return element;
}

function renderRow(
  surfaceId,
  component,
  dataContext = undefined,
  renderStack = new Set(),
) {
  const element = document.createElement('div');
  element.className = 'a2ui-row';
  element.replaceChildren(
    ...renderChildren(surfaceId, component.children || [], dataContext, renderStack),
  );
  return element;
}

function renderList(
  surfaceId,
  component,
  dataContext = undefined,
  renderStack = new Set(),
) {
  const element = document.createElement('ul');
  element.className = 'a2ui-list';
  for (const child of renderChildren(
    surfaceId,
    component.children || [],
    dataContext,
    renderStack,
  )) {
    const item = document.createElement('li');
    item.appendChild(child);
    element.appendChild(item);
  }
  return element;
}

function renderText(surfaceId, component, dataContext = undefined) {
  const variant = component.variant || 'body';
  const element = document.createElement(variant === 'h2' || variant === 'h3' ? 'h3' : 'p');
  element.className = variant === 'h2' ? 'a2ui-text-h2' : variant === 'h3' ? 'a2ui-text-h3' : 'a2ui-text';
  const text = resolveBoundValue(surfaceId, component.text, dataContext);
  element.textContent = String(text ?? '');
  return element;
}

function renderButton(
  surfaceId,
  component,
  dataContext = undefined,
  renderStack = new Set(),
) {
  const button = document.createElement('button');
  button.type = 'button';
  button.dataset.variant = component.variant || 'default';
  const label = component.child
    ? renderChildren(surfaceId, [component.child], dataContext, renderStack)[0]
    : document.createTextNode(component.label || 'Action');
  button.replaceChildren(label);
  const context = component.action && component.action.event
    ? component.action.event.context
    : null;
  if (context) {
    button.addEventListener('click', () => (
      postUserAction(buildUserAction(context, surfaceId, dataContext))
    ));
  }
  return button;
}

function renderTextField(surfaceId, component, dataContext = undefined) {
  const wrapper = document.createElement('label');
  wrapper.className = 'a2ui-field';
  const label = document.createElement('span');
  label.textContent = component.label || component.id || 'Field';
  const input = component.variant === 'longText'
    ? document.createElement('textarea')
    : document.createElement('input');
  const path = component.value && component.value.path;
  if (path) {
    input.value = String(valueAtPath(surfaceId, path, dataContext) ?? '');
    input.addEventListener('input', () => (
      setValueAtPath(surfaceId, path, input.value, dataContext)
    ));
  }
  wrapper.replaceChildren(label, input);
  return wrapper;
}

function renderCard(
  surfaceId,
  component,
  dataContext = undefined,
  renderStack = new Set(),
) {
  const element = document.createElement('article');
  element.className = 'a2ui-card';
  const children = [];
  if (hasRenderableValue(component.title)) {
    const title = document.createElement('h3');
    title.className = 'a2ui-card-title';
    title.textContent = String(component.title);
    children.push(title);
  }
  if (hasRenderableValue(component.body)) {
    const body = document.createElement('p');
    body.className = 'a2ui-card-body';
    body.textContent = String(component.body);
    children.push(body);
  }
  if (component.child) {
    children.push(
      ...renderChildren(surfaceId, [component.child], dataContext, renderStack),
    );
  }
  if (children.length > 0) {
    element.replaceChildren(...children);
  }
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

function renderAccordion(
  surfaceId,
  component,
  dataContext = undefined,
  renderStack = new Set(),
) {
  const details = document.createElement('details');
  const summary = document.createElement('summary');
  summary.textContent = component.title || 'Details';
  details.appendChild(summary);
  for (const child of renderChildren(
    surfaceId,
    component.children || [],
    dataContext,
    renderStack,
  )) {
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

function renderRecursiveComponent(componentId) {
  const element = document.createElement('p');
  element.className = 'a2ui-text';
  element.textContent = `Circular component reference: ${componentId}`;
  return element;
}

function buildUserAction(context, surfaceId, dataContext = undefined) {
  requireActionSurfaceContextMatchesRenderedSurface(context, surfaceId);
  state.activeSurfaceIdForPayload = surfaceId;
  state.activeDataContextForPayload = dataContext;
  try {
    requirePlanMetadataForPlanAction(context, surfaceId);
    const userAction = {
      type: context.type,
      surfaceId,
      payload: resolvePayload(context.payload || {}),
    };
    return { userAction };
  } finally {
    state.activeSurfaceIdForPayload = null;
    state.activeDataContextForPayload = undefined;
  }
}

function requireActionSurfaceContextMatchesRenderedSurface(context, surfaceId) {
  if (context.surfaceId !== surfaceId) {
    throw new Error('A2UI action surfaceId does not match rendered surface');
  }
}

function requirePlanMetadataForPlanAction(context, surfaceId) {
  if (!surfaceId || !surfaceId.startsWith('surface_plan_')) {
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
    if (Object.keys(value).length === 1 && typeof value.path === 'string') {
      return valueAtPath(
        state.activeSurfaceIdForPayload,
        value.path,
        state.activeDataContextForPayload,
      );
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

function resolveBoundValue(surfaceId, value, dataContext = undefined) {
  if (
    value
    && typeof value === 'object'
    && !Array.isArray(value)
    && Object.keys(value).length === 1
    && typeof value.path === 'string'
  ) {
    return valueAtPath(surfaceId, value.path, dataContext);
  }
  return value;
}

function valueAtPath(surfaceId, path, dataContext = undefined) {
  const surface = surfaceFor(surfaceId);
  if (!surface) {
    return undefined;
  }
  const parts = safePathParts(path);
  if (parts === null) {
    return undefined;
  }
  let current = dataContext !== undefined && !path.startsWith('/')
    ? dataContext
    : surface.dataModel;
  for (const part of parts) {
    if (!current || typeof current !== 'object') {
      return undefined;
    }
    current = current[part];
  }
  return current;
}

function setValueAtPath(surfaceId, path, value, dataContext = undefined) {
  const surface = surfaceFor(surfaceId);
  if (!surface) {
    return;
  }
  const parts = safePathParts(path);
  if (parts === null || parts.length === 0) {
    return;
  }
  if (dataContext !== undefined && !path.startsWith('/')) {
    if (!dataContext || typeof dataContext !== 'object') {
      return;
    }
    setValueAtPathPartsOnObject(dataContext, parts, value);
    return;
  }
  setValueAtPathParts(surface, parts, value);
}

function setValueAtPathParts(surface, parts, value) {
  if (
    !surface.dataModel
    || typeof surface.dataModel !== 'object'
  ) {
    surface.dataModel = {};
  }
  setValueAtPathPartsOnObject(surface.dataModel, parts, value);
}

function setValueAtPathPartsOnObject(target, parts, value) {
  let current = target;
  for (const part of parts.slice(0, -1)) {
    if (!current[part] || typeof current[part] !== 'object') {
      current[part] = {};
    }
    current = current[part];
  }
  current[parts[parts.length - 1]] = value;
}

function deleteValueAtPathParts(surface, parts) {
  let current = surface.dataModel;
  for (const part of parts.slice(0, -1)) {
    if (!current || typeof current !== 'object') {
      return;
    }
    current = current[part];
  }
  if (!current || typeof current !== 'object') {
    return;
  }
  const lastPart = parts[parts.length - 1];
  if (Array.isArray(current) && /^\d+$/.test(lastPart)) {
    current.splice(Number(lastPart), 1);
    return;
  }
  delete current[lastPart];
}

function safePathParts(path) {
  if (typeof path !== 'string') {
    return null;
  }
  const parts = path.split('/').filter(Boolean).map(decodeJsonPointerPart);
  if (parts.some((part) => BLOCKED_PATH_SEGMENTS.has(part))) {
    return null;
  }
  return parts;
}

function decodeJsonPointerPart(part) {
  return part.replace(/~1/g, '/').replace(/~0/g, '~');
}
