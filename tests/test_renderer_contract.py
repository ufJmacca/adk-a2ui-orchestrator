from __future__ import annotations

import json
from pathlib import Path
import subprocess
import textwrap

from orchestrator_demo.a2ui_support.approval_canvas import build_approval_canvas
from orchestrator_demo.contracts import AgentDescriptor, ExecutionPlan, PlanStep


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = REPOSITORY_ROOT / "orchestrator_demo" / "app" / "static"


def _descriptor(agent_id: str, display_name: str) -> AgentDescriptor:
    return AgentDescriptor(
        agent_id=agent_id,
        display_name=display_name,
        capabilities=[agent_id.replace("_", " ")],
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        a2ui_catalogs=["basic"],
        routing_examples=[],
        execution_mode="local_llm",
    )


def _approval_plan() -> ExecutionPlan:
    return ExecutionPlan(
        plan_id="plan_renderer_behavior",
        objective="Prepare me for tomorrow's meeting with ABC Manufacturing.",
        detected_intents=["meeting_prep", "relationship_summary"],
        selected_agents=["relationship_summary", "synthesis"],
        steps=[
            PlanStep(
                step_id="step_relationship_summary",
                agent_id="relationship_summary",
                instruction="Summarize the relationship history.",
                expected_output="Relationship history and open follow-ups.",
                data_source_categories=["relationship_history"],
                parallel_group="meeting_context",
            ),
            PlanStep(
                step_id="step_synthesis",
                agent_id="synthesis",
                instruction="Create the RM-ready meeting brief.",
                depends_on=["step_relationship_summary"],
                expected_output="Final meeting preparation brief.",
                data_source_categories=["specialist_outputs"],
            ),
        ],
        data_source_categories=["relationship_history"],
        risk_notes=["Synthetic demo data only."],
        approval_surface_id="surface_plan_renderer_behavior",
        plan_version=3,
    )


def _renderer_behavior_response() -> dict[str, object]:
    plan = _approval_plan()
    approval_messages = build_approval_canvas(
        plan,
        agent_descriptors=[
            _descriptor("relationship_summary", "Relationship Summary Agent"),
            _descriptor("synthesis", "Synthesis Agent"),
        ],
    )
    downstream_messages = [
        {
            "version": "0.1",
            "createSurface": {
                "surfaceId": "surface_product_opportunity_1",
                "catalogId": "basic",
            },
        },
        {
            "version": "0.1",
            "updateComponents": {
                "surfaceId": "surface_product_opportunity_1",
                "components": [
                    {
                        "component": "Column",
                        "id": "root",
                        "children": [
                            "downstream_text",
                            "downstream_card",
                            "downstream_table",
                            "downstream_button",
                        ],
                    },
                    {
                        "component": "Text",
                        "id": "downstream_text",
                        "text": (
                            "Product recommendation "
                            "<script>globalThis.__executed = true</script>"
                        ),
                    },
                    {
                        "component": "Card",
                        "id": "downstream_card",
                        "title": "Relationship signal",
                        "body": "Treasury services are relevant to this prospect.",
                    },
                    {
                        "component": "Table",
                        "id": "downstream_table",
                        "columns": [
                            {"key": "balance", "label": "Balance"},
                            {"key": "active", "label": "Active"},
                        ],
                        "rows": [
                            {"balance": 0, "active": False},
                        ],
                    },
                    {
                        "component": "Button",
                        "id": "downstream_button",
                        "label": "Show detail",
                        "action": {
                            "event": {
                                "name": "specialist_action",
                                "context": {
                                    "type": "specialist_action",
                                    "surfaceId": "surface_product_opportunity_stale",
                                    "payload": {
                                        "action": "show_more_detail",
                                        "path": "synthetic://product-opportunity/source",
                                    },
                                },
                            }
                        },
                    },
                ],
            },
        },
    ]
    return {
        "a2uiParts": [
            {"data": message} for message in [*approval_messages, *downstream_messages]
        ],
        "statusEvents": [
            {
                "status": "approval_required",
                "message": "Plan approval is pending.",
                "taskId": "task_renderer_1",
                "planId": plan.plan_id,
            },
            {
                "status": "step_completed",
                "message": "Relationship summary completed.",
                "taskId": "task_renderer_1",
                "planId": plan.plan_id,
            },
        ],
        "artifacts": {
            "final_response": {
                "agent_id": "synthesis",
                "content": "Draft meeting brief ready.",
            }
        },
    }


def test_renderer_static_shell_exposes_required_local_endpoints() -> None:
    # Arrange
    html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
    script = (STATIC_ROOT / "renderer.js").read_text(encoding="utf-8")

    # Act
    visible_surfaces = [
        'id="approval-surfaces"',
        'id="status-updates"',
        'id="artifact-list"',
        'id="downstream-surfaces"',
    ]

    # Assert
    for element_id in visible_surfaces:
        assert element_id in html
    assert 'src="/static/renderer.js"' in html
    assert "fetch('/api/request'" in script
    assert "fetch('/api/user-action'" in script
    assert "fetch('/api/status'" in script
    assert "fetch('/api/artifacts'" in script


def test_renderer_surfaces_non_ok_user_action_json_payload(tmp_path: Path) -> None:
    # Arrange
    node_script = tmp_path / "renderer_non_ok_user_action_test.cjs"
    node_script.write_text(
        textwrap.dedent(
            f"""
            const assert = require('node:assert/strict');
            const fs = require('node:fs');
            const vm = require('node:vm');

            class MiniNode {{
              constructor() {{
                this.parentNode = null;
              }}

              get textContent() {{
                return '';
              }}
            }}

            class MiniTextNode extends MiniNode {{
              constructor(text) {{
                super();
                this.text = String(text);
              }}

              get textContent() {{
                return this.text;
              }}

              set textContent(value) {{
                this.text = String(value);
              }}
            }}

            class MiniElement extends MiniNode {{
              constructor(tagName) {{
                super();
                this.nodeType = 1;
                this.tagName = tagName.toUpperCase();
                this.children = [];
                this.dataset = {{}};
                this.className = '';
                this.id = '';
                this.storedText = '';
              }}

              get textContent() {{
                if (this.children.length > 0) {{
                  return this.children.map((child) => child.textContent).join('');
                }}
                return this.storedText;
              }}

              set textContent(value) {{
                this.storedText = String(value);
                this.children = [];
              }}

              appendChild(child) {{
                child.parentNode = this;
                this.children.push(child);
                return child;
              }}

              replaceChildren(...children) {{
                this.children = [];
                this.storedText = '';
                for (const child of children) {{
                  this.appendChild(child);
                }}
              }}

              remove() {{
                if (!this.parentNode) {{
                  return;
                }}
                this.parentNode.children = this.parentNode.children.filter(
                  (child) => child !== this,
                );
                this.parentNode = null;
              }}

              addEventListener() {{}}
            }}

            class MiniDocument extends MiniElement {{
              constructor() {{
                super('#document');
                this.body = new MiniElement('body');
                this.appendChild(this.body);
              }}

              createElement(tagName) {{
                return new MiniElement(tagName);
              }}

              createTextNode(text) {{
                return new MiniTextNode(text);
              }}

              getElementById(id) {{
                let found = null;
                const visit = (node) => {{
                  if (found || node.nodeType !== 1) {{
                    return;
                  }}
                  if (node.id === id) {{
                    found = node;
                    return;
                  }}
                  for (const child of node.children) {{
                    visit(child);
                  }}
                }};
                visit(this);
                return found;
              }}
            }}

            function mount(document, tagName, id) {{
              const element = document.createElement(tagName);
              element.id = id;
              document.body.appendChild(element);
              return element;
            }}

            async function main() {{
              const rendererPath = {json.dumps(str(STATIC_ROOT / "renderer.js"))};
              const document = new MiniDocument();
              mount(document, 'form', 'request-form');
              mount(document, 'textarea', 'request-input');
              mount(document, 'section', 'approval-surfaces');
              mount(document, 'ol', 'status-updates');
              mount(document, 'pre', 'artifact-list');
              mount(document, 'section', 'downstream-surfaces');

              const retrySurfaceParts = [
                {{
                  data: {{
                    version: '0.1',
                    createSurface: {{
                      surfaceId: 'surface_plan_retry',
                      catalogId: 'basic',
                    }},
                  }},
                }},
                {{
                  data: {{
                    version: '0.1',
                    updateComponents: {{
                      surfaceId: 'surface_plan_retry',
                      components: [
                        {{
                          component: 'Text',
                          id: 'root',
                          text: 'Retryable approval controls',
                        }},
                      ],
                    }},
                  }},
                }},
              ];
              let statusRefreshed = false;
              const fetch = async (url) => {{
                if (url === '/api/user-action') {{
                  return {{
                    ok: false,
                    status: 500,
                    json: async () => ({{
                      status: 'error',
                      error: {{
                        code: 'graph_execution_failed',
                        message: 'Transient specialist failure.',
                        retryable: true,
                      }},
                      statusEvents: [
                        {{
                          status: 'step_failed',
                          message: 'Relationship summary failed.',
                        }},
                      ],
                      artifacts: {{
                        diagnostic: 'Approval can be retried after editing.',
                      }},
                    }}),
                  }};
                }}
                if (url === '/api/status') {{
                  statusRefreshed = true;
                  return {{
                    ok: true,
                    json: async () => ({{
                      statusEvents: [
                        {{ status: 'stale', message: 'Stale server status.' }},
                      ],
                    }}),
                  }};
                }}
                if (url === '/api/artifacts') {{
                  return {{
                    ok: true,
                    json: async () => ({{ artifacts: {{}}, a2uiParts: retrySurfaceParts }}),
                  }};
                }}
                throw new Error(`Unexpected fetch ${{url}}`);
              }};

              const context = {{
                document,
                fetch,
                console,
                Map,
                Error,
                JSON,
                String,
                Array,
                Object,
                Promise,
              }};
              context.window = context;
              context.globalThis = context;
              vm.createContext(context);
              const source = fs.readFileSync(rendererPath, 'utf8')
                + '\\nthis.__rendererApi = {{ handleTransportResponse, postUserAction }};';
              vm.runInContext(source, context, {{ filename: 'renderer.js' }});

              context.__rendererApi.handleTransportResponse({{
                a2uiParts: retrySurfaceParts,
                statusEvents: [],
                artifacts: {{}},
              }});
              await new Promise((resolve) => setImmediate(resolve));
              statusRefreshed = false;

              await context.__rendererApi.postUserAction({{
                userAction: {{
                  type: 'approve_plan',
                  surfaceId: 'surface_plan_retry',
                  payload: {{ planId: 'plan_retry', planVersion: 1 }},
                }},
              }});
              await new Promise((resolve) => setImmediate(resolve));

              const statusText = document.getElementById('status-updates').textContent;
              assert.match(statusText, /step_failed: Relationship summary failed/);
              assert.match(statusText, /graph_execution_failed: Transient specialist failure/);
              assert.match(
                document.getElementById('artifact-list').textContent,
                /Approval can be retried after editing/,
              );
              assert.match(
                document.getElementById('surface_plan_retry').textContent,
                /Retryable approval controls/,
              );
              assert.equal(statusRefreshed, false);
            }}

            main().catch((error) => {{
              console.error(error);
              process.exit(1);
            }});
            """
        ),
        encoding="utf-8",
    )

    # Act
    completed = subprocess.run(
        ["node", str(node_script)],
        check=False,
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )

    # Assert
    assert completed.returncode == 0, completed.stderr


def test_renderer_rejects_prototype_polluting_binding_paths(tmp_path: Path) -> None:
    # Arrange
    node_script = tmp_path / "renderer_binding_path_test.cjs"
    node_script.write_text(
        textwrap.dedent(
            f"""
            const assert = require('node:assert/strict');
            const fs = require('node:fs');
            const vm = require('node:vm');

            const context = {{
              document: {{ addEventListener() {{}} }},
              console,
              Map,
              Object,
              Set,
            }};
            context.window = context;
            context.globalThis = context;
            vm.createContext(context);

            const source = fs.readFileSync(
              {json.dumps(str(STATIC_ROOT / "renderer.js"))},
              'utf8',
            ) + '\\nthis.__rendererApi = {{ state, valueAtPath, setValueAtPath }};';
            vm.runInContext(source, context, {{ filename: 'renderer.js' }});

            const api = context.__rendererApi;
            api.state.surfaces.set('surface_dangerous_path', {{ dataModel: {{}} }});

            Object.prototype.pollutedRead = 'prototype value';
            try {{
              assert.equal(
                api.valueAtPath(
                  'surface_dangerous_path',
                  '__proto__/pollutedRead',
                ),
                undefined,
              );
            }} finally {{
              delete Object.prototype.pollutedRead;
            }}

            api.setValueAtPath(
              'surface_dangerous_path',
              '__proto__/pollutedWrite',
              'owned',
            );

            assert.equal(Object.prototype.pollutedWrite, undefined);
            assert.deepEqual(
              api.state.surfaces.get('surface_dangerous_path').dataModel,
              {{}},
            );
            """
        ),
        encoding="utf-8",
    )

    # Act
    completed = subprocess.run(
        ["node", str(node_script)],
        check=False,
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )

    # Assert
    assert completed.returncode == 0, completed.stderr


def test_renderer_guards_recursive_component_references(tmp_path: Path) -> None:
    # Arrange
    node_script = tmp_path / "renderer_recursive_component_test.cjs"
    node_script.write_text(
        textwrap.dedent(
            f"""
            const assert = require('node:assert/strict');
            const fs = require('node:fs');
            const vm = require('node:vm');

            class MiniNode {{
              constructor() {{
                this.parentNode = null;
              }}

              get textContent() {{
                return '';
              }}
            }}

            class MiniElement extends MiniNode {{
              constructor(tagName) {{
                super();
                this.nodeType = 1;
                this.tagName = tagName.toUpperCase();
                this.children = [];
                this.dataset = {{}};
                this.className = '';
                this.id = '';
                this.storedText = '';
              }}

              get textContent() {{
                if (this.children.length > 0) {{
                  return this.children.map((child) => child.textContent).join('');
                }}
                return this.storedText;
              }}

              set textContent(value) {{
                this.storedText = String(value);
                this.children = [];
              }}

              appendChild(child) {{
                child.parentNode = this;
                this.children.push(child);
                return child;
              }}

              replaceChildren(...children) {{
                this.children = [];
                this.storedText = '';
                for (const child of children) {{
                  this.appendChild(child);
                }}
              }}

              remove() {{
                if (!this.parentNode) {{
                  return;
                }}
                this.parentNode.children = this.parentNode.children.filter(
                  (child) => child !== this,
                );
                this.parentNode = null;
              }}

              addEventListener() {{}}
            }}

            class MiniDocument extends MiniElement {{
              constructor() {{
                super('#document');
                this.body = new MiniElement('body');
                this.appendChild(this.body);
              }}

              createElement(tagName) {{
                return new MiniElement(tagName);
              }}

              createTextNode(text) {{
                const node = new MiniNode();
                node.text = String(text);
                Object.defineProperty(node, 'textContent', {{
                  get() {{
                    return this.text;
                  }},
                  set(value) {{
                    this.text = String(value);
                  }},
                }});
                return node;
              }}

              getElementById(id) {{
                let found = null;
                const visit = (node) => {{
                  if (found || node.nodeType !== 1) {{
                    return;
                  }}
                  if (node.id === id) {{
                    found = node;
                    return;
                  }}
                  for (const child of node.children) {{
                    visit(child);
                  }}
                }};
                visit(this);
                return found;
              }}
            }}

            function mount(document, tagName, id) {{
              const element = document.createElement(tagName);
              element.id = id;
              document.body.appendChild(element);
              return element;
            }}

            const rendererPath = {json.dumps(str(STATIC_ROOT / "renderer.js"))};
            const document = new MiniDocument();
            mount(document, 'form', 'request-form');
            mount(document, 'textarea', 'request-input');
            mount(document, 'section', 'approval-surfaces');
            mount(document, 'ol', 'status-updates');
            mount(document, 'pre', 'artifact-list');
            mount(document, 'section', 'downstream-surfaces');

            const context = {{
              document,
              console,
              Map,
              Set,
              Error,
              JSON,
              String,
              Array,
              Object,
            }};
            context.window = context;
            context.globalThis = context;
            vm.createContext(context);

            const source = fs.readFileSync(rendererPath, 'utf8')
              + '\\nthis.__rendererApi = {{ handleTransportResponse }};';
            vm.runInContext(source, context, {{ filename: 'renderer.js' }});

            context.__rendererApi.handleTransportResponse({{
              httpOk: false,
              a2uiParts: [
                {{
                  data: {{
                    version: '0.1',
                    createSurface: {{
                      surfaceId: 'surface_recursive_component',
                      catalogId: 'basic',
                    }},
                  }},
                }},
                {{
                  data: {{
                    version: '0.1',
                    updateComponents: {{
                      surfaceId: 'surface_recursive_component',
                      components: [
                        {{
                          component: 'Column',
                          id: 'root',
                          children: ['root'],
                        }},
                      ],
                    }},
                  }},
                }},
              ],
              statusEvents: [],
              artifacts: {{}},
            }});

            const surface = document.getElementById('surface_recursive_component');
            assert.match(surface.textContent, /Circular component reference: root/);
            """
        ),
        encoding="utf-8",
    )

    # Act
    completed = subprocess.run(
        ["node", str(node_script)],
        check=False,
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )

    # Assert
    assert completed.returncode == 0, completed.stderr


def test_renderer_behaves_as_trusted_dom_mapper_for_a2ui_loop(tmp_path: Path) -> None:
    # Arrange
    response = _renderer_behavior_response()
    node_script = tmp_path / "renderer_behavior_test.cjs"
    node_script.write_text(
        textwrap.dedent(
            f"""
            const assert = require('node:assert/strict');
            const fs = require('node:fs');
            const vm = require('node:vm');

            class MiniNode {{
              constructor() {{
                this.parentNode = null;
              }}

              get textContent() {{
                return '';
              }}

              querySelectorAll() {{
                return [];
              }}

              querySelector() {{
                return null;
              }}
            }}

            class MiniTextNode extends MiniNode {{
              constructor(text) {{
                super();
                this.nodeType = 3;
                this.text = String(text);
              }}

              get textContent() {{
                return this.text;
              }}

              set textContent(value) {{
                this.text = String(value);
              }}
            }}

            class MiniElement extends MiniNode {{
              constructor(tagName) {{
                super();
                this.nodeType = 1;
                this.tagName = tagName.toUpperCase();
                this.children = [];
                this.dataset = {{}};
                this.eventListeners = new Map();
                this.className = '';
                this.id = '';
                this.type = '';
                this.value = '';
                this.storedText = '';
              }}

              get textContent() {{
                if (this.children.length > 0) {{
                  return this.children.map((child) => child.textContent).join('');
                }}
                return this.storedText;
              }}

              set textContent(value) {{
                this.storedText = String(value);
                this.children = [];
              }}

              set innerHTML(value) {{
                this.storedText = '';
                this.children = [];
                const raw = String(value);
                if (raw.includes('<script')) {{
                  const script = new MiniElement('script');
                  script.textContent = raw;
                  this.appendChild(script);
                  return;
                }}
                this.textContent = raw.replace(/<[^>]+>/g, '');
              }}

              appendChild(child) {{
                child.parentNode = this;
                this.children.push(child);
                return child;
              }}

              replaceChildren(...children) {{
                this.children = [];
                this.storedText = '';
                for (const child of children) {{
                  this.appendChild(child);
                }}
              }}

              remove() {{
                if (!this.parentNode) {{
                  return;
                }}
                this.parentNode.children = this.parentNode.children.filter(
                  (child) => child !== this,
                );
                this.parentNode = null;
              }}

              addEventListener(type, callback) {{
                if (!this.eventListeners.has(type)) {{
                  this.eventListeners.set(type, []);
                }}
                this.eventListeners.get(type).push(callback);
              }}

              dispatchEvent(event) {{
                for (const callback of this.eventListeners.get(event.type) || []) {{
                  callback(event);
                }}
              }}

              click() {{
                this.dispatchEvent({{ type: 'click', target: this }});
              }}

              querySelectorAll(selector) {{
                const results = [];
                const wantedTag = selector.toUpperCase();
                const visit = (node) => {{
                  if (node.nodeType !== 1) {{
                    return;
                  }}
                  if (node.tagName === wantedTag) {{
                    results.push(node);
                  }}
                  for (const child of node.children) {{
                    visit(child);
                  }}
                }};
                for (const child of this.children) {{
                  visit(child);
                }}
                return results;
              }}

              querySelector(selector) {{
                return this.querySelectorAll(selector)[0] || null;
              }}
            }}

            class MiniDocument extends MiniElement {{
              constructor() {{
                super('#document');
                this.body = new MiniElement('body');
                this.appendChild(this.body);
                this.readyListeners = [];
              }}

              createElement(tagName) {{
                return new MiniElement(tagName);
              }}

              createTextNode(text) {{
                return new MiniTextNode(text);
              }}

              addEventListener(type, callback) {{
                if (type === 'DOMContentLoaded') {{
                  this.readyListeners.push(callback);
                  return;
                }}
                super.addEventListener(type, callback);
              }}

              getElementById(id) {{
                let found = null;
                const visit = (node) => {{
                  if (found || node.nodeType !== 1) {{
                    return;
                  }}
                  if (node.id === id) {{
                    found = node;
                    return;
                  }}
                  for (const child of node.children) {{
                    visit(child);
                  }}
                }};
                visit(this);
                return found;
              }}
            }}

            function mount(document, tagName, id) {{
              const element = document.createElement(tagName);
              element.id = id;
              document.body.appendChild(element);
              return element;
            }}

            function byVisibleText(root, tagName, text) {{
              return root
                .querySelectorAll(tagName)
                .find((element) => element.textContent.includes(text));
            }}

            async function main() {{
              const response = {json.dumps(response)};
              const rendererPath = {json.dumps(str(STATIC_ROOT / "renderer.js"))};
              const document = new MiniDocument();
              mount(document, 'form', 'request-form');
              mount(document, 'textarea', 'request-input');
              mount(document, 'section', 'approval-surfaces');
              mount(document, 'ol', 'status-updates');
              mount(document, 'pre', 'artifact-list');
              mount(document, 'section', 'downstream-surfaces');

              const userActions = [];
              let artifactA2uiParts = response.a2uiParts;
              let requestA2uiParts = null;
              const directRequestParts = [
                {{
                  data: {{
                    version: '0.1',
                    createSurface: {{
                      surfaceId: 'surface_product_opportunity_fresh',
                      catalogId: 'basic',
                    }},
                  }},
                }},
                {{
                  data: {{
                    version: '0.1',
                    updateComponents: {{
                      surfaceId: 'surface_product_opportunity_fresh',
                      components: [
                        {{
                          component: 'Text',
                          id: 'root',
                          text: 'Fresh direct product surface',
                        }},
                      ],
                    }},
                  }},
                }},
              ];
              requestA2uiParts = directRequestParts;
              const approvalRefreshParts = response.a2uiParts.filter((part) => {{
                const data = part.data || {{}};
                const createSurface = data.createSurface || {{}};
                const updateComponents = data.updateComponents || {{}};
                return (
                  createSurface.surfaceId === 'surface_plan_renderer_behavior'
                  || updateComponents.surfaceId === 'surface_plan_renderer_behavior'
                );
              }});
              const fetch = async (url, options = {{}}) => {{
                if (url === '/api/request') {{
                  artifactA2uiParts = requestA2uiParts;
                  return {{
                    ok: true,
                    json: async () => ({{
                      a2uiParts: requestA2uiParts,
                      statusEvents: [],
                      artifacts: {{
                        final_response: {{ agent_id: 'product_opportunity' }},
                      }},
                    }}),
                  }};
                }}
                if (url === '/api/user-action') {{
                  const submittedAction = JSON.parse(options.body);
                  const userAction = submittedAction.userAction || {{}};
                  const isApprovalEdit = userAction.surfaceId === 'surface_plan_renderer_behavior';
                  userActions.push(submittedAction);
                  if (isApprovalEdit) {{
                    artifactA2uiParts = approvalRefreshParts;
                  }}
                  return {{
                    ok: true,
                    json: async () => ({{
                      status: isApprovalEdit ? 'draft_updated' : 'routed',
                      a2uiParts: isApprovalEdit ? approvalRefreshParts : [],
                      statusEvents: [],
                      artifacts: {{}},
                    }}),
                  }};
                }}
                if (url === '/api/status') {{
                  return {{ ok: true, json: async () => ({{ statusEvents: response.statusEvents }}) }};
                }}
                if (url === '/api/artifacts') {{
                  return {{
                    ok: true,
                    json: async () => ({{
                      artifacts: response.artifacts,
                      a2uiParts: artifactA2uiParts,
                    }}),
                  }};
                }}
                throw new Error(`Unexpected fetch ${{url}}`);
              }};

              const context = {{
                document,
                fetch,
                console,
                Map,
                Error,
                JSON,
                String,
                Array,
                Object,
                Promise,
                setImmediate,
              }};
              context.window = context;
              context.globalThis = context;
              vm.createContext(context);
              const source = fs.readFileSync(rendererPath, 'utf8')
                + '\\nthis.__rendererApi = {{ handleTransportResponse, submitRequest }};';
              vm.runInContext(source, context, {{ filename: 'renderer.js' }});

              context.__rendererApi.handleTransportResponse(response);

              const approvalText = document.getElementById('approval-surfaces').textContent;
              const downstreamText = document.getElementById('downstream-surfaces').textContent;
              const statusText = document.getElementById('status-updates').textContent;
              const artifactText = document.getElementById('artifact-list').textContent;
              assert.match(approvalText, /Approval plan/);
              assert.match(approvalText, /Prepare me for tomorrow's meeting/);
              assert.match(approvalText, /Selected agents: relationship_summary/);
              assert.match(approvalText, /step_relationship_summary/);
              assert.match(downstreamText, /Product recommendation/);
              assert.match(downstreamText, /Relationship signal/);
              assert.match(downstreamText, /Treasury services are relevant/);
              assert.match(downstreamText, /<script>globalThis.__executed = true<\\/script>/);
              assert.equal(document.querySelector('script'), null);
              assert.equal(context.__executed, undefined);
              const tableCells = document
                .getElementById('downstream-surfaces')
                .querySelectorAll('td')
                .map((cell) => cell.textContent);
              assert.deepEqual(tableCells, ['0', 'false']);
              context.__rendererApi.handleTransportResponse({{
                a2uiParts: [
                  {{
                    data: {{
                      version: '0.1',
                      updateComponents: {{
                        surfaceId: 'surface_product_opportunity_1',
                        components: [
                          {{
                            component: 'Text',
                            id: 'downstream_text',
                            text: 'Updated product recommendation',
                          }},
                        ],
                      }},
                    }},
                  }},
                ],
                statusEvents: [],
                artifacts: {{}},
              }});
              const partiallyUpdatedText = document
                .getElementById('surface_product_opportunity_1')
                .textContent;
              assert.match(partiallyUpdatedText, /Updated product recommendation/);
              assert.match(partiallyUpdatedText, /Show detail/);
              const preservedCells = document
                .getElementById('surface_product_opportunity_1')
                .querySelectorAll('td')
                .map((cell) => cell.textContent);
              assert.deepEqual(preservedCells, ['0', 'false']);

              context.__rendererApi.handleTransportResponse({{
                a2uiParts: [
                  {{
                    data: {{
                      version: '0.1',
                      createSurface: {{
                        surfaceId: 'surface_normalized_components',
                        catalogId: 'basic',
                      }},
                    }},
                  }},
                  {{
                    data: {{
                      version: '0.1',
                      updateComponents: {{
                        surfaceId: 'surface_normalized_components',
                        components: [
                          {{
                            component: 'Column',
                            id: 'root',
                            children: ['table_by_type', 'status_lower', 'conflict_by_type'],
                          }},
                          {{
                            type: 'Table',
                            id: 'table_by_type',
                            columns: [{{ key: 'fit', label: 'Fit' }}],
                            rows: [{{ fit: 'strong' }}],
                          }},
                          {{
                            component: 'status',
                            id: 'status_lower',
                            message: 'Normalized status rendered.',
                          }},
                          {{
                            type: 'Text',
                            component: 'Button',
                            id: 'conflict_by_type',
                            text: 'Conflict rendered as text.',
                            label: 'Do not dispatch',
                            action: {{
                              event: {{
                                name: 'specialist_action',
                                context: {{
                                  type: 'specialist_action',
                                  payload: {{ action: 'should_not_dispatch' }},
                                }},
                              }},
                            }},
                          }},
                        ],
                      }},
                    }},
                  }},
                ],
                statusEvents: [],
                artifacts: {{}},
              }});
              const normalizedSurface = document.getElementById('surface_normalized_components');
              assert.match(normalizedSurface.textContent, /Normalized status rendered/);
              assert.match(normalizedSurface.textContent, /Conflict rendered as text/);
              assert.equal(normalizedSurface.textContent.includes('Do not dispatch'), false);
              assert.equal(normalizedSurface.textContent.includes('Unsupported component'), false);
              assert.equal(normalizedSurface.querySelectorAll('button').length, 0);
              assert.deepEqual(
                normalizedSurface.querySelectorAll('td').map((cell) => cell.textContent),
                ['strong'],
              );
              context.__rendererApi.handleTransportResponse({{
                a2uiParts: [
                  {{
                    data: {{
                      version: '0.1',
                      createSurface: {{
                        surfaceId: 'surface_bound_renderer_data',
                        catalogId: 'basic',
                      }},
                    }},
                  }},
                  {{
                    data: {{
                      version: '0.1',
                      updateDataModel: {{
                        surfaceId: 'surface_bound_renderer_data',
                        path: '/',
                        value: {{
                          customer: {{ name: 'ABC Manufacturing' }},
                          products: [
                            {{
                              accountId: 'acct_working_capital',
                              name: 'Working capital line',
                              cardName: 'Card Working Capital',
                              accordionName: 'Accordion Working Capital',
                              buttonLabel: 'Inspect Working Capital',
                            }},
                            {{
                              accountId: 'acct_treasury',
                              name: 'Treasury services',
                              cardName: 'Card Treasury Services',
                              accordionName: 'Accordion Treasury Services',
                              buttonLabel: 'Inspect Treasury',
                            }},
                          ],
                        }},
                      }},
                    }},
                  }},
                  {{
                    data: {{
                      version: '0.1',
                      updateComponents: {{
                        surfaceId: 'surface_bound_renderer_data',
                        components: [
                          {{
                            component: 'Column',
                            id: 'root',
                            children: [
                              'bound_customer_name',
                              'product_list',
                              'product_card_list',
                              'product_accordion_list',
                              'product_edit_list',
                              'product_action_list',
                            ],
                          }},
                          {{
                            component: 'Text',
                            id: 'bound_customer_name',
                            text: {{ path: 'customer/name' }},
                          }},
                          {{
                            component: 'List',
                            id: 'product_list',
                            children: {{
                              componentId: 'product_name_template',
                              path: 'products',
                            }},
                          }},
                          {{
                            component: 'Text',
                            id: 'product_name_template',
                            text: {{ path: 'name' }},
                          }},
                          {{
                            component: 'List',
                            id: 'product_card_list',
                            children: {{
                              componentId: 'product_card_template',
                              path: 'products',
                            }},
                          }},
                          {{
                            component: 'Card',
                            id: 'product_card_template',
                            child: 'product_card_name',
                          }},
                          {{
                            component: 'Text',
                            id: 'product_card_name',
                            text: {{ path: 'cardName' }},
                          }},
                          {{
                            component: 'List',
                            id: 'product_accordion_list',
                            children: {{
                              componentId: 'product_accordion_template',
                              path: 'products',
                            }},
                          }},
                          {{
                            component: 'Accordion',
                            id: 'product_accordion_template',
                            title: 'Product detail',
                            children: ['product_accordion_name'],
                          }},
                          {{
                            component: 'Text',
                            id: 'product_accordion_name',
                            text: {{ path: 'accordionName' }},
                          }},
                          {{
                            component: 'List',
                            id: 'product_edit_list',
                            children: {{
                              componentId: 'product_name_edit_template',
                              path: 'products',
                            }},
                          }},
                          {{
                            component: 'TextField',
                            id: 'product_name_edit_template',
                            label: 'Product name',
                            value: {{ path: 'name' }},
                          }},
                          {{
                            component: 'List',
                            id: 'product_action_list',
                            children: {{
                              componentId: 'product_action_template',
                              path: 'products',
                            }},
                          }},
                          {{
                            component: 'Button',
                            id: 'product_action_template',
                            child: 'product_action_label',
                            action: {{
                              event: {{
                                name: 'specialist_action',
                                context: {{
                                  type: 'specialist_action',
                                  payload: {{
                                    action: 'inspect_product',
                                    accountId: {{ path: 'accountId' }},
                                    productName: {{ path: 'name' }},
                                  }},
                                }},
                              }},
                            }},
                          }},
                          {{
                            component: 'Text',
                            id: 'product_action_label',
                            text: {{ path: 'buttonLabel' }},
                          }},
                        ],
                      }},
                    }},
                  }},
                ],
                statusEvents: [],
                artifacts: {{}},
              }});
              const boundSurface = document.getElementById('surface_bound_renderer_data');
              assert.match(boundSurface.textContent, /ABC Manufacturing/);
              assert.match(boundSurface.textContent, /Working capital line/);
              assert.match(boundSurface.textContent, /Treasury services/);
              assert.match(boundSurface.textContent, /Card Working Capital/);
              assert.match(boundSurface.textContent, /Card Treasury Services/);
              assert.match(boundSurface.textContent, /Accordion Working Capital/);
              assert.match(boundSurface.textContent, /Accordion Treasury Services/);
              assert.match(boundSurface.textContent, /Inspect Working Capital/);
              assert.match(boundSurface.textContent, /Inspect Treasury/);
              assert.equal(boundSurface.textContent.includes('[object Object]'), false);
              const productNameInputs = boundSurface.querySelectorAll('input');
              assert.deepEqual(
                productNameInputs.map((input) => input.value),
                ['Working capital line', 'Treasury services'],
              );
              productNameInputs[1].value = 'Treasury management';
              productNameInputs[1].dispatchEvent({{ type: 'input', target: productNameInputs[1] }});
              context.__rendererApi.handleTransportResponse({{
                a2uiParts: [
                  {{
                    data: {{
                      version: '0.1',
                      updateDataModel: {{
                        surfaceId: 'surface_bound_renderer_data',
                        path: '/customer/name',
                        value: 'XYZ Supplies',
                      }},
                    }},
                  }},
                ],
                statusEvents: [],
                artifacts: {{}},
              }});
              const reboundSurfaceText = document
                .getElementById('surface_bound_renderer_data')
                .textContent;
              assert.match(reboundSurfaceText, /XYZ Supplies/);
              assert.match(reboundSurfaceText, /Working capital line/);
              assert.match(reboundSurfaceText, /Treasury management/);
              assert.match(reboundSurfaceText, /Card Working Capital/);
              assert.match(reboundSurfaceText, /Accordion Treasury Services/);
              assert.equal(reboundSurfaceText.includes('ABC Manufacturing'), false);
              context.__rendererApi.handleTransportResponse({{
                a2uiParts: [
                  {{
                    data: {{
                      version: '0.1',
                      updateDataModel: {{
                        surfaceId: 'surface_bound_renderer_data',
                        path: '/customer/name',
                      }},
                    }},
                  }},
                ],
                statusEvents: [],
                artifacts: {{}},
              }});
              const deletedValueSurfaceText = document
                .getElementById('surface_bound_renderer_data')
                .textContent;
              assert.equal(deletedValueSurfaceText.includes('XYZ Supplies'), false);
              assert.match(deletedValueSurfaceText, /Working capital line/);
              assert.match(statusText, /approval_required: Plan approval is pending/);
              assert.match(statusText, /step_completed: Relationship summary completed/);
              assert.match(artifactText, /Draft meeting brief ready/);

              const approvalRoot = document.getElementById('approval-surfaces');
              const instructionInput = approvalRoot
                .querySelectorAll('textarea')
                .find((element) => (
                  element.parentNode.textContent.includes(
                    'Instruction for step_relationship_summary',
                  )
                ));
              assert.ok(instructionInput);
              instructionInput.value = 'Prioritize covenant follow-ups.';
              instructionInput.dispatchEvent({{ type: 'input', target: instructionInput }});
              const addInstruction = byVisibleText(
                approvalRoot,
                'button',
                'Add instruction to step_relationship_summary',
              );
              assert.ok(addInstruction);
              addInstruction.click();
              await new Promise((resolve) => setImmediate(resolve));

              assert.equal(userActions.length, 1);
              assert.deepEqual(userActions[0], {{
                userAction: {{
                  type: 'add_instruction',
                  surfaceId: 'surface_plan_renderer_behavior',
                  payload: {{
                    planId: 'plan_renderer_behavior',
                    planVersion: 3,
                    editedPlanVersion: 3,
                    stepId: 'step_relationship_summary',
                    instruction: 'Prioritize covenant follow-ups.',
                  }},
                }},
              }});
              const refreshedInstructionInput = approvalRoot
                .querySelectorAll('textarea')
                .find((element) => (
                  element.parentNode.textContent.includes(
                    'Instruction for step_relationship_summary',
                  )
                ));
              assert.ok(refreshedInstructionInput);
              assert.equal(refreshedInstructionInput.value, '');

              const downstreamRoot = document.getElementById('downstream-surfaces');
              const showDetail = byVisibleText(downstreamRoot, 'button', 'Show detail');
              assert.ok(showDetail);
              showDetail.click();
              await new Promise((resolve) => setImmediate(resolve));

              assert.equal(userActions.length, 2);
              assert.deepEqual(userActions[1], {{
                userAction: {{
                  type: 'specialist_action',
                  surfaceId: 'surface_product_opportunity_1',
                  payload: {{
                    action: 'show_more_detail',
                    path: 'synthetic://product-opportunity/source',
                  }},
                }},
              }});
              assert.ok(document.getElementById('surface_product_opportunity_1'));

              const rowScopedAction = byVisibleText(
                document.getElementById('surface_bound_renderer_data'),
                'button',
                'Inspect Treasury',
              );
              assert.ok(rowScopedAction);
              rowScopedAction.click();
              await new Promise((resolve) => setImmediate(resolve));

              assert.equal(userActions.length, 3);
              assert.deepEqual(userActions[2], {{
                userAction: {{
                  type: 'specialist_action',
                  surfaceId: 'surface_bound_renderer_data',
                  payload: {{
                    action: 'inspect_product',
                    accountId: 'acct_treasury',
                    productName: 'Treasury management',
                  }},
                }},
              }});
              assert.ok(document.getElementById('surface_product_opportunity_1'));
              assert.ok(document.getElementById('surface_bound_renderer_data'));

              document.getElementById('request-input').value = 'new direct request';
              await context.__rendererApi.submitRequest({{
                preventDefault() {{}},
              }});
              await new Promise((resolve) => setImmediate(resolve));

              assert.equal(document.getElementById('surface_plan_renderer_behavior'), null);
              assert.equal(document.getElementById('surface_product_opportunity_1'), null);
              const freshSurface = document.getElementById('surface_product_opportunity_fresh');
              assert.ok(freshSurface);
              assert.match(freshSurface.textContent, /Fresh direct product surface/);

              artifactA2uiParts = [];
              context.__rendererApi.handleTransportResponse({{
                a2uiParts: [],
                statusEvents: [],
                artifacts: {{}},
              }});
              await new Promise((resolve) => setImmediate(resolve));

              assert.equal(document.getElementById('surface_plan_renderer_behavior'), null);
              assert.equal(document.getElementById('surface_product_opportunity_1'), null);
              assert.ok(document.getElementById('surface_product_opportunity_fresh'));

              requestA2uiParts = [];
              document.getElementById('request-input').value = 'new no-ui request';
              await context.__rendererApi.submitRequest({{
                preventDefault() {{}},
              }});
              await new Promise((resolve) => setImmediate(resolve));

              assert.equal(document.getElementById('surface_product_opportunity_fresh'), null);
              assert.equal(document.getElementById('approval-surfaces').textContent, '');
              assert.equal(document.getElementById('downstream-surfaces').textContent, '');
            }}

            main().catch((error) => {{
              console.error(error);
              process.exit(1);
            }});
            """
        ),
        encoding="utf-8",
    )

    # Act
    completed = subprocess.run(
        ["node", str(node_script)],
        check=False,
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
    )

    # Assert
    assert completed.returncode == 0, completed.stderr


def test_renderer_maps_basic_catalog_payloads_to_trusted_components() -> None:
    # Arrange
    script = (STATIC_ROOT / "renderer.js").read_text(encoding="utf-8")

    # Act
    supported_components = {
        "Column",
        "Row",
        "List",
        "Text",
        "Button",
        "TextField",
        "Card",
        "Table",
        "Accordion",
        "Timeline",
        "Status",
    }

    # Assert
    assert "const COMPONENT_RENDERERS" in script
    for component in supported_components:
        assert f"{component}:" in script
    assert "document.createElement" in script
    assert ".textContent" in script
    assert "renderUnsupportedComponent" in script


def test_renderer_emits_structured_user_actions_with_surface_and_plan_metadata() -> (
    None
):
    # Arrange
    script = (STATIC_ROOT / "renderer.js").read_text(encoding="utf-8")

    # Act / Assert
    assert "function buildUserAction" in script
    assert "type: context.type" in script
    assert "surfaceId," in script
    assert "payload: resolvePayload(context.payload || {})" in script
    assert "planId" in script
    assert "planVersion" in script
    assert "editedPlanVersion" in script


def test_renderer_does_not_execute_agent_provided_code_or_html() -> None:
    # Arrange
    script = (STATIC_ROOT / "renderer.js").read_text(encoding="utf-8")

    # Act
    unsafe_tokens = [
        ".innerHTML",
        ".outerHTML",
        "insertAdjacentHTML",
        "eval(",
        "new Function",
        "setAttribute('onclick'",
        'setAttribute("onclick"',
    ]

    # Assert
    for token in unsafe_tokens:
        assert token not in script
    assert ".textContent" in script
    assert "addEventListener('click'" in script
